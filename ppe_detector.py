
import os
import ast
import cv2
import json
import datetime
import base64
import logging
import time
import numpy as np
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from ultralytics import YOLO
import paho.mqtt.client as mqtt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

local_tz = ZoneInfo("Asia/Jakarta")

load_dotenv('.env')
device_id = os.getenv('DEVICE_ID')
device_code = os.getenv('DEVICE_CODE')
device_name = os.getenv('DEVICE_NAME')

MQTT_BROKER = os.getenv('MQTT_BROKER', 'localhost')
MQTT_PORT = int(os.getenv('MQTT_PORT', '1883'))
MQTT_USERNAME = os.getenv('MQTT_USERNAME')
MQTT_PASSWORD = os.getenv('MQTT_PASSWORD')
PPE_DETECTION_TOPIC = os.getenv('PPE_DETECTION_TOPIC', '/ppe_detection')

YOLO_CONFIDENCE = float(os.getenv('YOLO_CONFIDENCE', '0.5'))
YOLO_DEVICE = os.getenv('YOLO_DEVICE', 'auto')

# Send interval per track_id (seconds) — prevents burst on the same person
PPE_SEND_INTERVAL = float(os.getenv('PPE_SEND_INTERVAL', '2.0'))

DEBUG_MODE = os.getenv('DEBUG_MODE', 'true').lower() == 'true'

# Video stream settings
RTSP_URL = os.getenv('RTSP_URL')
resolution = ast.literal_eval(os.getenv('SCREEN_RESOLUTION', '[800, 600]'))
ENABLE_NVDEC = os.getenv('ENABLE_NVDEC', 'false').lower() == 'true'
FRAME_SKIP = int(os.getenv('FRAME_SKIP', '2'))
FPS_LIMIT = float(os.getenv('FPS_LIMIT', '0'))
FRAME_INTERVAL = 1.0 / FPS_LIMIT if FPS_LIMIT > 0 else 0

CROP_PADDING = int(os.getenv('CROP_PADDING', '30'))
MIN_CROP_SIZE = (128, 128)
JPEG_QUALITY = int(os.getenv('JPEG_QUALITY', '70'))

# PPE class definitions from ppe.pt
PPE_POSITIVE_CLASSES = {'helmet', 'gloves', 'vest', 'boots', 'goggles'}
PPE_NEGATIVE_CLASSES = {'no_helmet', 'no_goggle', 'no_gloves', 'no_boots'}
PPE_NEGATIVE_TO_ITEM = {
    'no_helmet': 'helmet',
    'no_goggle': 'goggles',
    'no_gloves': 'gloves',
    'no_boots': 'boots',
}
PPE_SKIP_CLASSES = {'none', 'Person'}

mqtt_client = None

# Per-track_id last send timestamps for rate limiting
last_send_times: dict = {}
is_midnight: bool = False


def resolve_yolo_device(device_str):
    if device_str.lower() == 'auto':
        import torch
        if torch.cuda.is_available():
            resolved = '0'
            logging.info(f"YOLO_DEVICE=auto: CUDA available, using GPU 0 ({torch.cuda.get_device_name(0)})")
        else:
            resolved = 'cpu'
            logging.info("YOLO_DEVICE=auto: No CUDA available, falling back to CPU")
        return resolved
    logging.info(f"YOLO_DEVICE set to: {device_str}")
    return device_str


def init_mqtt():
    global mqtt_client
    try:
        mqtt_client = mqtt.Client()
        if MQTT_USERNAME and MQTT_PASSWORD:
            mqtt_client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                logging.info("Connected to MQTT broker successfully")
            else:
                logging.error(f"Failed to connect to MQTT broker, return code {rc}")

        def on_disconnect(client, userdata, rc):
            logging.warning("Disconnected from MQTT broker")

        mqtt_client.on_connect = on_connect
        mqtt_client.on_disconnect = on_disconnect
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()
    except Exception as e:
        logging.error(f"Failed to initialize MQTT: {e}")
        mqtt_client = None


# ─── Video capture utilities ─────────────────────────────────────────────────

def validate_cctv_connection(rtsp_url, timeout=5):
    if not rtsp_url or rtsp_url.strip() == '':
        logging.warning("RTSP_URL is empty or not set")
        return False
    try:
        logging.info(f"Validating CCTV connection: {rtsp_url}")
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            cap.release()
            return False
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            logging.info("CCTV connection validated successfully")
            return True
        else:
            logging.warning("CCTV stream opened but failed to read frame")
            return False
    except Exception as e:
        logging.warning(f"CCTV validation error: {e}")
        return False


def initialize_video_capture(video_source):
    logging.info(f'Initializing video capture with source: {video_source}')
    if ENABLE_NVDEC:
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'hwaccel;cuda|video_codec;h264_cuvid|rtsp_transport;tcp'
        logging.info('NVDEC hardware decoding enabled (h264_cuvid)')
    if isinstance(video_source, int):
        logging.info(f"Opening webcam device {video_source}")
        cap = cv2.VideoCapture(video_source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 10)
        return cap
    cap = cv2.VideoCapture(video_source, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 10)
    return cap


def get_video_source():
    fallback_video = '1.mp4'
    if not RTSP_URL or RTSP_URL.strip() == '':
        logging.warning(f"RTSP_URL not set. Falling back to {fallback_video}")
        return fallback_video
    if RTSP_URL.strip().lstrip('-').isdigit():
        device_index = int(RTSP_URL.strip())
        logging.info(f"RTSP_URL={RTSP_URL} detected as webcam device index {device_index}")
        return device_index
    if validate_cctv_connection(RTSP_URL):
        logging.info("Using CCTV stream as video source")
        return RTSP_URL
    else:
        logging.warning(f"CCTV connection failed. Falling back to {fallback_video}")
        if os.path.exists(fallback_video):
            return fallback_video
        raise FileNotFoundError(f"Neither CCTV stream nor fallback video ({fallback_video}) is available")


def should_reset():
    now = datetime.datetime.now(local_tz)
    return now.hour == 0 and now.minute == 0 and now.second < 10


# ─── Image utilities ─────────────────────────────────────────────────────────

def crop_image(frame, box, padding=None):
    if padding is None:
        padding = CROP_PADDING
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]
    x1c = max(0, x1 - padding)
    y1c = max(0, y1 - padding)
    x2c = min(w, x2 + padding)
    y2c = min(h, y2 + padding)
    crop = frame[y1c:y2c, x1c:x2c]
    ch, cw = crop.shape[:2]
    if ch < MIN_CROP_SIZE[1] or cw < MIN_CROP_SIZE[0]:
        aspect = cw / ch if ch > 0 else 1
        if aspect > 1:
            new_w = max(MIN_CROP_SIZE[0], cw)
            new_h = int(new_w / aspect)
            if new_h < MIN_CROP_SIZE[1]:
                new_h = MIN_CROP_SIZE[1]
                new_w = int(new_h * aspect)
        else:
            new_h = max(MIN_CROP_SIZE[1], ch)
            new_w = int(new_h * aspect)
            if new_w < MIN_CROP_SIZE[0]:
                new_w = MIN_CROP_SIZE[0]
                new_h = int(new_w / aspect)
        crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    return crop


def encode_image(image):
    """Encode numpy image to base64 JPEG string."""
    try:
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        return base64.b64encode(buffer.tobytes()).decode('utf-8')
    except Exception as e:
        logging.error(f"Image encode error: {e}")
        return None


# ─── PPE inference utilities ─────────────────────────────────────────────────

def determine_compliance(detections):
    ppe_present = sorted({d["class"] for d in detections if d["class"] in PPE_POSITIVE_CLASSES})
    ppe_missing = sorted({
        PPE_NEGATIVE_TO_ITEM[d["class"]]
        for d in detections
        if d["class"] in PPE_NEGATIVE_CLASSES
    })
    compliant = len(ppe_missing) == 0 and len(ppe_present) > 0
    return compliant, ppe_present, ppe_missing


def should_send_for_track(track_id):
    """Rate limit: allow one send per track_id per PPE_SEND_INTERVAL seconds."""
    now = time.time()
    if now - last_send_times.get(track_id, 0) >= PPE_SEND_INTERVAL:
        last_send_times[track_id] = now
        return True
    return False


def publish_ppe_result(track_id, detections, compliance, ppe_present, ppe_missing, image=None):
    """Publish PPE result to MQTT including cropped person image."""
    if mqtt_client is None:
        return
    try:
        payload = {
            "track_id": track_id,
            "device_id": device_id,
            "device_code": device_code,
            "device_name": device_name,
            "timestamp": datetime.datetime.now(local_tz).isoformat(),
            "compliance": compliance,
            "detections": [{"class": d["class"], "confidence": d["confidence"]} for d in detections],
            "ppe_present": ppe_present,
            "ppe_missing": ppe_missing,
        }
        if image is not None:
            b64 = encode_image(image)
            if b64:
                payload["image"] = b64

        result = mqtt_client.publish(PPE_DETECTION_TOPIC, json.dumps(payload), qos=1)
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            status = "COMPLIANT" if compliance else "VIOLATION"
            logging.info(
                f"PPE [{status}] track_id={track_id} "
                f"present={ppe_present} missing={ppe_missing}"
            )
        else:
            logging.error(f"Failed to publish PPE result for track_id={track_id}, rc={result.rc}")
    except Exception as e:
        logging.error(f"Error publishing PPE result: {e}")


# ─── Main PPE video loop ──────────────────────────────────────────────────────

def run_ppe_loop(model, resolved_device):
    global is_midnight, last_send_times

    try:
        video_source = get_video_source()
        logging.info(f"PPE video source: {video_source}")
    except FileNotFoundError as e:
        logging.error(f"Fatal: {e}")
        return

    if DEBUG_MODE:
        cv2.namedWindow('PPE_Detector')

    while True:
        try:
            cap = initialize_video_capture(video_source)
            if not cap.isOpened():
                raise Exception(f"Failed to open video source: {video_source}")
            logging.info("PPE detection loop started")

            count = 0
            last_process_time = time.time()
            fps_counter = 0
            fps_timer = time.time()

            while True:
                count += 1
                if count % FRAME_SKIP != 0:
                    cap.grab()
                    continue

                ret, frame = cap.read()

                if FRAME_INTERVAL > 0:
                    elapsed = time.time() - last_process_time
                    if elapsed < FRAME_INTERVAL:
                        time.sleep(FRAME_INTERVAL - elapsed)
                    last_process_time = time.time()

                if not ret:
                    raise Exception(f"Frame read error: {video_source}")

                frame = cv2.resize(frame, (resolution[0], resolution[1]))

                results = model.track(
                    frame,
                    persist=True,
                    verbose=False,
                    conf=YOLO_CONFIDENCE,
                    device=resolved_device,
                )

                if results[0].boxes is not None:
                    boxes = results[0].boxes.xyxy.int().cpu().tolist()
                    class_ids = results[0].boxes.cls.int().cpu().tolist()
                    confidences = results[0].boxes.conf.cpu().tolist()
                    track_ids = (
                        results[0].boxes.id.int().cpu().tolist()
                        if results[0].boxes.id is not None
                        else [None] * len(boxes)
                    )
                    names = model.names

                    # Split into persons vs PPE items
                    person_boxes = []
                    ppe_boxes = []
                    for box, cid, conf, tid in zip(boxes, class_ids, confidences, track_ids):
                        class_name = names[cid]
                        if class_name == 'Person':
                            person_boxes.append((box, tid))
                        elif class_name not in PPE_SKIP_CLASSES:
                            ppe_boxes.append((box, class_name, conf))

                    for (px1, py1, px2, py2), tid in person_boxes:
                        # Associate PPE boxes whose center lies within this person's box
                        person_ppe = []
                        for (ex1, ey1, ex2, ey2), cls, conf in ppe_boxes:
                            cx, cy = (ex1 + ex2) // 2, (ey1 + ey2) // 2
                            if px1 <= cx <= px2 and py1 <= cy <= py2:
                                person_ppe.append({"class": cls, "confidence": conf})

                        compliance, ppe_present, ppe_missing = determine_compliance(person_ppe)

                        # Send on any person detection (compliant or violation), rate-limited
                        if should_send_for_track(tid):
                            person_crop = crop_image(frame, [px1, py1, px2, py2])
                            publish_ppe_result(
                                track_id=tid,
                                detections=person_ppe,
                                compliance=compliance,
                                ppe_present=ppe_present,
                                ppe_missing=ppe_missing,
                                image=person_crop,
                            )

                        if DEBUG_MODE:
                            color = (0, 255, 0) if compliance else (0, 0, 255)
                            cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)
                            status = "OK" if compliance else "VIOLATION"
                            cv2.putText(frame, f'ID:{tid} {status}',
                                        (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            if ppe_missing:
                                cv2.putText(frame, f'Missing: {", ".join(ppe_missing)}',
                                            (px1, py2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                            if ppe_present:
                                cv2.putText(frame, f'Has: {", ".join(ppe_present)}',
                                            (px1, py2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

                    if DEBUG_MODE:
                        # Draw PPE item boxes
                        for (ex1, ey1, ex2, ey2), cls, conf in ppe_boxes:
                            box_color = (0, 165, 255) if cls in PPE_POSITIVE_CLASSES else (0, 0, 200)
                            cv2.rectangle(frame, (ex1, ey1), (ex2, ey2), box_color, 1)
                            cv2.putText(frame, f'{cls} {conf:.2f}',
                                        (ex1, ey1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)

                if DEBUG_MODE:
                    cv2.imshow('PPE_Detector', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logging.info("User requested exit")
                        return

                fps_counter += 1
                if time.time() - fps_timer >= 10.0:
                    actual_fps = fps_counter / (time.time() - fps_timer)
                    logging.info(f"PPE FPS: {actual_fps:.1f} | Tracked persons: {len(person_boxes) if results[0].boxes is not None else 0}")
                    fps_counter = 0
                    fps_timer = time.time()

                if should_reset() and not is_midnight:
                    last_send_times.clear()
                    is_midnight = True
                    logging.info("== Midnight Reached: last_send_times reset ==")
                if not should_reset() and is_midnight:
                    is_midnight = False

        except Exception as error:
            logging.error(f"Error: {error}. Restarting in 5 seconds...")
            if 'cap' in locals():
                cap.release()
            time.sleep(5)
            continue

    if DEBUG_MODE:
        cv2.destroyAllWindows()


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    resolved_device = resolve_yolo_device(YOLO_DEVICE)
    model = YOLO('ppe.pt')
    logging.info(f"PPE model loaded. Classes: {model.names}")
    logging.info(f"Publishing to '{PPE_DETECTION_TOPIC}' | send interval: {PPE_SEND_INTERVAL}s | DEBUG_MODE: {DEBUG_MODE}")

    init_mqtt()

    try:
        run_ppe_loop(model, resolved_device)
    finally:
        cv2.destroyAllWindows()
        if mqtt_client:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()


if __name__ == '__main__':
    main()
