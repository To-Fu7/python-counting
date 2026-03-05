
import os
import cv2
import json
import datetime
import base64
import logging
import time
import ast
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
PERSON_DETECTION_TOPIC = os.getenv('PERSON_DETECTION_TOPIC', '/person_detection')

RTSP_URL = os.getenv('RTSP_URL')
ENABLE_NVDEC = os.getenv('ENABLE_NVDEC', 'false').lower() == 'true'
resolution = ast.literal_eval(os.getenv('SCREEN_RESOLUTION', '[800, 600]'))

YOLO_CONFIDENCE = float(os.getenv('YOLO_CONFIDENCE', '0.5'))
YOLO_DEVICE = os.getenv('YOLO_DEVICE', 'auto')

JPEG_QUALITY = int(os.getenv('JPEG_QUALITY', '70'))
CROP_PADDING = int(os.getenv('CROP_PADDING', '30'))
MIN_CROP_SIZE = (128, 128)

FPS_LIMIT = float(os.getenv('FPS_LIMIT', '0'))
FRAME_INTERVAL = 1.0 / FPS_LIMIT if FPS_LIMIT > 0 else 0
FRAME_SKIP = int(os.getenv('FRAME_SKIP', '2'))

DEBUG_MODE = os.getenv('DEBUG_MODE', 'true').lower() == 'true'

# Session state
seen_track_ids: set = set()
is_midnight: bool = False

mqtt_client = None


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


def crop_image(frame, box, padding=None):
    if padding is None:
        padding = CROP_PADDING
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]
    x1_crop = max(0, x1 - padding)
    y1_crop = max(0, y1 - padding)
    x2_crop = min(w, x2 + padding)
    y2_crop = min(h, y2 + padding)
    person_crop = frame[y1_crop:y2_crop, x1_crop:x2_crop]
    crop_h, crop_w = person_crop.shape[:2]
    if crop_h < MIN_CROP_SIZE[1] or crop_w < MIN_CROP_SIZE[0]:
        aspect_ratio = crop_w / crop_h if crop_h > 0 else 1
        if aspect_ratio > 1:
            new_w = max(MIN_CROP_SIZE[0], crop_w)
            new_h = int(new_w / aspect_ratio)
            if new_h < MIN_CROP_SIZE[1]:
                new_h = MIN_CROP_SIZE[1]
                new_w = int(new_h * aspect_ratio)
        else:
            new_h = max(MIN_CROP_SIZE[1], crop_h)
            new_w = int(new_h * aspect_ratio)
            if new_w < MIN_CROP_SIZE[0]:
                new_w = MIN_CROP_SIZE[0]
                new_h = int(new_w / aspect_ratio)
        person_crop = cv2.resize(person_crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    return person_crop


def should_reset():
    now = datetime.datetime.now(local_tz)
    return now.hour == 0 and now.minute == 0 and now.second < 10


def reset_seen_ids():
    global seen_track_ids
    seen_track_ids = set()
    logging.info("== Midnight Reached: seen_track_ids reset ==")


def send_person_detection(cropped_image, track_id, box, confidence):
    global mqtt_client
    if mqtt_client is None:
        logging.warning("MQTT client not initialized, skipping person detection send")
        return
    try:
        x1, y1, x2, y2 = box
        _, buffer = cv2.imencode('.jpg', cropped_image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        image_bytes = buffer.tobytes()
        payload = {
            "track_id": track_id,
            "device_id": device_id,
            "device_code": device_code,
            "device_name": device_name,
            "timestamp": datetime.datetime.now(local_tz).isoformat(),
            "image": base64.b64encode(image_bytes).decode('utf-8'),
            "bbox": {
                "x": int(x1),
                "y": int(y1),
                "w": int(x2 - x1),
                "h": int(y2 - y1),
                "center_x": int((x1 + x2) // 2),
                "center_y": int((y1 + y2) // 2),
            },
            "confidence": round(float(confidence), 4),
        }
        result = mqtt_client.publish(PERSON_DETECTION_TOPIC, json.dumps(payload), qos=1)
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            logging.info(f"Person detected: track_id={track_id} sent to {PERSON_DETECTION_TOPIC}")
        else:
            logging.error(f"Failed to send MQTT message for track_id={track_id}, rc={result.rc}")
    except Exception as e:
        logging.error(f"Error sending person detection MQTT: {e}")


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print([x, y])


def main():
    global is_midnight

    init_mqtt()

    model = YOLO('yolo11n.pt')
    resolved_device = resolve_yolo_device(YOLO_DEVICE)

    try:
        video_source = get_video_source()
        logging.info(f"Video source: {video_source}")
    except FileNotFoundError as e:
        logging.error(f"Fatal: {e}")
        return

    if DEBUG_MODE:
        cv2.namedWindow('PersonDetector')
        cv2.setMouseCallback('PersonDetector', RGB)

    last_waiting_log = time.time()

    while True:
        try:
            logging.info('Initializing PersonDetector service...')
            cap = initialize_video_capture(video_source)
            if not cap.isOpened():
                raise Exception(f"Failed to open video source: {video_source}")
            logging.info(f"Video source opened: {video_source}")

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
                    classes=[0],  # COCO class 0 = person
                )

                person_detected = False

                if results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.int().cpu().tolist()
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    confidences = results[0].boxes.conf.cpu().tolist()

                    for box, track_id, conf in zip(boxes, track_ids, confidences):
                        person_detected = True
                        x1, y1, x2, y2 = box

                        if DEBUG_MODE:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f'ID:{track_id} {"NEW" if track_id not in seen_track_ids else ""}'
                            cv2.putText(frame, label, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        if track_id not in seen_track_ids:
                            seen_track_ids.add(track_id)
                            cropped = crop_image(frame, box)
                            send_person_detection(cropped, track_id, box, conf)

                if not person_detected:
                    current_time = time.time()
                    if current_time - last_waiting_log >= 60:
                        logging.info("Waiting for person detection...")
                        last_waiting_log = current_time

                if DEBUG_MODE:
                    cv2.putText(frame, f'Seen IDs: {len(seen_track_ids)}',
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.imshow('PersonDetector', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logging.info("User requested exit")
                        return

                fps_counter += 1
                if time.time() - fps_timer >= 10.0:
                    actual_fps = fps_counter / (time.time() - fps_timer)
                    logging.info(f"Processing FPS: {actual_fps:.1f} | Seen IDs: {len(seen_track_ids)}")
                    fps_counter = 0
                    fps_timer = time.time()

                if should_reset() and not is_midnight:
                    reset_seen_ids()
                    is_midnight = True
                if not should_reset() and is_midnight:
                    is_midnight = False

        except Exception as error:
            logging.error(f"Error: {error}. Restarting in 5 seconds...")
            if 'cap' in locals():
                cap.release()
            cv2.destroyAllWindows()
            time.sleep(5)
            continue

    if 'cap' in locals():
        cap.release()
    cv2.destroyAllWindows()
    if mqtt_client:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()


if __name__ == '__main__':
    main()
