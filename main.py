import os
import cv2
import json
import time
import datetime
import psycopg2
import uuid
from ultralytics import YOLO
from collections import defaultdict
from zoneinfo import ZoneInfo
import paho.mqtt.client as mqtt
from dotenv import load_dotenv


# --- Load environment ---
load_dotenv()
MQTT_HOST = os.getenv('MQTT_HOST')
MQTT_PORT = int(os.getenv('MQTT_PORT'))
MQTT_TOPIC = os.getenv('MQTT_TOPIC')
PG_HOST = os.getenv('PG_HOST')
PG_PORT = int(os.getenv('PG_PORT'))
PG_DB = os.getenv('PG_DB')
PG_USER = os.getenv('PG_USER')
PG_PASS = os.getenv('PG_PASS')
INTERVAL = int(os.getenv('SEC_INTERVAL'))
RTSP_URL = os.getenv('RTSP_URL')
LINE_Y = int(os.getenv('LINE_Y'))
last_printed_second = -1



# --- MQTT Auto-Reconnect Setup ---
mqtt_client = mqtt.Client(client_id="yolo-vehicle-counter")
mqtt_connected = False
def on_connect(client, userdata, flags, rc):
    global mqtt_connected
    if rc == 0:
        print("Connected to MQTT broker.")
        mqtt_connected = True
    else:
        print(f"Failed to connect to MQTT broker. RC: {rc}")
        mqtt_connected = False

def on_disconnect(client, userdata, rc):
    global mqtt_connected
    print("MQTT disconnected! Will attempt reconnect.")
    mqtt_connected = False

mqtt_client.on_connect = on_connect
mqtt_client.on_disconnect = on_disconnect

# Initial connect MQTT
mqtt_client.connect_async(MQTT_HOST, MQTT_PORT)
mqtt_client.loop_start()

# --- Helper for MQTT publish with retry ---
def mqtt_publish_safe(topic, payload, qos=1, max_retry=5):
    retry = 0
    while retry < max_retry:
        if mqtt_connected:
            try:
                mqtt_client.publish(topic, payload, qos=qos)
                return True
            except Exception as e:
                print(f"MQTT publish failed: {e}")
        else:
            print("MQTT not connected, waiting to retry...")
        retry += 1
        time.sleep(2)
    print("MQTT publish failed after max retries.")
    return False

# --- PostgreSQL Auto-Reconnect Setup ---
def db_connect():
    return psycopg2.connect(
        dbname=PG_DB,
        user=PG_USER,
        password=PG_PASS,
        host=PG_HOST,
        port=PG_PORT
    )

# Create initial connection and cursor
def db_get_cursor():
    try:
        conn = db_connect()
        return conn, conn.cursor()
    except Exception as e:
        print(f"Database connection failed: {e}")
        return None, None

pg_conn, cursor = db_get_cursor()
if not cursor:
    print("Fatal: Could not connect to DB at startup.")
    exit(1)
print("----- Successfully connected to PostgreSQL database -----.")    

# Reconnect DB on failure
def db_execute_safe(sql, params=(), commit=False, max_retry=3):
    global pg_conn, cursor
    retry = 0
    while retry < max_retry:
        try:
            if cursor is None:
                print("DB cursor is None, attempting to reconnect...")
                pg_conn, cursor = db_get_cursor()
                if cursor is None:
                    print(f"Reconnection to DB failed (retry {retry+1}/{max_retry})")
                    retry += 1
                    time.sleep(2)
                    continue

            cursor.execute(sql, params)
            if commit:
                pg_conn.commit()
            return True

        except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
            print(f"Database lost connection: {e} (retry {retry+1}/{max_retry})")
            try:
                if cursor:
                    cursor.close()
                if pg_conn:
                    pg_conn.close()
            except:
                pass
            cursor = None
            pg_conn = None
            retry += 1
            time.sleep(2)

        except Exception as e:
            print(f"DB Error (not connection): {e}")
            return False

    print("Max retries reached. Entering persistent reconnect mode.")
    while True:
        pg_conn, cursor = db_get_cursor()
        if cursor:
            try:
                cursor.execute(sql, params)
                if commit:
                    pg_conn.commit()
                print("Recovered DB connection after persistent retry.")
                return True
            except Exception as e:
                print(f"Persistent retry DB error: {e}")
        time.sleep(5)


def db_fetchone_safe(sql, params=(), max_retry=3):
    global pg_conn, cursor
    retry = 0
    while retry < max_retry:
        try:
            if cursor is None:
                print("DB cursor is None, attempting to reconnect...")
                pg_conn, cursor = db_get_cursor()
                if cursor is None:
                    print(f"Reconnection to DB failed (retry {retry+1}/{max_retry})")
                    retry += 1
                    time.sleep(2)
                    continue

            cursor.execute(sql, params)
            return cursor.fetchone()

        except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
            print(f"Database lost connection: {e} (retry {retry+1}/{max_retry})")
            try:
                if cursor:
                    cursor.close()
                if pg_conn:
                    pg_conn.close()
            except:
                pass
            cursor = None
            pg_conn = None
            retry += 1
            time.sleep(2)

        except Exception as e:
            print(f"DB Error (not connection): {e}")
            return None

    # Persistent retry loop
    print("Max retries reached. Entering persistent reconnect mode.")
    while True:
        pg_conn, cursor = db_get_cursor()
        if cursor:
            try:
                cursor.execute(sql, params)
                return cursor.fetchone()
            except Exception as e:
                print(f"Persistent retry DB error: {e}")
        time.sleep(5)


# --- Globals, YOLO, etc ---
interval_counts = defaultdict(int)
class_counts = defaultdict(int)
crossed_ids = set()
last_positions = {}
allowed_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
red_line = LINE_Y
device_id = "20b71363-6208-4b64-8337-7ef6658c97d4"
model = YOLO('yolo11m.pt')
class_list = model.names
local_tz = ZoneInfo("Asia/Jakarta")

# --- Restore from DB ---
def get_latest_counts(device_id):
    row = db_fetchone_safe(
        "SELECT data, created_at FROM vehicle_countings WHERE device_id = %s ORDER BY created_at DESC LIMIT 1",
        (device_id,))
    if not row:
        return None, None
    last_data_json, last_created_utc = row
    last_created_local = last_created_utc.astimezone(local_tz)
    last_date_local = last_created_local.date()
    today_local = datetime.datetime.now(local_tz).date()
    if last_date_local == today_local:
        if isinstance(last_data_json, dict):
            return last_data_json, last_date_local
        else:
            return json.loads(last_data_json), last_date_local
    return None, last_date_local

restored_counts, last_date_local = get_latest_counts(device_id)
if restored_counts:
    for key, value in restored_counts.items():
        class_counts[key] = value
    print(f"Restored counts from DB ({last_date_local}):", class_counts)
else:
    print("No valid counts to restore, starting from zero.")

# --- RTSP Stream ---
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps != fps:
    fps = 30
frame_idx = 0
interval_frames = int(fps * INTERVAL)
reset_done_for_today = False

def should_reset():
    now = datetime.datetime.now(local_tz)
    return now.hour == 0 and now.minute == 0 and now.second < 10

def reset_counts():
    class_counts.clear()
    crossed_ids.clear()
    interval_counts.clear()
    print(f"\n== Midnight Reached (UTC+7): Resetting Totals ==")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            print("xxxxx Frame grab failed - resetting stream xxxxx")
            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps != fps:
                fps = 30
            interval_frames = int(fps * INTERVAL)
            continue

        results = model.track(frame, persist=True, verbose=False)
        cv2.line(frame, (20, red_line), (frame.shape[1] - 20, red_line), (0, 0, 255), 2)

        if results[0].boxes.data is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                if class_id not in allowed_classes:
                    continue
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                class_name = class_list[class_id]
                id_class_tuple = (track_id, class_name)
                prev_cy = last_positions.get(track_id)
                last_positions[track_id] = cy
                if prev_cy is not None and prev_cy <= red_line < cy and id_class_tuple not in crossed_ids:
                    crossed_ids.add(id_class_tuple)
                    class_counts[class_name] += 1
                    interval_counts[class_name] += 1

        frame_idx += 1
        
        current_second = int(frame_idx // fps)
        
        if current_second != last_printed_second:
            last_printed_second = current_second
            payload = {class_list[cid]: class_counts[class_list[cid]] for cid in allowed_classes}
            print(f"[Second {current_second}] Counts: {payload}")

        if should_reset() and not reset_done_for_today:
            reset_counts()
            reset_done_for_today = True
        if not should_reset() and reset_done_for_today:
            reset_done_for_today = False

        # Debug
        # cv2.imshow("YOLO RTSP Vehicle Counter", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        # --- Interval reporting ---
        if frame_idx % interval_frames == 0:
            payload = {class_list[cid]: class_counts[class_list[cid]] for cid in allowed_classes}
            print(f"\nInterval accumulation at frame {frame_idx}: {payload}")
            mqtt_publish_safe(MQTT_TOPIC, json.dumps(payload), qos=1)
            record_id = str(uuid.uuid4())
            db_execute_safe(
                "INSERT INTO vehicle_countings (id, device_id, data, created_at) VALUES (%s, %s, %s, %s)",
                (record_id, device_id, json.dumps(payload), datetime.datetime.now(datetime.UTC)), commit=True
                # "INSERT INTO vehicle_counting (device_id, data, created_at) VALUES (%s, %s, %s)",
                # (device_id, json.dumps(payload), datetime.datetime.now(datetime.UTC)), commit=True
            )
            interval_counts.clear()

    # Final leftover publish
    if frame_idx % interval_frames != 0:
        payload = {class_list[cid]: class_counts[class_list[cid]] for cid in allowed_classes}
        print(f"\nFinal interval at frame {frame_idx}: {payload}")
        mqtt_publish_safe(MQTT_TOPIC, json.dumps(payload), qos=2)

finally:
    print("\nSession totals:")
    for cid in allowed_classes:
        name = class_list[cid]
        print(f"  {name}: {class_counts[name]}")
    cap.release()
    cv2.destroyAllWindows()
    mqtt_client.loop_stop()
    mqtt_client.disconnect()
    try:
        cursor.close()
        pg_conn.close()
    except:
        pass
