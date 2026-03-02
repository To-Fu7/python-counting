
import os
import cv2
import json
import datetime
import uuid
import math
import numpy as np
from ultralytics import YOLO
import cvzone
import pandas as pd
import psycopg2
import time
import logging
from collections import defaultdict
from zoneinfo import ZoneInfo
import asyncio
import base64
from dotenv import load_dotenv
import paho.mqtt.client as mqtt
import threading
import ast
import string
import queue

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# GLOBAL VARIABLES
local_tz = ZoneInfo("Asia/Jakarta")
last_points = defaultdict(lambda: (None, None))
state_in = defaultdict(lambda: False)
state_out = defaultdict(lambda: False)
prev_intersecting = defaultdict(lambda: False)
class_counts = defaultdict(int)
person_history = {}
is_midnight = False
record_id = ''
overide_insert_id = ''
person_in = 0
person_out = 0

interval_person_in = 0
interval_person_out = 0

interval_record_id = ''

person_kid = 0
person_adult = 0
person_old = 0
person_man = 0
person_woman = 0

last_mqtt_send = None
last_daily_send = None

# Store latest person coordinates for MQTT
latest_person_coordinates = []

# Load environment variables
load_dotenv('.env')
PG_HOST = os.getenv('PG_HOST')
PG_PORT = int(os.getenv('PG_PORT', 5432))
PG_DB = os.getenv('PG_DB')
PG_USER = os.getenv('PG_USER')
PG_PASS = os.getenv('PG_PASS')
device_id = os.getenv('DEVICE_ID')
device_code = os.getenv('DEVICE_CODE')
device_name = os.getenv('DEVICE_NAME')

# MQTT Configuration
MQTT_BROKER = os.getenv('MQTT_BROKER', 'localhost')
MQTT_PORT = int(os.getenv('MQTT_PORT', '1883'))
MQTT_USERNAME = os.getenv('MQTT_USERNAME')
MQTT_PASSWORD = os.getenv('MQTT_PASSWORD')
MQTT_TOPIC = os.getenv('MQTT_TOPIC', '/xxx') # example /person_in
MQTT_INTERVAL_TOPIC = os.getenv('MQTT_INTERVAL_TOPIC', '/resampling_person/xxx')

# Interval settings
MQTT_INTERVAL_MINUTES = int(os.getenv('MQTT_INTERVAL_MINUTES', 5))
DAILY_SEND_TIME = os.getenv('DAILY_SEND_TIME', '23:59')  # Format: HH:MM

RTSP_URL = os.getenv('RTSP_URL')
resolution = ast.literal_eval(os.getenv("SCREEN_RESOLUTION"))

# Hardware video decoding (NVDEC) - set to 'true' to enable CUDA hardware decoding
ENABLE_NVDEC = os.getenv('ENABLE_NVDEC', 'false').lower() == 'true'

# Line coordinates (support multiple gates)
POINT_AXIS = os.getenv('POINT_AXIS', 'X')
DETECTION_STYLE = os.getenv('DETECTION_STYLE', 'dot').lower()

LINE_OFFSET = os.getenv('LINE_OFFSET', 'X')
LINE_OFFSET_AMOUNT = int(os.getenv('LINE_OFFSET_AMOUNT', 5))

DOT_OFFSET = os.getenv('DOT_OFFSET', 'Y')
DOT_OFFSET_AMOUNT = int(os.getenv('DOT_OFFSET_AMOUNT', 0))

# Swap IN/OUT detection order
# False = Cross OUT line first, then IN line to count IN (default)
# True = Cross IN line first, then OUT line to count IN
SWAP_IN_OUT = os.getenv('SWAP_IN_OUT', 'false').lower() == 'true'

# Merge all gates into one logical gate
# When true, crossing ANY in_line then ANY out_line (from any gate) counts as a single event
MERGE_GATES = os.getenv('MERGE_GATES', 'false').lower() == 'true'


def _compute_offset_line(base_line, offset_value, axis):
    """Compute an offset line from base_line along the given axis (X or Y)."""
    if axis == 'X':
        return [
            (base_line[0][0] + offset_value, base_line[0][1]),
            (base_line[1][0] + offset_value, base_line[1][1]),
        ]
    elif axis == 'Y':
        return [
            (base_line[0][0], base_line[0][1] + offset_value),
            (base_line[1][0], base_line[1][1] + offset_value),
        ]
    else:
        # If LINE_OFFSET is not recognized, just return the base line
        return base_line


def load_line_pairs_from_env():
    """
    Load dynamic line pairs from environment variables.

    Pairs are defined alphabetically:
      - (lineA, lineB) -> first gate
      - (lineC, lineD) -> second gate
      - (lineE, lineF) -> third gate
      - and so on...

    Rules:
      - If only the first line of a pair exists (e.g. lineA, but no lineB),
        the second line is generated using LINE_OFFSET and LINE_OFFSET_AMOUNT.
      - If both lines exist (e.g. lineC and lineD), they are used as-is.
      - If neither exists, that pair is skipped.
    """
    line_pairs = []

    # Go over letters in pairs: (A,B), (C,D), (E,F), ...
    letters = string.ascii_uppercase
    for i in range(0, len(letters), 2):
        first_letter = letters[i]
        # Ensure we have a second letter for the pair
        if i + 1 >= len(letters):
            break
        second_letter = letters[i + 1]

        first_name = f"line{first_letter}"
        second_name = f"line{second_letter}"

        first_val = os.getenv(first_name)
        second_val = os.getenv(second_name)

        # Skip if nothing defined for this pair
        if first_val is None and second_val is None:
            continue

        # Require at least the first line of the pair
        if first_val is None:
            logging.warning(
                f"{second_name} is set but {first_name} is missing. "
                f"Skipping this pair."
            )
            continue

        try:
            first_line = ast.literal_eval(first_val)
        except Exception as e:
            logging.error(f"Failed to parse {first_name} from env: {e}")
            continue

        if second_val is not None:
            # Use explicit second line from env
            try:
                second_line = ast.literal_eval(second_val)
            except Exception as e:
                logging.error(f"Failed to parse {second_name} from env: {e}")
                continue
        else:
            # Generate second line from first using offset
            second_line = _compute_offset_line(first_line, LINE_OFFSET_AMOUNT, LINE_OFFSET)

        line_pairs.append(
            {
                "in_name": first_name,
                "out_name": second_name,
                "in_line": first_line,
                "out_line": second_line,
            }
        )

    if not line_pairs:
        raise RuntimeError(
            "No valid line pairs found in environment. "
            "Please define at least 'lineA' (and optionally 'lineB')."
        )

    for lp in line_pairs:
        logging.info(
            f"Loaded line pair {lp['in_name']}/{lp['out_name']}: "
            f"{lp['in_line']} -> {lp['out_line']}"
        )

    return line_pairs


# Load all line pairs at startup
LINE_PAIRS = load_line_pairs_from_env()

# For backward-compatibility, keep references to the first pair as lineA/lineB
lineA = LINE_PAIRS[0]["in_line"]
lineB = LINE_PAIRS[0]["out_line"]

logging.info(f"Total line pairs loaded: {len(LINE_PAIRS)}")
logging.info(f"SWAP_IN_OUT = {SWAP_IN_OUT} ({'IN line first → count IN' if SWAP_IN_OUT else 'OUT line first → count IN'})")
logging.info(f"MERGE_GATES = {MERGE_GATES} ({'all gates unified' if MERGE_GATES else 'gates isolated'})")


# Detection region parameters (based on all line pairs)
# 'false' or '0' = global detection (full frame), any number = margin in pixels
_detection_margin_raw = os.getenv('DETECTION_MARGIN', '160').strip().lower()
GLOBAL_DETECTION = _detection_margin_raw in ('false', '0')
DETECTION_MARGIN = 0 if GLOBAL_DETECTION else int(_detection_margin_raw)

all_line_points_y = []
for lp in LINE_PAIRS:
    for (x, y) in lp["in_line"] + lp["out_line"]:
        all_line_points_y.append(y)

if GLOBAL_DETECTION:
    DETECTION_Y_MIN = 0
    DETECTION_Y_MAX = None  # Will use full frame height
    logging.info("Detection region: GLOBAL (full frame)")
else:
    DETECTION_Y_MIN = max(0, min(all_line_points_y) - DETECTION_MARGIN)
    DETECTION_Y_MAX = max(all_line_points_y) + DETECTION_MARGIN
    logging.info(f"Detection region: Y from {DETECTION_Y_MIN} to {DETECTION_Y_MAX} (margin: {DETECTION_MARGIN}px)")

# Image quality settings
CROP_PADDING = 30
MIN_CROP_SIZE = (128, 128)
JPEG_QUALITY = int(os.getenv('JPEG_QUALITY', 70))  # Lower quality = faster encoding, smaller payload

# YOLO CONFIG
YOLO_MODEL = os.getenv('YOLO_MODEL', 'yolo11n.pt')
YOLO_CONFIDENCE = float(os.getenv('YOLO_CONFIDENCE', 0.3))  # Confidence threshold (0.0-1.0)
YOLO_DEVICE = os.getenv('YOLO_DEVICE', 'auto')  # Device: 'auto', 'cpu', '0', 'cuda:0'

# PERFORMANCE
FPS_LIMIT = float(os.getenv('FPS_LIMIT', '0'))  # 0 = no limit, >0 = max processing FPS
FRAME_INTERVAL = 1.0 / FPS_LIMIT if FPS_LIMIT > 0 else 0
FRAME_SKIP = int(os.getenv('FRAME_SKIP', '2'))  # Process 1 out of every N frames

# DEBUG MODE
DEBUG_MODE = os.getenv('DEBUG_MODE', 'true').lower() == 'true'

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

# MQTT Client
mqtt_client = None

def init_mqtt():
    """Initialize MQTT client"""
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

def send_person_in_mqtt(cropped_image, record_id, event_type="person_in"):
    """Send cropped image via MQTT when person enters"""
    global mqtt_client
    
    if DEBUG_MODE:
        logging.info(f"DEBUG_MODE: Skipping MQTT send for {event_type}")
        return
    
    if mqtt_client is None:
        logging.warning("MQTT client not initialized, skipping message")
        return
    
    try:
        # Convert cropped image to bytes with higher quality
        _, buffer = cv2.imencode('.jpg', cropped_image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        image_bytes = buffer.tobytes()
        
        # Create payload
        payload = {
            "record_id": record_id,
            "device_id": device_id,
            "device_code" : device_code,
            "device_name" : device_name,
            "timestamp": datetime.datetime.now(local_tz).isoformat(),
            "event": event_type,
            "image": base64.b64encode(image_bytes).decode('utf-8')
        }
        
        # Send to MQTT
        result = mqtt_client.publish(MQTT_TOPIC, json.dumps(payload), qos=1)
        
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            logging.info(f"Person {event_type.upper()} image sent via MQTT for record {record_id}")
        else:
            logging.error(f"Failed to send MQTT message, error code: {result.rc}")
            
    except Exception as e:
        logging.error(f"Error sending MQTT message: {e}")

def get_age_gender_data_from_db():
    """Fetch age and gender data from database"""
    try:
        row = db_fetch(
            "SELECT data FROM person_inout WHERE id = %s",
            (record_id,)
        )
        
        if row and row[0]:
            # Check if data is already a dict or needs JSON parsing
            if isinstance(row[0], dict):
                data_json = row[0]
            else:
                data_json = json.loads(row[0])
                
            return {
                "age": {
                    "kid": data_json.get("person", {}).get("age", {}).get("kid", 0),
                    "adult": data_json.get("person", {}).get("age", {}).get("adult", 0),
                    "old": data_json.get("person", {}).get("age", {}).get("old", 0)
                },
                "gender": {
                    "man": data_json.get("person", {}).get("gender", {}).get("man", 0),
                    "woman": data_json.get("person", {}).get("gender", {}).get("woman", 0)
                }
            }
    except (json.JSONDecodeError, Exception) as e:
        logging.warning(f"Error fetching age/gender data from DB: {e}")
    
    # Return default values if error or no data
    return {
        "age": {"kid": 0, "adult": 0, "old": 0},
        "gender": {"man": 0, "woman": 0}
    }
    
def send_interval_mqtt_data():
    """Send interval data via MQTT (every 5 minutes and at 23:59)"""
    global mqtt_client, last_mqtt_send, last_daily_send, interval_person_in, interval_person_out

    if DEBUG_MODE:
        # logging.info("DEBUG_MODE: Skipping interval MQTT data send")
        return

    if mqtt_client is None:
        logging.warning("MQTT client not initialized, skipping interval data")
        return

    # Guard against rapid re-entry (multiple frames triggering in the same tick)
    current_time = datetime.datetime.now(local_tz)
    if last_mqtt_send is not None:
        elapsed = (current_time - last_mqtt_send).total_seconds()
        if elapsed < (MQTT_INTERVAL_MINUTES * 60) - 5:
            logging.warning(f"Skipping duplicate interval send (only {elapsed:.0f}s since last)")
            return

    # Mark send time BEFORE publish to block any concurrent calls
    last_mqtt_send = current_time

    # Snapshot and reset counters atomically before publish
    snapshot_in = interval_person_in
    snapshot_out = interval_person_out
    interval_person_in = 0
    interval_person_out = 0

    try:
        # Create payload with current interval counts (not total)
        payload = {
            "record_id": record_id,
            "device_id": device_id,
            "device_code": device_code,
            "device_name": device_name,
            "timestamp": current_time.isoformat(),
            "event": "interval_data",
            "data": {
                "interval_in": snapshot_in,
                "interval_out": snapshot_out,
                "total_in": person_in,  # Keep total for reference
                "total_out": person_out,  # Keep total for reference
                "net_count": person_in - person_out,
                "interval_net": snapshot_in - snapshot_out,
                "interval_minutes": MQTT_INTERVAL_MINUTES
            }
        }

        # Send to MQTT
        result = mqtt_client.publish(MQTT_INTERVAL_TOPIC, json.dumps(payload), qos=1)

        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            logging.info(f"Interval data sent via MQTT - Interval IN: {snapshot_in}, Interval OUT: {snapshot_out}, Total IN: {person_in}, Total OUT: {person_out}")
        else:
            # Restore counters on publish failure
            interval_person_in += snapshot_in
            interval_person_out += snapshot_out
            logging.error(f"Failed to send interval MQTT data, error code: {result.rc}")

    except Exception as e:
        # Restore counters on exception
        interval_person_in += snapshot_in
        interval_person_out += snapshot_out
        logging.error(f"Error sending interval MQTT data: {e}")


def should_send_interval_mqtt():
    """Check if it's time to send interval MQTT data"""
    global last_mqtt_send, last_daily_send
    
    current_time = datetime.datetime.now(local_tz)
    
    # Check for daily send time (23:59)
    daily_hour, daily_minute = map(int, DAILY_SEND_TIME.split(':'))
    if (current_time.hour == daily_hour and current_time.minute == daily_minute and 
        current_time.second < 10):
        
        # Check if we haven't sent today's daily report yet
        if (last_daily_send is None or 
            last_daily_send.date() != current_time.date()):
            last_daily_send = current_time
            logging.info("Daily MQTT send triggered at 23:59")
            return True
    
    # Check for interval send (every X minutes)
    if last_mqtt_send is None:
        return True
    
    time_diff = current_time - last_mqtt_send
    if time_diff.total_seconds() >= (MQTT_INTERVAL_MINUTES * 60):
        return True
    
    return False

def is_in_detection_region(box):
    """Check if detected person is within the detection region"""
    if GLOBAL_DETECTION:
        return True
    x1, y1, x2, y2 = box
    return not (y2 < DETECTION_Y_MIN or y1 > DETECTION_Y_MAX)

def is_crossing_line(p1, p2, line):
    """Check if segment p1-p2 crosses the given line using cross product"""
    if p1 is None or p2 is None:
        return False
    try:
        a, b = line[0], line[1]
        d1 = (b[0]-a[0])*(p1[1]-a[1]) - (b[1]-a[1])*(p1[0]-a[0])
        d2 = (b[0]-a[0])*(p2[1]-a[1]) - (b[1]-a[1])*(p2[0]-a[0])
        if d1 * d2 >= 0:
            return False
        d3 = (p2[0]-p1[0])*(a[1]-p1[1]) - (p2[1]-p1[1])*(a[0]-p1[0])
        d4 = (p2[0]-p1[0])*(b[1]-p1[1]) - (p2[1]-p1[1])*(b[0]-p1[0])
        return d3 * d4 < 0
    except Exception:
        return False

def is_edge_intersecting(edge_start, edge_end, detection_line):
    """Check if bbox edge segment intersects with the detection line using cross product"""
    try:
        p1, p2 = edge_start, edge_end
        a, b = detection_line[0], detection_line[1]
        d1 = (b[0]-a[0])*(p1[1]-a[1]) - (b[1]-a[1])*(p1[0]-a[0])
        d2 = (b[0]-a[0])*(p2[1]-a[1]) - (b[1]-a[1])*(p2[0]-a[0])
        if d1 * d2 > 0:
            return False
        d3 = (p2[0]-p1[0])*(a[1]-p1[1]) - (p2[1]-p1[1])*(a[0]-p1[0])
        d4 = (p2[0]-p1[0])*(b[1]-p1[1]) - (p2[1]-p1[1])*(b[0]-p1[0])
        return d3 * d4 <= 0
    except Exception:
        return False

def validate_cctv_connection(rtsp_url, timeout=5):
    """Validate CCTV connection by attempting to open and read a frame"""
    if not rtsp_url or rtsp_url.strip() == '':
        logging.warning("RTSP_URL is empty or not set")
        return False
    
    try:
        logging.info(f"Validating CCTV connection: {rtsp_url}")
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            logging.warning("CCTV stream failed to open")
            cap.release()
            return False
        
        # Try to read a frame to verify connection
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

def safe_destroy_windows():
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass

def initialize_video_capture(video_source):
    """Initialize video capture with the given video source (RTSP URL or file path)"""
    logging.info(f'Initializing video capture with source: {video_source}')

    # Enable NVIDIA hardware video decoding (NVDEC) if configured
    if ENABLE_NVDEC:
        # Set FFmpeg options for CUDA hardware decoding
        # This offloads H.264/HEVC decoding from CPU to GPU
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'hwaccel;cuda|video_codec;h264_cuvid|rtsp_transport;tcp'
        logging.info('NVDEC hardware decoding enabled (h264_cuvid)')

    cap = cv2.VideoCapture(video_source, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 10)
    return cap

def get_video_source():
    """Get video source with CCTV validation and fallback to 1.mp4"""
    fallback_video = '1.mp4'
    
    # Check if RTSP_URL is set
    if not RTSP_URL or RTSP_URL.strip() == '':
        logging.warning(f"RTSP_URL is not set or empty. Falling back to {fallback_video}")
        return fallback_video
    
    # Validate CCTV connection
    if validate_cctv_connection(RTSP_URL):
        logging.info("Using CCTV stream as video source")
        return RTSP_URL
    else:
        logging.warning(f"CCTV connection failed or disabled. Falling back to {fallback_video}")
        # Verify fallback file exists
        if os.path.exists(fallback_video):
            logging.info(f"Fallback video file found: {fallback_video}")
            return fallback_video
        else:
            logging.error(f"Fallback video file not found: {fallback_video}")
            raise FileNotFoundError(f"Neither CCTV stream nor fallback video file ({fallback_video}) is available")

def db_connect():
    """Create database connection"""
    return psycopg2.connect(
        dbname=PG_DB,
        user=PG_USER,
        password=PG_PASS,
        host=PG_HOST,
        port=PG_PORT
    )

def db_get_cursor():
    """Get database connection and cursor"""
    try:
        connect = db_connect()
        return connect, connect.cursor()
    except Exception as error:
        logging.error(f"Postgres Connection Failed: {error}")
        return None, None

# Initialize database connection
if not DEBUG_MODE:
    pg_conn, cursor = db_get_cursor()
    if not cursor:
        logging.error("Fatal: Error on Connecting DB at Start Up")
        exit(1)
    logging.info("Successfully connected to PostgreSQL database")
else:
    pg_conn, cursor = None, None
    # logging.info("DEBUG_MODE: Skipping database connection initialization")

logging.info(f"RESOLUTION = {resolution[0],resolution[1]}")

# Async database write queue for non-blocking DB operations
db_queue = queue.Queue()
db_thread_running = True

def db_worker():
    """Background worker thread for async database writes"""
    global db_thread_running
    while db_thread_running:
        try:
            item = db_queue.get(timeout=1)
            if item is None:
                break
            query, params = item
            db_query(query, params, commit=True)
            db_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            logging.error(f"DB worker error: {e}")

def db_queue_write(query, params):
    """Queue a database write operation for async execution"""
    if DEBUG_MODE:
        logging.info(f"DEBUG_MODE: Skipping async DB write: {query}")
        return
    db_queue.put((query, params))


def should_reset():
    """Check if it's time to reset counters (midnight)"""
    now = datetime.datetime.now(local_tz)
    return now.hour == 0 and now.minute == 0 and now.second < 10

def db_query(sql, params=(), commit=False, max_retry=3):
    """Execute database query with retry logic"""
    global pg_conn, cursor
    
    if DEBUG_MODE:
        logging.info(f"DEBUG_MODE: Skipping DB query: {sql}")
        return True
    
    retry = 0
    while retry < max_retry:
        try:
            cursor.execute(sql, params)
            if commit:
                pg_conn.commit()
            return True
        except (psycopg2.OperationalError, psycopg2.InterfaceError) as error:
            logging.error(f"Database lost connection: {error} (retry {retry+1}/{max_retry})")
            try:
                cursor.close()
                pg_conn.close()
            except: 
                pass
            pg_conn, cursor = db_get_cursor()
            if not cursor:
                logging.error("DB Reconnection failed.")
                time.sleep(2)
                retry += 1
                continue
        except Exception as error:
            logging.error(f"DB Error (not connection): {error}")
            time.sleep(2)
            retry += 1
            continue

    logging.error("DB operation failed after max retries.")
    return False

def db_fetch(sql, params=(), max_retry=3):
    """Fetch data from database with retry logic"""
    global pg_conn, cursor
    
    if DEBUG_MODE:
        logging.info(f"DEBUG_MODE: Skipping DB fetch: {sql}")
        return None
    
    retry = 0
    while retry < max_retry:
        try:
            cursor.execute(sql, params)
            return cursor.fetchone()
        except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
            logging.error(f"Database lost connection: {e} (retry {retry+1}/{max_retry})")
            try:
                cursor.close()
                pg_conn.close()
            except: 
                pass
            pg_conn, cursor = db_get_cursor()
            if not cursor:
                logging.error("Reconnection to DB failed.")
                time.sleep(2)
                retry += 1
                continue
        except Exception as e:
            logging.error(f"DB Error (not connection): {e}")
            time.sleep(2)
            retry += 1
            continue
        
    logging.error("DB fetch failed after max retries.")
    return None

def get_latest_counts(device_id):
    """Get latest counts from database"""
    row = db_fetch(
        "SELECT id, total_in, total_out, data, created_at FROM person_inout WHERE device_id = %s ORDER BY created_at DESC LIMIT 1",
        (device_id,))
    if not row:
        return None, None, None, None

    last_device_id, total_in, total_out, data, last_created_utc = row
    last_id = {"id": last_device_id}
    last_data_json = {
        "in": total_in if total_in is not None else 0,
        "out": total_out if total_out is not None else 0
    }
    
    # Extract extra data (age/gender info) from database
    extra_data = None
    if data:
        try:
            if isinstance(data, dict):
                extra_data = data
            else:
                extra_data = json.loads(data)
        except (json.JSONDecodeError, Exception) as e:
            logging.warning(f"Error parsing extra data from DB: {e}")
            extra_data = None
    
    logging.info(f"Last record ID: {last_device_id}")
    
    last_created_local = last_created_utc.astimezone(local_tz)
    last_date_local = last_created_local.date()
    today_local = datetime.datetime.now(local_tz).date()

    if last_date_local == today_local:
        return last_data_json, last_date_local, last_id, extra_data

    return None, last_date_local, last_id, extra_data

def initialize_counts():
    """Initialize counters from database or create new record"""
    global person_in, person_out, record_id, class_counts, interval_person_in, interval_person_out
    
    # Initialize interval counters
    interval_person_in = 0
    interval_person_out = 0
    
    if DEBUG_MODE:
        # Skip DB validation in debug mode
        new_id = str(uuid.uuid4())
        record_id = new_id
        class_counts['in'] = 0
        class_counts['out'] = 0
        person_in = 0
        person_out = 0
        logging.info(f"DEBUG_MODE: Skipping DB validation, initialized with new record id {new_id}")
        return {"id": new_id}
    
    # FIX: unpack four values (restored_counts, last_date_local, last_data_id, extra_data)
    restored_counts, last_date_local, last_data_id, extra_data = get_latest_counts(device_id)

    if restored_counts:
        class_counts.update(restored_counts)
        person_in = class_counts['in']
        person_out = class_counts['out']
        record_id = last_data_id['id']
        logging.info(f"Restored counts from DB ({last_date_local}): IN={person_in}, OUT={person_out}")
        return last_data_id
    else:
        new_id = str(uuid.uuid4())
        success = db_query("INSERT INTO person_inout (id, device_id, total_in, total_out) VALUES (%s, %s, %s, %s)",
                        (new_id, device_id, 0, 0), commit=True)
        if success:
            record_id = new_id
            class_counts['in'] = 0
            class_counts['out'] = 0
            person_in = 0
            person_out = 0
            logging.info(f"No valid counts to restore, created new record with id {new_id}")
            return {"id": new_id}
        else:
            logging.error("Failed to create new row")
            return None

def reset_counts():
    """Reset counts at midnight"""
    global class_counts, state_in, state_out, prev_intersecting, person_history, record_id, person_in, person_out, last_mqtt_send, last_daily_send, interval_person_in, interval_person_out
    
    # Send final daily report before reset
    send_interval_mqtt_data()
    
    person_in, person_out = 0, 0
    interval_person_in, interval_person_out = 0, 0  # Reset interval counters too
    class_counts.clear()
    class_counts['in'] = 0
    class_counts['out'] = 0
    state_in.clear()
    state_out.clear()
    prev_intersecting.clear()
    person_history.clear()
    
    new_id = str(uuid.uuid4())
    
    if DEBUG_MODE:
        record_id = new_id
        last_mqtt_send = None
        logging.info("== Midnight Reached: Totals Reset (DEBUG_MODE: Skipping DB operation) ==")
    else:
        success = db_query("INSERT INTO person_inout (id, device_id, total_in, total_out) VALUES (%s, %s, %s, %s)",
                        (new_id, device_id, 0, 0), commit=True)   
        if success:
            record_id = new_id
            # Reset MQTT timers
            last_mqtt_send = None
            logging.info("== Midnight Reached: Totals Reset ==")
        else:
            logging.error("Error creating new record at midnight")

def crop_image(frame, box, padding=None):
    """Crop image around detected person with improved quality"""
    if padding is None:
        padding = CROP_PADDING
        
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]

    # Add padding around the detection box
    x1_crop = max(0, x1 - padding)
    y1_crop = max(0, y1 - padding)
    x2_crop = min(w, x2 + padding)
    y2_crop = min(h, y2 + padding)
    
    # Crop the image
    person_crop = frame[y1_crop:y2_crop, x1_crop:x2_crop]
    
    # Check if crop is too small and resize if necessary
    crop_h, crop_w = person_crop.shape[:2]
    if crop_h < MIN_CROP_SIZE[1] or crop_w < MIN_CROP_SIZE[0]:
        # Calculate aspect ratio preserving resize
        aspect_ratio = crop_w / crop_h
        if aspect_ratio > 1:  # Wider than tall
            new_w = max(MIN_CROP_SIZE[0], crop_w)
            new_h = int(new_w / aspect_ratio)
            if new_h < MIN_CROP_SIZE[1]:
                new_h = MIN_CROP_SIZE[1]
                new_w = int(new_h * aspect_ratio)
        else:  # Taller than wide
            new_h = max(MIN_CROP_SIZE[1], crop_h)
            new_w = int(new_h * aspect_ratio)
            if new_w < MIN_CROP_SIZE[0]:
                new_w = MIN_CROP_SIZE[0]
                new_h = int(new_w / aspect_ratio)
        
        person_crop = cv2.resize(person_crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    return person_crop

def RGB(event, x, y, flags, param):
    """Mouse callback function for RGB window"""
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

def main():
    """Main function"""
    global person_in, person_out, is_midnight, record_id, latest_person_coordinates, interval_person_in, interval_person_out, db_thread_running

    # Start async database worker thread
    if not DEBUG_MODE:
        db_worker_thread = threading.Thread(target=db_worker, daemon=True)
        db_worker_thread.start()
        logging.info("Async database worker thread started")

    # Initialize MQTT
    init_mqtt()

    # Initialize database
    last_data_id = initialize_counts()
    if not last_data_id:
        if not DEBUG_MODE:
            logging.error("Failed to initialize database")
            return
        else:
            logging.info("DEBUG_MODE: Continuing without database initialization")
    
    # Log configuration
    logging.info(f"MQTT interval: {MQTT_INTERVAL_MINUTES} minutes")
    logging.info(f"Daily MQTT send time: {DAILY_SEND_TIME}")
    logging.info(f"Image crop settings - Padding: {CROP_PADDING}px, Min size: {MIN_CROP_SIZE}, Quality: {JPEG_QUALITY}%")
    
    # Get video source with CCTV validation and fallback
    try:
        video_source = get_video_source()
        logging.info(f"Video source selected: {video_source}")
    except FileNotFoundError as e:
        logging.error(f"Fatal error: {e}")
        return
    
    model = YOLO(YOLO_MODEL)
    resolved_device = resolve_yolo_device(YOLO_DEVICE)
    names = model.names

    last_waiting_log = time.time()
    
    while True:
        try:
            logging.info('Initializing Service...')
            
            cap = initialize_video_capture(video_source)
            if not cap.isOpened():
                logging.error(f"Failed to open video source: {video_source}")
                raise Exception(f"Failed to open video source: {video_source}")
            else:
                logging.info(f'Person IN: {person_in}, Person OUT: {person_out}')
                logging.info(f"Video source opened successfully: {video_source}")
                logging.info(f"Device ID = {device_id}")
            
            
            #get point   
            if DEBUG_MODE: 
                cv2.namedWindow('RGB')
                cv2.setMouseCallback('RGB', RGB)

            count = 0
            last_process_time = time.time()
            fps_counter = 0
            fps_timer = time.time()
            while True:
                count += 1
                if count % FRAME_SKIP != 0:
                    cap.grab()  # Advance buffer without decoding
                    continue
                ret, frame = cap.read()

                # FPS limiting
                if FRAME_INTERVAL > 0:
                    elapsed = time.time() - last_process_time
                    if elapsed < FRAME_INTERVAL:
                        time.sleep(FRAME_INTERVAL - elapsed)
                    last_process_time = time.time()

                if not ret:
                    logging.error(f"Failed to read frame from video source: {video_source}")
                    raise Exception(f"Frame read error or video source disconnected: {video_source}")

                # Screen Resolution
                # frame = cv2.resize(frame, (1280, 720))
                frame = cv2.resize(frame, (resolution[0], resolution[1]))

                # Create detection frame (cropped region or full frame)
                if GLOBAL_DETECTION:
                    detection_frame = frame
                else:
                    detection_frame = frame[DETECTION_Y_MIN:DETECTION_Y_MAX, :]
                
                # Run YOLO only on the detection region with confidence threshold
                results = model.track(detection_frame, persist=True, verbose=False, conf=YOLO_CONFIDENCE, device=resolved_device, classes=[0], iou=0.45, tracker="bytetrack.yaml")

                # Draw all IN/OUT lines (only in DEBUG_MODE)
                if DEBUG_MODE:
                    for lp in LINE_PAIRS:
                        # IN LINE ( BLUE ) (BGR)
                        cv2.line(frame, lp["in_line"][0], lp["in_line"][1], (255, 0, 0), 4)
                        # OUT LINE ( YELLOW ) (BGR)
                        cv2.line(frame, lp["out_line"][0], lp["out_line"][1], (0, 255, 255), 4)

                        # Draw IN/OUT direction arrows
                        in_mid = ((lp["in_line"][0][0] + lp["in_line"][1][0]) // 2,
                                  (lp["in_line"][0][1] + lp["in_line"][1][1]) // 2)
                        out_mid = ((lp["out_line"][0][0] + lp["out_line"][1][0]) // 2,
                                   (lp["out_line"][0][1] + lp["out_line"][1][1]) // 2)
                        dx = out_mid[0] - in_mid[0]
                        dy = out_mid[1] - in_mid[1]
                        dist = math.sqrt(dx * dx + dy * dy)
                        if dist > 0:
                            ux, uy = dx / dist, dy / dist
                            # Arrow on in_line's outer side (pointing toward in_line)
                            a_start = (int(in_mid[0] - ux * 50), int(in_mid[1] - uy * 50))
                            a_end = (int(in_mid[0] - ux * 20), int(in_mid[1] - uy * 20))
                            # Arrow on out_line's outer side (pointing toward out_line)
                            b_start = (int(out_mid[0] + ux * 50), int(out_mid[1] + uy * 50))
                            b_end = (int(out_mid[0] + ux * 20), int(out_mid[1] + uy * 20))

                            if SWAP_IN_OUT:
                                in_start, in_end = a_start, a_end
                                out_start, out_end = b_start, b_end
                            else:
                                in_start, in_end = b_start, b_end
                                out_start, out_end = a_start, a_end

                            cv2.arrowedLine(frame, in_start, in_end, (0, 255, 0), 2, tipLength=0.3)
                            cv2.putText(frame, 'IN', (in_start[0] - 5, in_start[1] - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            cv2.arrowedLine(frame, out_start, out_end, (0, 0, 255), 2, tipLength=0.3)
                            cv2.putText(frame, 'OUT', (out_start[0] - 10, out_start[1] - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    # Draw detection region boundaries (skip in global mode)
                    if not GLOBAL_DETECTION:
                        cv2.line(frame, (0, DETECTION_Y_MIN), (1920, DETECTION_Y_MIN), (0, 255, 0), 2)
                        cv2.line(frame, (0, DETECTION_Y_MAX), (1920, DETECTION_Y_MAX), (0, 222, 0), 2)
                        cv2.putText(frame, 'Detection Region', (10, DETECTION_Y_MIN - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Only copy frame in DEBUG_MODE, otherwise use reference (saves CPU)
                if DEBUG_MODE:
                    original_frame = frame.copy()
                else:
                    original_frame = frame

                person_detected = False
                region_detections = 0
                
                # Clear and update person coordinates for current frame
                latest_person_coordinates = []
                
                if results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.int().cpu().tolist()
                    class_ids = results[0].boxes.cls.int().cpu().tolist()
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    confidences = results[0].boxes.conf.cpu().tolist()

                    for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
                        c = names[class_id]

                        if 'person' in c:
                            person_detected = True
                            region_detections += 1
                            
                            # Adjust box coordinates back to full frame
                            x1, y1, x2, y2 = box
                            y1 += DETECTION_Y_MIN
                            y2 += DETECTION_Y_MIN
                            adjusted_box = [x1, y1, x2, y2]
                            
                            # Store person coordinates for MQTT
                            person_coord = {
                                "track_id": track_id,
                                "x": int(x1),
                                "y": int(y1),
                                "w": int(x2 - x1),
                                "h": int(y2 - y1),
                                "confidence": float(conf),
                                "center_x": int((x1 + x2) // 2),
                                "center_y": int((y1 + y2) // 2)
                            }
                            latest_person_coordinates.append(person_coord)
                            
                            
                            # Calculate detection points/edges based on DETECTION_STYLE
                            if DETECTION_STYLE == 'line':
                                if POINT_AXIS == "Y":
                                    first_edge = ((x1, y1), (x2, y1))
                                    second_edge = ((x1, y2), (x2, y2))
                                elif POINT_AXIS == "X":
                                    first_edge = ((x1, y1), (x1, y2))
                                    second_edge = ((x2, y1), (x2, y2))
                            else:
                                if POINT_AXIS == "Y":
                                    first_point = ((x1 + x2) // 2, y1 - DOT_OFFSET_AMOUNT)
                                    second_point = ((x1 + x2) // 2, y2 + DOT_OFFSET_AMOUNT)
                                elif POINT_AXIS == "X":
                                    first_point = (x1 - DOT_OFFSET_AMOUNT, (y1 + y2) // 2)
                                    second_point = (x2 + DOT_OFFSET_AMOUNT, (y1 + y2) // 2)

                            # Draw person detection (only in DEBUG_MODE)
                            if DEBUG_MODE:
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                cvzone.putTextRect(frame, f'{track_id}', (x1, y1), 1, 1)
                                if DETECTION_STYLE == 'line':
                                    cv2.line(frame, first_edge[0], first_edge[1], (255, 0, 0), 2)
                                    cv2.line(frame, second_edge[0], second_edge[1], (0, 255, 255), 2)
                                else:
                                    cv2.circle(frame, first_point, 4, (255, 0, 0), 2)
                                    cv2.circle(frame, second_point, 4, (0, 255, 255), 2)

                            # Determine if crossing detection can run
                            can_check_crossing = True
                            if DETECTION_STYLE != 'line':
                                prev_points = last_points[track_id]
                                if prev_points[0] is None or prev_points[1] is None:
                                    can_check_crossing = False
                                else:
                                    prev_top, prev_bottom = prev_points

                            if can_check_crossing:
                                # Check crossings against all line pairs
                                for gate_index, lp in enumerate(LINE_PAIRS):
                                    gate_key = track_id if MERGE_GATES else (track_id, gate_index)
                                    gate_label = "merged" if MERGE_GATES else f"gate {gate_index}"

                                    if DETECTION_STYLE == 'line':
                                        # Edge intersection with entry-event detection
                                        touching_in = is_edge_intersecting(first_edge[0], first_edge[1], lp["in_line"])
                                        touching_out = is_edge_intersecting(second_edge[0], second_edge[1], lp["out_line"])
                                        in_key = (track_id, 'in') if MERGE_GATES else (track_id, gate_index, 'in')
                                        out_key = (track_id, 'out') if MERGE_GATES else (track_id, gate_index, 'out')
                                        crossed_A = touching_in and not prev_intersecting.get(in_key, False)
                                        crossed_B = touching_out and not prev_intersecting.get(out_key, False)
                                        prev_intersecting[in_key] = touching_in
                                        prev_intersecting[out_key] = touching_out
                                    else:
                                        # Point movement crossing detection
                                        crossed_A = is_crossing_line(prev_top, first_point, lp["in_line"])
                                        crossed_B = is_crossing_line(prev_bottom, second_point, lp["out_line"])

                                    # SWAP_IN_OUT swaps which line triggers IN vs OUT
                                    # False (default): Cross B first, then A = IN | Cross A first, then B = OUT
                                    # True (swapped):  Cross A first, then B = IN | Cross B first, then A = OUT
                                    if SWAP_IN_OUT:
                                        # Swapped mode
                                        if crossed_A:
                                            if state_out.get(gate_key):
                                                person_out += 1
                                                interval_person_out += 1
                                                class_counts['out'] = person_out

                                                # Async DB update (non-blocking)
                                                db_queue_write(
                                                    "UPDATE person_inout SET total_out = %s WHERE id = %s",
                                                    (person_out, record_id)
                                                )

                                                send_person_in_mqtt(original_frame, record_id, "person_out")

                                                logging.info(
                                                    f'Person {track_id} OUT through {gate_label} - Total OUT: {person_out}'
                                                )
                                                state_out[gate_key] = False
                                            else:
                                                state_in[gate_key] = True
                                                logging.info(
                                                    f'Person {track_id} crossed IN line of {gate_label} (preparing for In)'
                                                )

                                        elif crossed_B:
                                            if state_in.get(gate_key):
                                                person_in += 1
                                                interval_person_in += 1
                                                class_counts['in'] = person_in

                                                # Async DB update (non-blocking)
                                                db_queue_write(
                                                    "UPDATE person_inout SET total_in = %s WHERE id = %s",
                                                    (person_in, record_id)
                                                )

                                                send_person_in_mqtt(original_frame, record_id, "person_in")

                                                logging.info(
                                                    f'Person {track_id} IN through {gate_label} - Total IN: {person_in}'
                                                )
                                                state_in[gate_key] = False
                                            else:
                                                state_out[gate_key] = True
                                                logging.info(
                                                    f'Person {track_id} crossed OUT line of {gate_label} (preparing for Out)'
                                                )
                                    else:
                                        # Default mode
                                        if crossed_A:
                                            if state_in.get(gate_key):
                                                person_in += 1
                                                interval_person_in += 1
                                                class_counts['in'] = person_in

                                                # Async DB update (non-blocking)
                                                db_queue_write(
                                                    "UPDATE person_inout SET total_in = %s WHERE id = %s",
                                                    (person_in, record_id)
                                                )

                                                send_person_in_mqtt(original_frame, record_id, "person_in")

                                                logging.info(
                                                    f'Person {track_id} IN through {gate_label} - Total IN: {person_in}'
                                                )
                                                state_in[gate_key] = False
                                            else:
                                                state_out[gate_key] = True
                                                logging.info(
                                                    f'Person {track_id} crossed IN line of {gate_label} (preparing for Out)'
                                                )

                                        elif crossed_B:
                                            if state_out.get(gate_key):
                                                person_out += 1
                                                interval_person_out += 1
                                                class_counts['out'] = person_out

                                                # Async DB update (non-blocking)
                                                db_queue_write(
                                                    "UPDATE person_inout SET total_out = %s WHERE id = %s",
                                                    (person_out, record_id)
                                                )

                                                send_person_in_mqtt(original_frame, record_id, "person_out")

                                                logging.info(
                                                    f'Person {track_id} OUT through {gate_label} - Total OUT: {person_out}'
                                                )
                                                state_out[gate_key] = False
                                            else:
                                                state_in[gate_key] = True
                                                logging.info(
                                                    f'Person {track_id} crossed OUT line of {gate_label} (preparing for In)'
                                                )

                            if DETECTION_STYLE != 'line':
                                last_points[track_id] = (first_point, second_point)
                
                # Check for interval MQTT sending
                if should_send_interval_mqtt():
                    send_interval_mqtt_data()
                
                if not person_detected:
                    current_time = time.time()
                    if current_time - last_waiting_log >= 60:
                        logging.info("Waiting for person detection...")
                        last_waiting_log = current_time

                # Display counters (only in DEBUG_MODE)
                if DEBUG_MODE:
                    cv2.putText(frame, f'IN: {person_in}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f'OUT: {person_out}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(frame, f'Region Detections: {region_detections}', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                #SCREEN
                if DEBUG_MODE:
                    cv2.imshow("RGB", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        logging.info("User requested exit")
                        return

                # Processing FPS monitor
                fps_counter += 1
                if time.time() - fps_timer >= 10.0:
                    actual_fps = fps_counter / (time.time() - fps_timer)
                    logging.info(f"Processing FPS: {actual_fps:.1f}")
                    fps_counter = 0
                    fps_timer = time.time()

                # Handle midnight reset
                if should_reset() and not is_midnight:
                    reset_counts()
                    is_midnight = True
                    
                if not should_reset() and is_midnight:
                    is_midnight = False

        except Exception as error:
            logging.error(f"Error occurred: {str(error)}. Restarting in 5 seconds...")
            if 'cap' in locals():
                cap.release()
            safe_destroy_windows()
            last_points.clear()
            prev_intersecting.clear()
            person_history.clear()
            time.sleep(5)
            continue
    
    # Cleanup
    if 'cap' in locals():
        cap.release()
    safe_destroy_windows()
    if mqtt_client:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()

if __name__ == "__main__":
    main()
