
import os
import cv2
import json
import datetime
import uuid
import numpy as np
from ultralytics import YOLO
import cvzone
import pandas as pd
import psycopg2
import time
import logging
from collections import defaultdict
from shapely.geometry import LineString
from zoneinfo import ZoneInfo
import asyncio
import base64
from dotenv import load_dotenv
import paho.mqtt.client as mqtt
import threading
import ast

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# GLOBAL VARIABLES
local_tz = ZoneInfo("Asia/Jakarta")
last_points = defaultdict(lambda: (None, None)) 
state_in = defaultdict(lambda: False)
state_out = defaultdict(lambda: False)
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
load_dotenv()
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
MQTT_PORT = int(os.getenv('MQTT_PORT', 1883))
MQTT_USERNAME = os.getenv('MQTT_USERNAME')
MQTT_PASSWORD = os.getenv('MQTT_PASSWORD')
MQTT_TOPIC = os.getenv('MQTT_TOPIC', '/xxx') # example /person_in
MQTT_INTERVAL_TOPIC = os.getenv('MQTT_INTERVAL_TOPIC', '/resampling_person/xxx')

# Interval settings
MQTT_INTERVAL_MINUTES = int(os.getenv('MQTT_INTERVAL_MINUTES', 5))
DAILY_SEND_TIME = os.getenv('DAILY_SEND_TIME', '23:59')  # Format: HH:MM

RTSP_URL = os.getenv('RTSP_URL')

# Line coordinates
lineA = ast.literal_eval(os.getenv("lineA"))
lineB = ast.literal_eval(os.getenv("lineB"))

# Detection region parameters
DETECTION_MARGIN = 160
DETECTION_Y_MIN = min(lineA[0][1], lineB[0][1]) - DETECTION_MARGIN
DETECTION_Y_MAX = max(lineA[1][1], lineB[1][1]) + DETECTION_MARGIN

# Image quality settings
CROP_PADDING = 30
MIN_CROP_SIZE = (128, 128)
JPEG_QUALITY = 95

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
    
    if mqtt_client is None:
        logging.warning("MQTT client not initialized, skipping interval data")
        return
    
    try:
        current_time = datetime.datetime.now(local_tz)
        
        # Create payload with current interval counts (not total)
        payload = {
            "record_id": record_id,
            "device_id": device_id,
            "device_code": device_code,
            "device_name": device_name,
            "timestamp": current_time.isoformat(),
            "event": "interval_data",
            "data": {
                "interval_in": interval_person_in,  # Current interval count
                "interval_out": interval_person_out,  # Current interval count
                "total_in": person_in,  # Keep total for reference
                "total_out": person_out,  # Keep total for reference
                "net_count": person_in - person_out,
                "interval_net": interval_person_in - interval_person_out,  # Net for this interval
                "interval_minutes": MQTT_INTERVAL_MINUTES
            }
        }
        
        # Send to MQTT
        result = mqtt_client.publish(MQTT_INTERVAL_TOPIC, json.dumps(payload), qos=1)
        
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            logging.info(f"Interval data sent via MQTT - Interval IN: {interval_person_in}, Interval OUT: {interval_person_out}, Total IN: {person_in}, Total OUT: {person_out}")
            last_mqtt_send = current_time
            
            # Reset interval counters after sending
            interval_person_in = 0
            interval_person_out = 0
        else:
            logging.error(f"Failed to send interval MQTT data, error code: {result.rc}")
            
    except Exception as e:
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
    x1, y1, x2, y2 = box
    return not (y2 < DETECTION_Y_MIN or y1 > DETECTION_Y_MAX)

def is_crossing_line(p1, p2, line):
    """Check if line segment p1-p2 crosses the given line"""
    if p1 is None or p2 is None:
        return False
    try:
        person_line = LineString([p1, p2])
        the_line = LineString(line)
        return person_line.crosses(the_line)
    except Exception as e:
        logging.warning(f"Error checking line crossing: {e}")
        return False

def initialize_video_capture(rtsp_url):
    """Initialize video capture with the given RTSP URL"""
    logging.info('Initialize video capture with the given RTSP URL.')
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 10)
    return cap

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
pg_conn, cursor = db_get_cursor()
if not cursor:
    logging.error("Fatal: Error on Connecting DB at Start Up")
    exit(1)
logging.info("Successfully connected to PostgreSQL database")

def should_reset():
    """Check if it's time to reset counters (midnight)"""
    now = datetime.datetime.now(local_tz)
    return now.hour == 0 and now.minute == 0 and now.second < 10

def db_query(sql, params=(), commit=False, max_retry=3):
    """Execute database query with retry logic"""
    global pg_conn, cursor
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
    global class_counts, state_in, state_out, person_history, record_id, person_in, person_out, last_mqtt_send, last_daily_send, interval_person_in, interval_person_out
    
    # Send final daily report before reset
    send_interval_mqtt_data()
    
    person_in, person_out = 0, 0
    interval_person_in, interval_person_out = 0, 0  # Reset interval counters too
    class_counts.clear()
    class_counts['in'] = 0
    class_counts['out'] = 0
    state_in.clear()
    state_out.clear()
    person_history.clear()
    
    new_id = str(uuid.uuid4())
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

# def RGB(event, x, y, flags, param):
#     """Mouse callback function for RGB window"""
#     if event == cv2.EVENT_MOUSEMOVE:
#         point = [x, y]
#         print(point)

def main():
    """Main function"""
    global person_in, person_out, is_midnight, record_id, latest_person_coordinates
    
    # Initialize MQTT
    init_mqtt()
    
    # Initialize database
    last_data_id = initialize_counts()
    if not last_data_id:
        logging.error("Failed to initialize database")
        return
    
    # Log configuration
    logging.info(f"Detection region: Y from {DETECTION_Y_MIN} to {DETECTION_Y_MAX} (margin: {DETECTION_MARGIN}px)")
    logging.info(f"MQTT interval: {MQTT_INTERVAL_MINUTES} minutes")
    logging.info(f"Daily MQTT send time: {DAILY_SEND_TIME}")
    logging.info(f"Image crop settings - Padding: {CROP_PADDING}px, Min size: {MIN_CROP_SIZE}, Quality: {JPEG_QUALITY}%")
    
    rtsp_url = RTSP_URL
    model = YOLO("yolo11l.pt")
    names = model.names

    last_waiting_log = time.time()
    
    while True:
        try:
            logging.info('Initializing Service...')
            
            cap = initialize_video_capture(rtsp_url)
            if not cap.isOpened():
                logging.error("Failed to open RTSP stream.")
                raise Exception("Failed to open RTSP stream")
            else:
                logging.info(f'Person IN: {person_in}, Person OUT: {person_out}')
                logging.info("RTSP stream opened successfully.")
                logging.info(f"Device ID = {device_id}")
            
            
            #get point    
            # cv2.namedWindow('RGB')
            # cv2.setMouseCallback('RGB', RGB)

            count = 0
            while True:
                ret, frame = cap.read()
                count += 1
                if count % 2 != 0:
                    continue
                if not ret:
                    logging.error("Failed to read frame from RTSP stream")
                    raise Exception("Frame read error or RTSP stream disconnected")

                # Screen Resolution
                # frame = cv2.resize(frame, (1280, 720))
                frame = cv2.resize(frame, (1920, 1080))

                # Create cropped frame for YOLO detection (only the detection region)
                detection_frame = frame[DETECTION_Y_MIN:DETECTION_Y_MAX, :]
                
                # Run YOLO only on the detection region
                results = model.track(detection_frame, persist=True, verbose=False, device=0)

                # IN LINE ( BLUE )
                cv2.putText(frame, 'IN', (lineA[0][0] + 2, lineA[1][1] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3),
                cv2.arrowedLine(frame, (lineA[0][0] + 52, lineA[1][1] - 42), (lineA[0][0] + 52, lineA[1][1] - 20), (255, 0, 0), 2, tipLength=0.4)
                cv2.line(frame, lineA[0], lineA[1], (255, 0, 0), 4)
                
                #OUT LINE ( YELLOW )
                cv2.putText(frame, 'OUT', (lineB[0][0] + 2, lineB[1][1] + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3),
                cv2.arrowedLine(frame, (lineB[0][0] + 80, lineB[1][1] + 42), (lineB[0][0] + 80, lineB[1][1] + 20), (0, 255, 255), 2, tipLength=0.4)
                cv2.line(frame, lineB[0], lineB[1], (0, 255, 255), 4)
                
                # Draw detection region boundaries
                cv2.line(frame, (0, DETECTION_Y_MIN), (1920, DETECTION_Y_MIN), (0, 255, 0), 2)
                cv2.line(frame, (0, DETECTION_Y_MAX), (1920, DETECTION_Y_MAX), (0, 255, 0), 2)
                
                cv2.putText(frame, 'Detection Region', (10, DETECTION_Y_MIN - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                original_frame = frame.copy()

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
                            
                            top_point = ((x1 + x2) // 2, y1)
                            bottom_point = ((x1 + x2) // 2, y2)

                            prev_points = last_points[track_id]

                            # Draw person detection
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cvzone.putTextRect(frame, f'{track_id}', (x1, y1), 1, 1)
                            cv2.circle(frame, top_point, 4, (255, 0, 255), 2)
                            cv2.circle(frame, bottom_point, 4, (0, 255, 255), 2)

                            if prev_points[0] is not None and prev_points[1] is not None:
                                prev_top, prev_bottom = prev_points

                                crossed_A = is_crossing_line(prev_top, top_point, lineA)
                                crossed_B = is_crossing_line(prev_bottom, bottom_point, lineB)

                                if crossed_A:
                                    if state_in.get(track_id):
                                        global interval_person_in
                                        person_in += 1
                                        interval_person_in += 1  # Add this line
                                        class_counts['in'] = person_in

                                        # Update database
                                        db_query(
                                            "UPDATE person_inout SET total_in = %s WHERE id = %s",
                                            (person_in, record_id), commit=True
                                        )
                                        

                                        # Crop and send via MQTT
                                        send_person_in_mqtt(original_frame, record_id, "person_in")

                                        logging.info(f'Person {track_id} In - Total in: {person_in}')
                                        state_in[track_id] = False
                                    else:
                                        state_out[track_id] = True
                                        logging.info(f'Person {track_id} crossed line A (preparing for In)')

                                # When a person exits (crossed_B section):
                                elif crossed_B: 
                                    if state_out.get(track_id):
                                        global interval_person_out
                                        person_out += 1
                                        interval_person_out += 1  # Add this line
                                        class_counts['out'] = person_out

                                        db_query(
                                            "UPDATE person_inout SET total_out = %s WHERE id = %s",
                                            (person_out, record_id), commit=True
                                        )

                                        # Crop and send via MQTT
                                        send_person_in_mqtt(original_frame, record_id, "person_out")

                                        logging.info(f'Person {track_id} OUT - Total OUT: {person_out}')
                                        state_out[track_id] = False 
                                    else:
                                        state_in[track_id] = True
                                        logging.info(f'Person {track_id} crossed line B (preparing for Out)')

                            last_points[track_id] = (top_point, bottom_point)
                
                # Check for interval MQTT sending
                if should_send_interval_mqtt():
                    send_interval_mqtt_data()
                
                if not person_detected:
                    current_time = time.time()
                    if current_time - last_waiting_log >= 60:
                        logging.info("Waiting for person detection...")
                        last_waiting_log = current_time

                # Display counters
                cv2.putText(frame, f'IN: {person_in}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f'OUT: {person_out}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, f'Region Detections: {region_detections}', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                
                #SCREEN
                # cv2.imshow("RGB", frame)
                # if cv2.waitKey(1) & 0xFF == ord("q"):
                #     logging.info("User requested exit")
                #     return

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
            cv2.destroyAllWindows()
            last_points.clear()
            person_history.clear()
            time.sleep(5)
            continue
    
    # Cleanup
    if 'cap' in locals():
        cap.release()
    cv2.destroyAllWindows()
    if mqtt_client:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()

if __name__ == "__main__":
    main()
