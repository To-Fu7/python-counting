
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
import string
import subprocess
import sys
from pathlib import Path
import argparse

# Load environment variables before configuration
load_dotenv('.env')

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

PG_HOST = os.getenv('PG_HOST')
PG_PORT = int(os.getenv('PG_PORT', 5432))
PG_DB = os.getenv('PG_DB')
PG_USER = os.getenv('PG_USER')
PG_PASS = os.getenv('PG_PASS')
device_id = os.getenv('DEVICE_ID')
device_code = os.getenv('DEVICE_CODE')
device_name = os.getenv('DEVICE_NAME')

# Logging configuration with device context
LOG_DEVICE_LABEL = device_name or device_code or device_id or "camera"


class DeviceLogFilter(logging.Filter):
    def filter(self, record):
        record.device = LOG_DEVICE_LABEL
        return True


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(device)s] - %(message)s'
)
logging.getLogger().addFilter(DeviceLogFilter())

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
resolution = ast.literal_eval(os.getenv("SCREEN_RESOLUTION", "[1280, 720]"))

# Optional multi-camera config
MULTI_CAMERA_CONFIG = os.getenv("MULTI_CAMERA_CONFIG")
RUNNING_AS_WORKER = os.getenv("RUNNING_AS_WORKER", "0") == "1"

# Line coordinates (support multiple gates)
POINT_AXIS = os.getenv('POINT_AXIS', 'X')

LINE_OFFSET = os.getenv('LINE_OFFSET', 'X')
offset = int(os.getenv('OFFSET_AMOUNT', 0))


def load_cameras_config(config_path):
    """Load multi-camera config JSON from disk."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Multi-camera config not found at {path}")

    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, list):
        raise ValueError("Camera config must be a list of camera objects")

    return data


def _coerce_list(value, fallback):
    """Ensure lists are serialized for env overrides."""
    if value is None:
        return json.dumps(fallback)
    if isinstance(value, str):
        return value
    return json.dumps(value)


def build_env_for_camera(camera_cfg, base_env):
    """
    Build environment variables for a worker process from a camera config entry.
    """
    env = base_env.copy()
    env["RUNNING_AS_WORKER"] = "1"

    env["DEVICE_ID"] = str(camera_cfg.get("device_id", env.get("DEVICE_ID", "")))
    env["DEVICE_CODE"] = str(camera_cfg.get("device_code", env.get("DEVICE_CODE", "")))
    env["DEVICE_NAME"] = str(camera_cfg.get("device_name", env.get("DEVICE_NAME", "")))

    if "rtsp_url" in camera_cfg:
        env["RTSP_URL"] = str(camera_cfg["rtsp_url"])

    if "screen_resolution" in camera_cfg:
        env["SCREEN_RESOLUTION"] = _coerce_list(camera_cfg["screen_resolution"], resolution)

    env["POINT_AXIS"] = str(camera_cfg.get("point_axis", env.get("POINT_AXIS", POINT_AXIS)))
    env["LINE_OFFSET"] = str(camera_cfg.get("line_offset", env.get("LINE_OFFSET", LINE_OFFSET)))
    env["OFFSET_AMOUNT"] = str(camera_cfg.get("offset_amount", env.get("OFFSET_AMOUNT", offset)))

    # Optional MQTT overrides
    if "mqtt_topic" in camera_cfg:
        env["MQTT_TOPIC"] = str(camera_cfg["mqtt_topic"])
    if "mqtt_interval_topic" in camera_cfg:
        env["MQTT_INTERVAL_TOPIC"] = str(camera_cfg["mqtt_interval_topic"])

    # Optional YOLO model override per camera
    if "yolo_model" in camera_cfg:
        env["YOLO_MODEL"] = str(camera_cfg["yolo_model"])

    # Line definitions
    for line_name, coords in camera_cfg.get("lines", {}).items():
        env[line_name] = json.dumps(coords)

    return env


class CameraProcessor:
    """
    Per-camera processing pipeline that runs in its own thread.
    Each instance maintains isolated state (counters, tracking, MQTT client, etc.)
    while sharing the YOLO model across all cameras.
    """
    def __init__(self, camera_cfg, shared_model, camera_id):
        self.camera_cfg = camera_cfg
        self.shared_model = shared_model
        self.camera_id = camera_id
        self.name = camera_cfg.get("name") or camera_cfg.get("device_name") or f"camera-{camera_id}"
        self.device_id = str(camera_cfg.get("device_id", ""))
        self.device_code = str(camera_cfg.get("device_code", ""))
        self.device_name = str(camera_cfg.get("device_name", ""))
        self.rtsp_url = camera_cfg.get("rtsp_url")
        # Support "original" or null to use original stream resolution
        screen_res = camera_cfg.get("screen_resolution", [1280, 720])
        if screen_res is None or (isinstance(screen_res, str) and screen_res.lower() == "original"):
            self.resolution = "original"
        else:
            self.resolution = screen_res
        self.point_axis = camera_cfg.get("point_axis", "Y")
        self.line_offset = camera_cfg.get("line_offset", "X")
        self.offset_amount = int(camera_cfg.get("offset_amount", 0))
        self.mqtt_topic = camera_cfg.get("mqtt_topic", MQTT_TOPIC)
        self.mqtt_interval_topic = camera_cfg.get("mqtt_interval_topic", MQTT_INTERVAL_TOPIC)
        self.yolo_model_path = camera_cfg.get("yolo_model", YOLO_MODEL)
        
        # Per-camera state (isolated from other cameras)
        self.last_points = defaultdict(lambda: (None, None))
        self.state_in = defaultdict(lambda: False)
        self.state_out = defaultdict(lambda: False)
        self.class_counts = defaultdict(int)
        self.person_history = {}
        self.is_midnight = False
        self.record_id = ''
        self.person_in = 0
        self.person_out = 0
        self.interval_person_in = 0
        self.interval_person_out = 0
        self.last_mqtt_send = None
        self.last_daily_send = None
        self.latest_person_coordinates = []
        
        # Per-camera MQTT client
        self.mqtt_client = None
        
        # Per-camera database connection
        self.pg_conn = None
        self.cursor = None
        
        # Per-camera line pairs and detection region
        self.line_pairs = []
        self.detection_y_min = 0
        self.detection_y_max = 0
        
        # Per-camera health metrics
        self.frame_count = 0
        self.frame_start_time = time.time()
        self.last_heartbeat = time.time()
        self.heartbeat_interval = 60
        self.last_waiting_log = time.time()
        
        # Per-camera logger with device context
        self.logger = logging.getLogger(f"camera.{self.camera_id}")
        self.logger.setLevel(logging.INFO)
        # Avoid duplicate handlers if logger already exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(f'%(asctime)s - %(levelname)s - [{self.name}] - %(message)s'))
            self.logger.addHandler(handler)
        
        # Thread control
        self.stop_event = threading.Event()
        self.thread = None
        
    def _load_line_pairs_from_config(self):
        """Load line pairs from camera config."""
        lines = self.camera_cfg.get("lines", {})
        if not isinstance(lines, dict):
            raise ValueError(f"Camera {self.name}: 'lines' must be a dict")
        
        line_pairs = []
        letters = string.ascii_uppercase
        for i in range(0, len(letters), 2):
            first_letter = letters[i]
            if i + 1 >= len(letters):
                break
            second_letter = letters[i + 1]
            
            first_name = f"line{first_letter}"
            second_name = f"line{second_letter}"
            
            if first_name not in lines:
                continue
                
            first_line = lines[first_name]
            if second_name in lines:
                second_line = lines[second_name]
            else:
                # Generate second line using offset
                second_line = self._compute_offset_line(first_line, self.offset_amount, self.line_offset)
            
            line_pairs.append({
                "in_name": first_name,
                "out_name": second_name,
                "in_line": first_line,
                "out_line": second_line,
            })
        
        if not line_pairs:
            raise ValueError(f"Camera {self.name}: No valid line pairs found")
        
        self.line_pairs = line_pairs
        
        # Calculate detection region
        all_line_points_y = []
        for lp in line_pairs:
            for (x, y) in lp["in_line"] + lp["out_line"]:
                all_line_points_y.append(y)
        
        self.detection_y_min = min(all_line_points_y) - DETECTION_MARGIN
        self.detection_y_max = max(all_line_points_y) + DETECTION_MARGIN
        
        self.logger.info(f"Loaded {len(line_pairs)} line pairs, detection region Y: {self.detection_y_min} to {self.detection_y_max}")
    
    def _compute_offset_line(self, base_line, offset_value, mode):
        """Compute an offset line from base_line."""
        if mode == 'positive' or mode == 'X':
            return [
                (base_line[0][0] + offset_value, base_line[0][1] + offset_value),
                (base_line[1][0] + offset_value, base_line[1][1] + offset_value),
            ]
        elif mode == 'negative':
            return [
                (base_line[0][0] - offset_value, base_line[0][1] - offset_value),
                (base_line[1][0] - offset_value, base_line[1][1] - offset_value),
            ]
        else:
            return base_line
    
    def _init_mqtt(self):
        """Initialize per-camera MQTT client."""
        try:
            self.mqtt_client = mqtt.Client(client_id=f"{self.device_id}_{self.camera_id}")
            if MQTT_USERNAME and MQTT_PASSWORD:
                self.mqtt_client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
            
            def on_connect(client, userdata, flags, rc):
                if rc == 0:
                    self.logger.info(f"MQTT connected for {self.name}")
                else:
                    self.logger.error(f"MQTT connection failed for {self.name}, return code {rc}")
            
            def on_disconnect(client, userdata, rc):
                self.logger.warning(f"MQTT disconnected for {self.name}")
            
            self.mqtt_client.on_connect = on_connect
            self.mqtt_client.on_disconnect = on_disconnect
            self.mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
            self.mqtt_client.loop_start()
        except Exception as e:
            self.logger.error(f"Failed to initialize MQTT for {self.name}: {e}")
            self.mqtt_client = None
    
    def _init_database(self):
        """Initialize per-camera database connection."""
        if DEBUG_MODE:
            return
        
        try:
            self.pg_conn = psycopg2.connect(
                dbname=PG_DB,
                user=PG_USER,
                password=PG_PASS,
                host=PG_HOST,
                port=PG_PORT
            )
            self.cursor = self.pg_conn.cursor()
            self.logger.info(f"Database connected for {self.name}")
        except Exception as e:
            self.logger.error(f"Database connection failed for {self.name}: {e}")
            self.pg_conn = None
            self.cursor = None
    
    def _db_query(self, sql, params=(), commit=False, max_retry=3):
        """Execute database query with retry logic."""
        if DEBUG_MODE:
            return True
        
        if not self.cursor:
            return False
        
        retry = 0
        while retry < max_retry:
            try:
                self.cursor.execute(sql, params)
                if commit:
                    self.pg_conn.commit()
                return True
            except (psycopg2.OperationalError, psycopg2.InterfaceError) as error:
                self.logger.error(f"Database connection lost for {self.name}: {error} (retry {retry+1}/{max_retry})")
                try:
                    self.cursor.close()
                    self.pg_conn.close()
                except:
                    pass
                self._init_database()
                if not self.cursor:
                    time.sleep(2)
                    retry += 1
                    continue
            except Exception as error:
                self.logger.error(f"DB Error for {self.name}: {error}")
                time.sleep(2)
                retry += 1
                continue
        
        return False
    
    def _db_fetch(self, sql, params=(), max_retry=3):
        """Fetch data from database with retry logic."""
        if DEBUG_MODE:
            return None
        
        if not self.cursor:
            return None
        
        retry = 0
        while retry < max_retry:
            try:
                self.cursor.execute(sql, params)
                return self.cursor.fetchone()
            except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
                self.logger.error(f"Database connection lost for {self.name}: {e} (retry {retry+1}/{max_retry})")
                try:
                    self.cursor.close()
                    self.pg_conn.close()
                except:
                    pass
                self._init_database()
                if not self.cursor:
                    time.sleep(2)
                    retry += 1
                    continue
            except Exception as e:
                self.logger.error(f"DB Error for {self.name}: {e}")
                time.sleep(2)
                retry += 1
                continue
        
        return None
    
    def _initialize_counts(self):
        """Initialize counters from database or create new record."""
        self.interval_person_in = 0
        self.interval_person_out = 0
        
        if DEBUG_MODE:
            new_id = str(uuid.uuid4())
            self.record_id = new_id
            self.class_counts['in'] = 0
            self.class_counts['out'] = 0
            self.person_in = 0
            self.person_out = 0
            self.logger.info(f"DEBUG_MODE: Initialized with new record id {new_id}")
            return {"id": new_id}
        
        row = self._db_fetch(
            "SELECT id, total_in, total_out, data, created_at FROM person_inout WHERE device_id = %s ORDER BY created_at DESC LIMIT 1",
            (self.device_id,)
        )
        
        if row:
            last_device_id, total_in, total_out, data, last_created_utc = row
            last_created_local = last_created_utc.astimezone(local_tz)
            last_date_local = last_created_local.date()
            today_local = datetime.datetime.now(local_tz).date()
            
            if last_date_local == today_local:
                self.class_counts['in'] = total_in if total_in is not None else 0
                self.class_counts['out'] = total_out if total_out is not None else 0
                self.person_in = self.class_counts['in']
                self.person_out = self.class_counts['out']
                self.record_id = last_device_id
                self.logger.info(f"Restored counts from DB: IN={self.person_in}, OUT={self.person_out}")
                return {"id": last_device_id}
        
        # Create new record
        new_id = str(uuid.uuid4())
        success = self._db_query(
            "INSERT INTO person_inout (id, device_id, total_in, total_out) VALUES (%s, %s, %s, %s)",
            (new_id, self.device_id, 0, 0), commit=True
        )
        if success:
            self.record_id = new_id
            self.class_counts['in'] = 0
            self.class_counts['out'] = 0
            self.person_in = 0
            self.person_out = 0
            self.logger.info(f"Created new record with id {new_id}")
            return {"id": new_id}
        else:
            self.logger.error("Failed to create new row")
            return None
    
    def _send_person_in_mqtt(self, cropped_image, record_id, event_type="person_in"):
        """Send cropped image via MQTT when person enters/exits."""
        if DEBUG_MODE:
            return
        
        if self.mqtt_client is None:
            return
        
        try:
            _, buffer = cv2.imencode('.jpg', cropped_image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            image_bytes = buffer.tobytes()
            
            payload = {
                "record_id": record_id,
                "device_id": self.device_id,
                "device_code": self.device_code,
                "device_name": self.device_name,
                "timestamp": datetime.datetime.now(local_tz).isoformat(),
                "event": event_type,
                "image": base64.b64encode(image_bytes).decode('utf-8')
            }
            
            result = self.mqtt_client.publish(self.mqtt_topic, json.dumps(payload), qos=1)
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                self.logger.info(f"Person {event_type.upper()} image sent via MQTT for record {record_id}")
        except Exception as e:
            self.logger.error(f"Error sending MQTT message for {self.name}: {e}")
    
    def _should_send_interval_mqtt(self):
        """Check if it's time to send interval MQTT data."""
        current_time = datetime.datetime.now(local_tz)
        
        # Check for daily send time
        daily_hour, daily_minute = map(int, DAILY_SEND_TIME.split(':'))
        if (current_time.hour == daily_hour and current_time.minute == daily_minute and 
            current_time.second < 10):
            if (self.last_daily_send is None or 
                self.last_daily_send.date() != current_time.date()):
                self.last_daily_send = current_time
                return True
        
        # Check for interval send
        if self.last_mqtt_send is None:
            return True
        
        time_diff = current_time - self.last_mqtt_send
        if time_diff.total_seconds() >= (MQTT_INTERVAL_MINUTES * 60):
            return True
        
        return False
    
    def _send_interval_mqtt_data(self):
        """Send interval data via MQTT."""
        if DEBUG_MODE:
            return
        
        if self.mqtt_client is None:
            return
        
        try:
            current_time = datetime.datetime.now(local_tz)
            payload = {
                "record_id": self.record_id,
                "device_id": self.device_id,
                "device_code": self.device_code,
                "device_name": self.device_name,
                "timestamp": current_time.isoformat(),
                "event": "interval_data",
                "data": {
                    "interval_in": self.interval_person_in,
                    "interval_out": self.interval_person_out,
                    "total_in": self.person_in,
                    "total_out": self.person_out,
                    "net_count": self.person_in - self.person_out,
                    "interval_net": self.interval_person_in - self.interval_person_out,
                    "interval_minutes": MQTT_INTERVAL_MINUTES
                }
            }
            
            result = self.mqtt_client.publish(self.mqtt_interval_topic, json.dumps(payload), qos=1)
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                self.logger.info(
                    f"Interval data sent - IN: {self.interval_person_in}, OUT: {self.interval_person_out}, "
                    f"Total IN: {self.person_in}, Total OUT: {self.person_out}"
                )
                self.last_mqtt_send = current_time
                self.interval_person_in = 0
                self.interval_person_out = 0
        except Exception as e:
            self.logger.error(f"Error sending interval MQTT data for {self.name}: {e}")
    
    def _should_reset(self):
        """Check if it's time to reset counters (midnight)."""
        now = datetime.datetime.now(local_tz)
        return now.hour == 0 and now.minute == 0 and now.second < 10
    
    def _reset_counts(self):
        """Reset counts at midnight."""
        self._send_interval_mqtt_data()
        
        self.person_in, self.person_out = 0, 0
        self.interval_person_in, self.interval_person_out = 0, 0
        self.class_counts.clear()
        self.class_counts['in'] = 0
        self.class_counts['out'] = 0
        self.state_in.clear()
        self.state_out.clear()
        self.person_history.clear()
        
        new_id = str(uuid.uuid4())
        
        if DEBUG_MODE:
            self.record_id = new_id
            self.last_mqtt_send = None
            self.logger.info("== Midnight Reached: Totals Reset (DEBUG_MODE) ==")
        else:
            success = self._db_query(
                "INSERT INTO person_inout (id, device_id, total_in, total_out) VALUES (%s, %s, %s, %s)",
                (new_id, self.device_id, 0, 0), commit=True
            )
            if success:
                self.record_id = new_id
                self.last_mqtt_send = None
                self.logger.info("== Midnight Reached: Totals Reset ==")
    
    def _is_crossing_line(self, p1, p2, line):
        """Check if line segment p1-p2 crosses the given line."""
        if p1 is None or p2 is None:
            return False
        try:
            person_line = LineString([p1, p2])
            the_line = LineString(line)
            return person_line.crosses(the_line)
        except Exception as e:
            self.logger.warning(f"Error checking line crossing: {e}")
            return False
    
    def _is_in_detection_region(self, box):
        """Check if detected person is within the detection region."""
        x1, y1, x2, y2 = box
        return not (y2 < self.detection_y_min or y1 > self.detection_y_max)
    
    def _crop_image(self, frame, box, padding=None):
        """Crop image around detected person."""
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
            aspect_ratio = crop_w / crop_h
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
    
    def run(self):
        """Main processing loop for this camera."""
        try:
            # Initialize line pairs
            self._load_line_pairs_from_config()
            
            # Initialize MQTT
            self._init_mqtt()
            
            # Initialize database
            self._init_database()
            
            # Initialize counts
            last_data_id = self._initialize_counts()
            if not last_data_id and not DEBUG_MODE:
                self.logger.error("Failed to initialize database")
                return
            
            self.logger.info(f"Starting processing for {self.name} (device_id: {self.device_id})")
            self.logger.info(f"Detection region: Y from {self.detection_y_min} to {self.detection_y_max}")
            self.logger.info(f"MQTT interval: {MQTT_INTERVAL_MINUTES} minutes")
            
            # Get video source
            video_source = self.rtsp_url
            if not video_source or video_source.strip() == '':
                fallback_video = '1.mp4'
                self.logger.warning(f"RTSP_URL not set, falling back to {fallback_video}")
                video_source = fallback_video
            
            # Initialize video capture
            cap = cv2.VideoCapture(video_source, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 10)
            
            if not cap.isOpened():
                self.logger.error(f"Failed to open video source: {video_source}")
                return
            
            # Get original resolution if needed
            if self.resolution == "original":
                # Read first frame to get dimensions
                ret_test, test_frame = cap.read()
                if ret_test and test_frame is not None:
                    h, w = test_frame.shape[:2]
                    self.resolution = [w, h]
                    self.logger.info(f"Using original stream resolution: {w}x{h}")
                else:
                    self.logger.warning("Failed to get original resolution, falling back to 1280x720")
                    self.resolution = [1280, 720]
                # Reset capture to beginning
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            self.logger.info(f"Video source opened: {video_source}")
            self.logger.info(f"Resolution: {self.resolution[0]}x{self.resolution[1]}")
            self.logger.info(f"Person IN: {self.person_in}, Person OUT: {self.person_out}")
            
            names = self.shared_model.names
            count = 0
            
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                count += 1
                if count % 2 != 0:
                    continue
                
                if not ret:
                    self.logger.error(f"Failed to read frame from {video_source}")
                    break
                
                # Resize frame only if not using original resolution
                if self.resolution != "original":
                    frame = cv2.resize(frame, (int(self.resolution[0]), int(self.resolution[1])))
                
                # Create cropped frame for YOLO detection
                detection_frame = frame[self.detection_y_min:self.detection_y_max, :]
                
                # Run YOLO (shared model, thread-safe for inference)
                results = self.shared_model.track(detection_frame, persist=True, verbose=False)
                
                # Draw lines
                for lp in self.line_pairs:
                    cv2.line(frame, lp["in_line"][0], lp["in_line"][1], (255, 0, 0), 4)
                    cv2.line(frame, lp["out_line"][0], lp["out_line"][1], (0, 255, 255), 4)
                
                # Draw detection region
                cv2.line(frame, (0, self.detection_y_min), (1920, self.detection_y_min), (0, 255, 0), 2)
                cv2.line(frame, (0, self.detection_y_max), (1920, self.detection_y_max), (0, 222, 0), 2)
                cv2.putText(frame, 'Detection Region', (10, self.detection_y_min - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                original_frame = frame.copy()
                person_detected = False
                region_detections = 0
                self.latest_person_coordinates = []
                
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
                            
                            # Adjust box coordinates
                            x1, y1, x2, y2 = box
                            y1 += self.detection_y_min
                            y2 += self.detection_y_min
                            
                            # Store coordinates
                            person_coord = {
                                "track_id": int(track_id),
                                "x": int(x1),
                                "y": int(y1),
                                "w": int(x2 - x1),
                                "h": int(y2 - y1),
                                "confidence": float(conf),
                                "center_x": int((x1 + x2) // 2),
                                "center_y": int((y1 + y2) // 2)
                            }
                            self.latest_person_coordinates.append(person_coord)
                            
                            # Calculate points
                            if self.point_axis == "Y":
                                first_point = ((x1 + x2) // 2, y1)
                                second_point = ((x1 + x2) // 2, y2)
                            elif self.point_axis == "X":
                                first_point = (x1, (y1 + y2) // 2)
                                second_point = (x2, (y1 + y2) // 2)
                            
                            prev_points = self.last_points[track_id]
                            
                            # Draw detection
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cvzone.putTextRect(frame, f'{track_id}', (x1, y1), 1, 1)
                            cv2.circle(frame, first_point, 4, (255, 0, 0), 2)
                            cv2.circle(frame, second_point, 4, (0, 255, 255), 2)
                            
                            if prev_points[0] is not None and prev_points[1] is not None:
                                prev_top, prev_bottom = prev_points
                                
                                # Check crossings
                                for gate_index, lp in enumerate(self.line_pairs):
                                    gate_key = (track_id, gate_index)
                                    
                                    crossed_A = self._is_crossing_line(prev_top, first_point, lp["in_line"])
                                    crossed_B = self._is_crossing_line(prev_bottom, second_point, lp["out_line"])
                                    
                                    if crossed_A:
                                        if self.state_in.get(gate_key):
                                            self.person_in += 1
                                            self.interval_person_in += 1
                                            self.class_counts['in'] = self.person_in
                                            
                                            if not DEBUG_MODE:
                                                self._db_query(
                                                    "UPDATE person_inout SET total_in = %s WHERE id = %s",
                                                    (self.person_in, self.record_id), commit=True
                                                )
                                            
                                            self._send_person_in_mqtt(original_frame, self.record_id, "person_in")
                                            self.logger.info(
                                                f'Person {track_id} In through gate {gate_index} - Total in: {self.person_in}'
                                            )
                                            self.state_in[gate_key] = False
                                        else:
                                            self.state_out[gate_key] = True
                                    elif crossed_B:
                                        if self.state_out.get(gate_key):
                                            self.person_out += 1
                                            self.interval_person_out += 1
                                            self.class_counts['out'] = self.person_out
                                            
                                            if not DEBUG_MODE:
                                                self._db_query(
                                                    "UPDATE person_inout SET total_out = %s WHERE id = %s",
                                                    (self.person_out, self.record_id), commit=True
                                                )
                                            
                                            self._send_person_in_mqtt(original_frame, self.record_id, "person_out")
                                            self.logger.info(
                                                f'Person {track_id} OUT through gate {gate_index} - Total OUT: {self.person_out}'
                                            )
                                            self.state_out[gate_key] = False
                                        else:
                                            self.state_in[gate_key] = True
                            
                            self.last_points[track_id] = (first_point, second_point)
                
                # Check for interval MQTT sending
                if self._should_send_interval_mqtt():
                    self._send_interval_mqtt_data()
                
                # Health metrics and heartbeat
                self.frame_count += 1
                current_time = time.time()
                if current_time - self.last_heartbeat >= self.heartbeat_interval:
                    elapsed = current_time - self.frame_start_time
                    fps = self.frame_count / elapsed if elapsed > 0 else 0
                    self.logger.info(
                        f"[HEALTH] {self.name} - Frames: {self.frame_count}, FPS: {fps:.2f}, "
                        f"IN: {self.person_in}, OUT: {self.person_out}, "
                        f"Active tracks: {len(self.last_points)}, Stream: {video_source}"
                    )
                    self.last_heartbeat = current_time
                    self.frame_count = 0
                    self.frame_start_time = current_time
                
                if not person_detected:
                    if current_time - self.last_waiting_log >= 60:
                        self.logger.info("Waiting for person detection...")
                        self.last_waiting_log = current_time
                
                # Display counters
                cv2.putText(frame, f'IN: {self.person_in}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f'OUT: {self.person_out}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, f'Region Detections: {region_detections}', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, f'{self.name}', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Handle midnight reset
                if self._should_reset() and not self.is_midnight:
                    self._reset_counts()
                    self.is_midnight = True
                
                if not self._should_reset() and self.is_midnight:
                    self.is_midnight = False
                
                # Debug display
                if DEBUG_MODE:
                    cv2.imshow(f"RGB-{self.name}", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        self.logger.info("User requested exit")
                        break
        
        except Exception as error:
            self.logger.error(f"Error in camera {self.name}: {str(error)}. Restarting in 5 seconds...")
            time.sleep(5)
        finally:
            if 'cap' in locals():
                cap.release()
            if self.mqtt_client:
                self.mqtt_client.loop_stop()
                self.mqtt_client.disconnect()
            if self.cursor:
                self.cursor.close()
            if self.pg_conn:
                self.pg_conn.close()
            cv2.destroyAllWindows()
    
    def start(self):
        """Start the camera processing thread."""
        self.thread = threading.Thread(target=self.run, daemon=True, name=f"Camera-{self.name}")
        self.thread.start()
        return self.thread
    
    def stop(self):
        """Stop the camera processing thread."""
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5)


def start_multi_camera(config_path):
    """
    Start multi-camera processing using threads instead of subprocesses.
    Shares YOLO model across all cameras for efficiency.
    """
    cameras = load_cameras_config(config_path)
    if not cameras:
        raise RuntimeError("Camera config is empty; add at least one camera entry.")
    
    # Load shared YOLO model (thread-safe for inference)
    shared_model = YOLO(YOLO_MODEL)
    logging.info(f"[ORCHESTRATOR] Loaded shared YOLO model: {YOLO_MODEL}")
    
    # Create camera processors
    processors = []
    logging.info(f"[ORCHESTRATOR] Starting {len(cameras)} camera processors from {config_path}")
    
    for idx, camera_cfg in enumerate(cameras):
        processor = CameraProcessor(camera_cfg, shared_model, idx)
        processors.append(processor)
        friendly_name = processor.name
        device_id = processor.device_id
        rtsp_url = processor.rtsp_url
        logging.info(f"[ORCHESTRATOR] Created processor for {friendly_name} (device_id: {device_id}, stream: {rtsp_url})")
    
    # Start all processors
    threads = []
    for processor in processors:
        thread = processor.start()
        threads.append((processor, thread))
        logging.info(f"[ORCHESTRATOR] Started thread for {processor.name}")
    
    logging.info(f"[ORCHESTRATOR] All {len(threads)} camera processors started. Monitoring threads...")
    
    try:
        # Wait for all threads
        for processor, thread in threads:
            thread.join()
            logging.info(f"Camera processor {processor.name} thread exited")
    except KeyboardInterrupt:
        logging.info("Received stop signal; stopping camera processors...")
        for processor, _ in threads:
            processor.stop()
        for processor, thread in threads:
            thread.join(timeout=5)


def _draw_camera_lines(frame, camera_cfg):
    """
    Draw lines from cameras.json entry onto the frame.
    Expects camera_cfg["lines"] keys like lineA/lineB/lineC/lineD...
    """
    lines = camera_cfg.get("lines") or {}
    if not isinstance(lines, dict):
        return frame

    # Draw known pairs (A/B, C/D, ...)
    letters = string.ascii_uppercase
    for i in range(0, len(letters), 2):
        a = f"line{letters[i]}"
        b = f"line{letters[i + 1]}" if i + 1 < len(letters) else None
        if a not in lines:
            continue
        try:
            in_line = lines[a]
            if b and b in lines:
                out_line = lines[b]
            else:
                out_line = None

            # Normalize to tuples
            in_p1, in_p2 = tuple(in_line[0]), tuple(in_line[1])
            cv2.line(frame, in_p1, in_p2, (255, 0, 0), 4)  # IN (blue)

            if out_line is not None:
                out_p1, out_p2 = tuple(out_line[0]), tuple(out_line[1])
                cv2.line(frame, out_p1, out_p2, (0, 255, 255), 4)  # OUT (yellow)
        except Exception as e:
            logging.warning(f"Failed drawing lines for {a}/{b}: {e}")

    return frame


def preview_cameras(config_path):
    """
    Debug helper: open each camera stream and show a frame with line overlays.
    Best used outside Docker on a desktop with a display.
    """
    cameras = load_cameras_config(config_path)
    if not cameras:
        raise RuntimeError("Camera config is empty; add at least one camera entry.")

    for idx, camera_cfg in enumerate(cameras):
        name = camera_cfg.get("name") or camera_cfg.get("device_name") or f"camera-{idx}"
        rtsp_url = camera_cfg.get("rtsp_url")
        screen_res = camera_cfg.get("screen_resolution") or resolution
        
        # Support "original" or null to use original stream resolution
        if screen_res is None or (isinstance(screen_res, str) and screen_res.lower() == "original"):
            res = "original"
        else:
            res = screen_res

        logging.info(f"[preview] Opening {name}: {rtsp_url}")
        cap = initialize_video_capture(rtsp_url)
        if not cap.isOpened():
            logging.error(f"[preview] Failed to open stream for {name}")
            cap.release()
            continue

        ret, frame = cap.read()
        if not ret or frame is None:
            logging.error(f"[preview] Failed to read frame for {name}")
            cap.release()
            continue
        
        # Get original resolution if needed
        if res == "original":
            h, w = frame.shape[:2]
            res = [w, h]
            logging.info(f"[preview] Using original stream resolution: {w}x{h}")
        
        cap.release()

        try:
            frame = cv2.resize(frame, (int(res[0]), int(res[1])))
        except Exception:
            pass

        frame = _draw_camera_lines(frame, camera_cfg)
        cv2.putText(frame, f"{name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        win = f"preview: {name}"
        cv2.imshow(win, frame)
        logging.info("[preview] Press any key for next camera, or 'q' to quit.")
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow(win)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


def debug_line_editor(config_path, camera_index=0):
    """
    Interactive debug mode for setting line points.
    Similar to --preview but allows clicking to set line points interactively.
    Uses the RGB mouse callback function to capture clicks.
    """
    global line_edit_state
    
    cameras = load_cameras_config(config_path)
    if not cameras:
        raise RuntimeError("Camera config is empty; add at least one camera entry.")
    
    if camera_index >= len(cameras):
        raise ValueError(f"Camera index {camera_index} out of range (0-{len(cameras)-1})")
    
    camera_cfg = cameras[camera_index]
    name = camera_cfg.get("name") or camera_cfg.get("device_name") or f"camera-{camera_index}"
    rtsp_url = camera_cfg.get("rtsp_url")
    screen_res = camera_cfg.get("screen_resolution") or resolution
    
    # Support "original" or null to use original stream resolution
    if screen_res is None or (isinstance(screen_res, str) and screen_res.lower() == "original"):
        res = "original"
    else:
        res = screen_res
    
    logging.info(f"[debug-line-editor] Opening {name}: {rtsp_url}")
    cap = initialize_video_capture(rtsp_url)
    if not cap.isOpened():
        logging.error(f"[debug-line-editor] Failed to open stream for {name}")
        cap.release()
        return
    
    # Read frame to get dimensions
    ret, frame = cap.read()
    if not ret or frame is None:
        logging.error(f"[debug-line-editor] Failed to read frame for {name}")
        cap.release()
        return
    
    # Get original resolution if needed
    if res == "original":
        h, w = frame.shape[:2]
        res = [w, h]
        logging.info(f"[debug-line-editor] Using original stream resolution: {w}x{h}")
    else:
        frame = cv2.resize(frame, (int(res[0]), int(res[1])))
    
    cap.release()
    
    # Initialize line editing state
    line_edit_state["editing"] = True
    line_edit_state["current_line"] = "lineA"
    line_edit_state["current_point_index"] = 0
    line_edit_state["points"] = {}
    line_edit_state["preview_point"] = None
    
    # Load existing lines if any
    existing_lines = camera_cfg.get("lines", {})
    for line_name, line_coords in existing_lines.items():
        if isinstance(line_coords, list) and len(line_coords) == 2:
            line_edit_state["points"][line_name] = [list(line_coords[0]), list(line_coords[1])]
    
    logging.info("=" * 60)
    logging.info("INTERACTIVE LINE EDITOR MODE")
    logging.info("=" * 60)
    logging.info("Instructions:")
    logging.info("  - Left click to set line points (first point, then second point)")
    logging.info("  - Right click to reset current line")
    logging.info("  - Press 's' to save and update cameras.json")
    logging.info("  - Press 'q' to quit without saving")
    logging.info("  - Press 'n' to move to next line")
    logging.info(f"Currently editing: {line_edit_state['current_line']} - click to set first point")
    logging.info("=" * 60)
    
    # Create window and set mouse callback
    cv2.namedWindow('RGB')
    cv2.setMouseCallback('RGB', RGB)
    
    while True:
        # Create display frame
        display_frame = frame.copy()
        
        # Draw existing lines
        for line_name, points in line_edit_state["points"].items():
            if points[0] is not None and points[1] is not None:
                color = (255, 0, 0) if "A" in line_name or "C" in line_name or "E" in line_name else (0, 255, 255)
                cv2.line(display_frame, tuple(points[0]), tuple(points[1]), color, 4)
                # Draw points
                cv2.circle(display_frame, tuple(points[0]), 6, (0, 255, 0), -1)
                cv2.circle(display_frame, tuple(points[1]), 6, (0, 255, 0), -1)
                cv2.putText(display_frame, line_name, (points[0][0] + 10, points[0][1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw current line being edited
        current_line = line_edit_state["current_line"]
        if current_line in line_edit_state["points"]:
            points = line_edit_state["points"][current_line]
            point_idx = line_edit_state["current_point_index"]
            
            # Draw first point if set
            if points[0] is not None:
                cv2.circle(display_frame, tuple(points[0]), 8, (0, 255, 0), -1)
                # Draw line from first point to preview or second point
                if points[1] is not None:
                    cv2.line(display_frame, tuple(points[0]), tuple(points[1]), (255, 0, 0), 4)
                elif line_edit_state.get("preview_point") is not None:
                    cv2.line(display_frame, tuple(points[0]), tuple(line_edit_state["preview_point"]), (128, 128, 128), 2)
            
            # Draw second point if set
            if points[1] is not None:
                cv2.circle(display_frame, tuple(points[1]), 8, (0, 255, 0), -1)
            
            # Draw preview point
            if point_idx == 0 and line_edit_state.get("preview_point") is not None:
                cv2.circle(display_frame, tuple(line_edit_state["preview_point"]), 6, (255, 255, 0), 2)
            elif point_idx == 1 and points[0] is not None and line_edit_state.get("preview_point") is not None:
                cv2.circle(display_frame, tuple(line_edit_state["preview_point"]), 6, (255, 255, 0), 2)
        
        # Show instructions
        cv2.putText(display_frame, f"Editing: {current_line} (point {line_edit_state['current_point_index'] + 1}/2)", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(display_frame, "Left click: Set point | Right click: Reset | 's': Save | 'q': Quit | 'n': Next line", 
                   (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show mouse position
        if line_edit_state.get("preview_point") is not None:
            mx, my = line_edit_state["preview_point"]
            cv2.putText(display_frame, f"Mouse: [{mx}, {my}]", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.imshow('RGB', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            logging.info("Quitting without saving")
            break
        elif key == ord('s'):
            # Save to cameras.json
            lines_to_save = {}
            for line_name, points in line_edit_state["points"].items():
                if points[0] is not None and points[1] is not None:
                    lines_to_save[line_name] = [points[0], points[1]]
            
            if lines_to_save:
                camera_cfg["lines"] = lines_to_save
                cameras[camera_index] = camera_cfg
                
                # Write back to file
                config_file = Path(config_path)
                with config_file.open("w", encoding="utf-8") as f:
                    json.dump(cameras, f, indent=2)
                
                logging.info(f"Saved {len(lines_to_save)} lines to {config_path}")
                logging.info(f"Lines saved: {list(lines_to_save.keys())}")
            else:
                logging.warning("No complete lines to save")
            break
        elif key == ord('n'):
            # Move to next line
            letters = string.ascii_uppercase
            current_letter = current_line[-1]
            current_letter_idx = ord(current_letter) - ord('A')
            if current_letter_idx + 1 < len(letters):
                next_letter = letters[current_letter_idx + 1]
                line_edit_state["current_line"] = f"line{next_letter}"
                line_edit_state["current_point_index"] = 0
                if line_edit_state["current_line"] not in line_edit_state["points"]:
                    line_edit_state["points"][line_edit_state["current_line"]] = [None, None]
                logging.info(f"Now editing {line_edit_state['current_line']} - click to set first point")
    
    cv2.destroyAllWindows()
    line_edit_state["editing"] = False


def _compute_offset_line(base_line, offset_value, mode):
    """Compute an offset line from base_line using LINE_OFFSET and OFFSET_AMOUNT."""
    if mode == 'positive':
        return [
            (base_line[0][0] + offset_value, base_line[0][1] + offset_value),
            (base_line[1][0] + offset_value, base_line[1][1] + offset_value),
        ]
    elif mode == 'negative':
        return [
            (base_line[0][0] - offset_value, base_line[0][1] - offset_value),
            (base_line[1][0] - offset_value, base_line[1][1] - offset_value),
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
        the second line is generated using LINE_OFFSET and OFFSET_AMOUNT.
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
            second_line = _compute_offset_line(first_line, offset, LINE_OFFSET)

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


LINE_PAIRS = None
lineA = None
lineB = None

# Detection region parameters (initialized at runtime)
DETECTION_MARGIN = 160
DETECTION_Y_MIN = 0
DETECTION_Y_MAX = 0


def init_lines_and_detection_region():
    """
    Initialize LINE_PAIRS and detection region boundaries.
    Must be called at runtime (worker/single-camera), not at import-time.
    """
    global LINE_PAIRS, lineA, lineB, DETECTION_Y_MIN, DETECTION_Y_MAX

    LINE_PAIRS = load_line_pairs_from_env()
    lineA = LINE_PAIRS[0]["in_line"]
    lineB = LINE_PAIRS[0]["out_line"]

    logging.info(f"Total line pairs loaded: {len(LINE_PAIRS)}")

    all_line_points_y = []
    for lp in LINE_PAIRS:
        for (x, y) in lp["in_line"] + lp["out_line"]:
            all_line_points_y.append(y)

    DETECTION_Y_MIN = min(all_line_points_y) - DETECTION_MARGIN
    DETECTION_Y_MAX = max(all_line_points_y) + DETECTION_MARGIN

# Image quality settings
CROP_PADDING = 30
MIN_CROP_SIZE = (128, 128)
JPEG_QUALITY = 95

# YOLO CONFIG
YOLO_MODEL = os.getenv('YOLO_MODEL', 'yolo11n.pt')

# DEBUG MODE
DEBUG_MODE = os.getenv('DEBUG_MODE', 'true').lower() == 'true'

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

def initialize_video_capture(video_source):
    """Initialize video capture with the given video source (RTSP URL or file path)"""
    logging.info(f'Initializing video capture with source: {video_source}')
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

# Global variables for interactive line editing
line_edit_state = {
    "current_line": None,  # "lineA", "lineB", etc.
    "points": {},  # {"lineA": [point1, point2], "lineB": [point1, point2]}
    "current_point_index": 0,  # 0 for first point, 1 for second point
    "editing": False
}

def RGB(event, x, y, flags, param):
    """Mouse callback function for RGB window - interactive line point editor"""
    global line_edit_state
    
    if event == cv2.EVENT_MOUSEMOVE:
        # Show current mouse position
        point = [x, y]
        if line_edit_state["editing"]:
            # Update preview point
            line_edit_state["preview_point"] = point
    
    elif event == cv2.EVENT_LBUTTONDOWN:
        # Left click to set point
        if not line_edit_state["editing"]:
            return
        
        current_line = line_edit_state["current_line"]
        point_idx = line_edit_state["current_point_index"]
        
        if current_line not in line_edit_state["points"]:
            line_edit_state["points"][current_line] = [None, None]
        
        line_edit_state["points"][current_line][point_idx] = [x, y]
        logging.info(f"Set {current_line} point {point_idx + 1}: [{x}, {y}]")
        
        # Move to next point or next line
        if point_idx == 0:
            line_edit_state["current_point_index"] = 1
            logging.info(f"Click to set second point for {current_line}")
        else:
            # Line complete, move to next line
            letters = string.ascii_uppercase
            current_letter_idx = ord(current_line[-1]) - ord('A')
            if current_letter_idx + 1 < len(letters):
                next_letter = letters[current_letter_idx + 1]
                line_edit_state["current_line"] = f"line{next_letter}"
                line_edit_state["current_point_index"] = 0
                logging.info(f"Line {current_line} complete. Now editing {line_edit_state['current_line']} - click to set first point")
            else:
                logging.info(f"All lines complete! Press 's' to save or 'q' to quit")
    
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Right click to cancel current line or go back
        if line_edit_state["editing"]:
            current_line = line_edit_state["current_line"]
            if current_line in line_edit_state["points"]:
                line_edit_state["points"][current_line] = [None, None]
                line_edit_state["current_point_index"] = 0
                logging.info(f"Reset {current_line}, click to set first point again")

def main():
    """Main function"""
    global person_in, person_out, is_midnight, record_id, latest_person_coordinates
    
    # Initialize line pairs + detection region (worker/single-camera runtime)
    init_lines_and_detection_region()
    
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
    logging.info(f"Detection region: Y from {DETECTION_Y_MIN} to {DETECTION_Y_MAX} (margin: {DETECTION_MARGIN}px)")
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
    names = model.names

    last_waiting_log = time.time()
    last_heartbeat = time.time()
    HEARTBEAT_INTERVAL = 60  # Log heartbeat every 60 seconds
    
    # Metrics tracking
    frame_count = 0
    frame_start_time = time.time()
    
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
            
            
            # Get original resolution if needed
            if resolution == "original" or (isinstance(resolution, str) and resolution.lower() == "original"):
                # Read first frame to get dimensions
                ret_test, test_frame = cap.read()
                if ret_test and test_frame is not None:
                    h, w = test_frame.shape[:2]
                    resolution = [w, h]
                    logging.info(f"Using original stream resolution: {w}x{h}")
                else:
                    logging.warning("Failed to get original resolution, falling back to 1280x720")
                    resolution = [1280, 720]
                # Reset capture to beginning
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            #get point   
            if DEBUG_MODE: 
                cv2.namedWindow('RGB')
                cv2.setMouseCallback('RGB', RGB)

            count = 0
            frame_count = 0
            frame_start_time = time.time()
            last_heartbeat = time.time()
            
            while True:
                ret, frame = cap.read()
                count += 1
                if count % 2 != 0:
                    continue
                if not ret:
                    logging.error(f"Failed to read frame from video source: {video_source}")
                    raise Exception(f"Frame read error or video source disconnected: {video_source}")

                # Screen Resolution - only resize if not using original
                if resolution != "original" and not (isinstance(resolution, str) and resolution.lower() == "original"):
                    frame = cv2.resize(frame, (resolution[0], resolution[1]))

                # Create cropped frame for YOLO detection (only the detection region)
                detection_frame = frame[DETECTION_Y_MIN:DETECTION_Y_MAX, :]
                
                # Run YOLO only on the detection region
                results = model.track(detection_frame, persist=True, verbose=False)

                # Draw all IN/OUT lines
                for lp in LINE_PAIRS:
                    # IN LINE ( BLUE ) (BGR)
                    cv2.line(frame, lp["in_line"][0], lp["in_line"][1], (255, 0, 0), 4)
                    # OUT LINE ( YELLOW ) (BGR)
                    cv2.line(frame, lp["out_line"][0], lp["out_line"][1], (0, 255, 255), 4)
                
                # Draw detection region boundaries
                cv2.line(frame, (0, DETECTION_Y_MIN), (1920, DETECTION_Y_MIN), (0, 255, 0), 2)
                cv2.line(frame, (0, DETECTION_Y_MAX), (1920, DETECTION_Y_MAX), (0, 222, 0), 2)
                
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
                            
                            
                            if POINT_AXIS == "Y":
                                first_point = ((x1 + x2) // 2, y1)
                                second_point = ((x1 + x2) // 2, y2)
                            elif POINT_AXIS == "X":
                                first_point = (x1, (y1 + y2) // 2)
                                second_point = (x2, (y1 + y2) // 2)

                            prev_points = last_points[track_id]

                            # Draw person detection
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cvzone.putTextRect(frame, f'{track_id}', (x1, y1), 1, 1)
                            cv2.circle(frame, first_point, 4, (255, 0, 0), 2)
                            cv2.circle(frame, second_point, 4, (0, 255, 255), 2)

                            if prev_points[0] is not None and prev_points[1] is not None:
                                prev_top, prev_bottom = prev_points
                                # Check crossings against all line pairs
                                for gate_index, lp in enumerate(LINE_PAIRS):
                                    gate_key = (track_id, gate_index)

                                    crossed_A = is_crossing_line(prev_top, first_point, lp["in_line"])
                                    crossed_B = is_crossing_line(prev_bottom, second_point, lp["out_line"])

                                    if crossed_A:
                                        if state_in.get(gate_key):
                                            global interval_person_in
                                            person_in += 1
                                            interval_person_in += 1
                                            class_counts['in'] = person_in

                                            # Update database
                                            if not DEBUG_MODE:
                                                db_query(
                                                    "UPDATE person_inout SET total_in = %s WHERE id = %s",
                                                    (person_in, record_id), commit=True
                                                )
                                            else:
                                                logging.info("DEBUG_MODE: Skipping DB update for person_in")

                                            # Crop and send via MQTT
                                            send_person_in_mqtt(original_frame, record_id, "person_in")

                                            logging.info(
                                                f'Person {track_id} In through gate {gate_index} - Total in: {person_in}'
                                            )
                                            state_in[gate_key] = False
                                        else:
                                            state_out[gate_key] = True
                                            logging.info(
                                                f'Person {track_id} crossed IN line of gate {gate_index} (preparing for In)'
                                            )

                                    # When a person exits (crossed_B section):
                                    elif crossed_B:
                                        if state_out.get(gate_key):
                                            global interval_person_out
                                            person_out += 1
                                            interval_person_out += 1
                                            class_counts['out'] = person_out

                                            if not DEBUG_MODE:
                                                db_query(
                                                    "UPDATE person_inout SET total_out = %s WHERE id = %s",
                                                    (person_out, record_id), commit=True
                                                )
                                            else:
                                                logging.info("DEBUG_MODE: Skipping DB update for person_out")

                                            # Crop and send via MQTT
                                            send_person_in_mqtt(original_frame, record_id, "person_out")

                                            logging.info(
                                                f'Person {track_id} OUT through gate {gate_index} - Total OUT: {person_out}'
                                            )
                                            state_out[gate_key] = False
                                        else:
                                            state_in[gate_key] = True
                                            logging.info(
                                                f'Person {track_id} crossed OUT line of gate {gate_index} (preparing for Out)'
                                            )

                            last_points[track_id] = (first_point, second_point)
                
                # Check for interval MQTT sending
                if should_send_interval_mqtt():
                    send_interval_mqtt_data()
                
                # Health metrics and heartbeat logging
                frame_count += 1
                current_time = time.time()
                if current_time - last_heartbeat >= HEARTBEAT_INTERVAL:
                    elapsed = current_time - frame_start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    logging.info(
                        f"[HEALTH] Frames: {frame_count}, FPS: {fps:.2f}, "
                        f"IN: {person_in}, OUT: {person_out}, "
                        f"Active tracks: {len(last_points)}, Stream: {video_source}"
                    )
                    last_heartbeat = current_time
                    frame_count = 0
                    frame_start_time = current_time
                
                if not person_detected:
                    if current_time - last_waiting_log >= 60:
                        logging.info("Waiting for person detection...")
                        last_waiting_log = current_time

                # Display counters
                cv2.putText(frame, f'IN: {person_in}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f'OUT: {person_out}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, f'Region Detections: {region_detections}', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                
                #SCREEN
                if DEBUG_MODE:
                    cv2.imshow("RGB", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        logging.info("User requested exit")
                        return

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true", help="Preview cameras.json streams with line overlays")
    parser.add_argument("--config", default=None, help="Path to cameras.json for preview (defaults to MULTI_CAMERA_CONFIG)")
    parser.add_argument("--debug-lines", action="store_true", help="Interactive line point editor (similar to --preview but with click-to-set functionality)")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index for --debug-lines (default: 0)")
    args = parser.parse_args()

    if args.debug_lines:
        cfg = args.config or MULTI_CAMERA_CONFIG
        if not cfg:
            raise RuntimeError("Set MULTI_CAMERA_CONFIG or pass --config for --debug-lines")
        debug_line_editor(cfg, args.camera_index)
    elif args.preview:
        cfg = args.config or MULTI_CAMERA_CONFIG
        if not cfg:
            raise RuntimeError("Set MULTI_CAMERA_CONFIG or pass --config for --preview")
        preview_cameras(cfg)
    elif MULTI_CAMERA_CONFIG and not RUNNING_AS_WORKER:
        start_multi_camera(MULTI_CAMERA_CONFIG)
    else:
        main()
