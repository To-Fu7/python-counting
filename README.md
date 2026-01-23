# Line Crossing Detection

# Setup Docker
## WINDOWS
--
## LINUX
--

# Debug Test Python

- install requirement
...
pip install -r requirements.txt
...

- install torch cuda support if using cuda
...
pip install torch==2.7.1+cu126 \
    torchvision==0.22.1+cu126 \
    --index-url https://download.pytorch.org/whl/cu126
... 

- Run Code in Terminal
...
python ./main.py
...

## Running multiple cameras in one container

The service supports running multiple cameras concurrently in a single container using threading. This approach shares the YOLO model and resources across all cameras for efficient resource usage.

### Setup Steps

1. **Configure shared infrastructure** (once in `.env`):
   - Database connection settings (`PG_HOST`, `PG_PORT`, `PG_DB`, `PG_USER`, `PG_PASS`)
   - MQTT broker settings (`MQTT_BROKER`, `MQTT_PORT`, `MQTT_USERNAME`, `MQTT_PASSWORD`)
   - Optional: `MULTI_CAMERA_CONFIG=./config/cameras.json` (or set via docker-compose environment)

2. **Define cameras in `config/cameras.json`**:
   Each camera entry should include:
   - `name`: Friendly name for the camera
   - `device_id`: Unique UUID for this camera
   - `device_code`: Short code identifier
   - `device_name`: Display name
   - `rtsp_url`: RTSP stream URL or video file path
   - `screen_resolution`: `[width, height]` array (e.g., `[1280, 720]`) or `"original"` to use the original stream resolution
   - `point_axis`: `"X"` or `"Y"` for detection axis
   - `line_offset`: Offset mode (optional)
   - `offset_amount`: Offset value (optional)
   - `lines`: Object with line definitions (e.g., `{"lineA": [[x1, y1], [x2, y2]], "lineB": [...]}`)
   - `mqtt_topic`: Optional per-camera MQTT topic (defaults to global `MQTT_TOPIC`)
   - `mqtt_interval_topic`: Optional per-camera interval topic
   - `yolo_model`: Optional per-camera model override (defaults to global `YOLO_MODEL`)

3. **Start the service**:
   ```bash
   # Direct Python
   python main.py
   
   # Docker Compose
   docker-compose up
   ```

### Adding More Cameras

**To add cameras without creating new containers:**

1. Edit `config/cameras.json` and add a new camera object to the array
2. Restart the service:
   ```bash
   docker-compose restart services-person-counter
   # or
   # Stop and start your Python process
   ```

The service automatically detects all cameras in the config file and starts processing them concurrently. Each camera runs in its own thread with isolated state (counters, tracking, MQTT client, database connection).

### Health Monitoring

Each camera logs health metrics every 60 seconds with the format:
```
[HEALTH] <camera-name> - Frames: <count>, FPS: <fps>, IN: <in_count>, OUT: <out_count>, Active tracks: <track_count>, Stream: <stream_url>
```

All logs are prefixed with the camera name for easy filtering and monitoring.

### Example: Adding a New Camera

To add a third camera (e.g., "parking-lot"), edit `config/cameras.json`:

```json
[
  {
    "name": "lobby-cam",
    "device_id": "958e0a40-05ed-414e-ac70-a54c2dcf60cf",
    ...
  },
  {
    "name": "warehouse-cam",
    "device_id": "c41a8de2-7a0a-4e8b-b7e2-1b5f6a2a1111",
    ...
  },
  {
    "name": "parking-lot-cam",
    "device_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "device_code": "parking-03",
    "device_name": "Parking Lot Entrance",
    "rtsp_url": "rtsp://username:password@192.168.1.12/stream",
    "screen_resolution": [1280, 720],
    "point_axis": "Y",
    "line_offset": "X",
    "offset_amount": 5,
    "lines": {
      "lineA": [[300, 220], [300, 560]],
      "lineB": [[305, 225], [305, 565]]
    },
    "mqtt_topic": "/person_in/parking",
    "mqtt_interval_topic": "/resampling_person/parking",
    "yolo_model": "yolo11n.pt"
  }
]
```

Then restart the service - the new camera will start processing automatically.

### Benefits of Multi-Camera Mode

- **Resource Efficiency**: Single YOLO model instance shared across all cameras (thread-safe inference)
- **Simplified Deployment**: One container/service manages all cameras
- **Easy Scaling**: Add cameras by editing JSON config, no container changes needed
- **Isolated State**: Each camera maintains its own counters, tracking, MQTT client, and database connection
- **Per-Camera Metrics**: Individual health monitoring for each camera stream
- **No Process Overhead**: Threading-based approach avoids subprocess spawning overhead

## Preview cameras (easy debug outside Docker)
This opens a window per camera, draws the lines from `config/cameras.json`, and waits for a key press.

```bash
python main.py --preview --config .\\config\\cameras.json
```

- Press any key to move to the next camera
- Press `q` to quit

## Interactive Line Point Editor (Debug Mode)
This mode allows you to interactively set line points by clicking on the video frame, similar to `--preview` but with click-to-set functionality.

```bash
python main.py --debug-lines --config .\\config\\cameras.json --camera-index 0
```

**Instructions:**
- **Left click**: Set line points (first point, then second point)
- **Right click**: Reset current line
- **Press 's'**: Save and update `cameras.json`
- **Press 'q'**: Quit without saving
- **Press 'n'**: Move to next line (lineA → lineB → lineC, etc.)

The editor shows:
- Current mouse position coordinates
- Preview of the line being drawn
- Existing lines from the config
- Visual feedback for points being set

This is useful for quickly setting up line coordinates without manually editing the JSON file.
