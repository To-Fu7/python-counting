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
- Populate your database/MQTT settings in `.env` once.
- List cameras in `config/cameras.json` (one object per camera with `device_id`, `rtsp_url`, `screen_resolution`, `lines`, and optional MQTT topics).
- Set `MULTI_CAMERA_CONFIG=./config/cameras.json` in `.env` (or via docker-compose) to enable multi-camera mode.
- Start the service as usual (`python main.py` or `docker-compose up`); the app will spawn one worker process per camera entry.
