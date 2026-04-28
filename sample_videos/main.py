"""
RTSP Multi-Camera GPU Recorder
================================
Record banyak RTSP stream secara paralel dengan GPU acceleration (NVENC/QSV/AMF).
Semua konfigurasi via .env — tidak perlu edit script ini.

Usage:
    python main.py
    python main.py --duration 3600      # record 1 jam
    python main.py --list               # list kamera dari .env

Dependencies:
    pip install python-dotenv
    ffmpeg harus terinstall di sistem (https://ffmpeg.org/download.html)

GPU Encoder yang didukung:
    NVIDIA  → GPU_TYPE=nvidia  → h264_nvenc / hevc_nvenc
    Intel   → GPU_TYPE=intel   → h264_qsv   / hevc_qsv
    AMD     → GPU_TYPE=amd     → h264_amf   / hevc_amf
    CPU     → GPU_TYPE=cpu     → libx264    / libx265
"""

import os
import sys
import re
import time
import signal
import shutil
import argparse
import threading
import subprocess
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── ANSI Colors ─────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
WHITE  = "\033[97m"
DIM    = "\033[2m"

COLORS = ["\033[92m", "\033[93m", "\033[96m", "\033[95m",
          "\033[94m", "\033[91m", "\033[97m", "\033[33m"]


# ─── GPU Encoder Map ─────────────────────────────────────────────────────────
GPU_ENCODER = {
    # (gpu_type, codec) → (encoder_name, extra_flags)
    ("nvidia", "h264"): ("h264_nvenc",  ["-preset", "p4", "-rc", "vbr"]),
    ("nvidia", "h265"): ("hevc_nvenc",  ["-preset", "p4", "-rc", "vbr"]),
    ("intel",  "h264"): ("h264_qsv",   ["-preset", "medium"]),
    ("intel",  "h265"): ("hevc_qsv",   ["-preset", "medium"]),
    ("amd",    "h264"): ("h264_amf",   ["-quality", "balanced"]),
    ("amd",    "h265"): ("hevc_amf",   ["-quality", "balanced"]),
    ("cpu",    "h264"): ("libx264",    ["-preset", "fast", "-crf", "23"]),
    ("cpu",    "h265"): ("libx265",    ["-preset", "fast", "-crf", "28"]),
}

# Hardware decoder per GPU (untuk decode side juga pakai GPU)
HW_DECODER = {
    "nvidia": ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"],
    "intel":  ["-hwaccel", "qsv"],
    "amd":    ["-hwaccel", "dxva2"],
    "cpu":    [],
}

# Scale filter per GPU (resize di GPU jika bisa)
HW_SCALE = {
    "nvidia": "scale_cuda",
    "intel":  "vpp_qsv",
    "amd":    "scale",   # AMF tidak punya dedicated scale filter yang universal
    "cpu":    "scale",
}


# ─── Helpers ─────────────────────────────────────────────────────────────────
def env(key, default=""):
    return os.getenv(key, default).strip()

def env_int(key, default):
    try:
        return int(env(key, str(default)))
    except ValueError:
        return default

def env_bool(key, default="false"):
    return env(key, default).lower() == "true"

def fmt_duration(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def file_size(path):
    try:
        size = os.path.getsize(path)
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"
    except Exception:
        return "N/A"

def check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        print(f"{RED}[ERROR] ffmpeg tidak ditemukan di PATH.{RESET}")
        print(f"  Install: https://ffmpeg.org/download.html")
        sys.exit(1)


# ─── Baca Kamera dari .env ────────────────────────────────────────────────────
def load_cameras():
    """
    Scan environment untuk pola CAM_<N>_URL dan CAM_<N>_NAME.
    Kembalikan list dict per kamera.
    """
    cam_ids = set()
    for key in os.environ:
        m = re.match(r"^CAM_(\d+)_URL$", key)
        if m:
            cam_ids.add(int(m.group(1)))

    if not cam_ids:
        print(f"{RED}[ERROR] Tidak ada kamera ditemukan di .env.{RESET}")
        print(f"  Pastikan ada minimal CAM_1_URL dan CAM_1_NAME.")
        sys.exit(1)

    # Global defaults
    g_w         = env_int("RESOLUTION_WIDTH",  1920)
    g_h         = env_int("RESOLUTION_HEIGHT", 1080)
    g_fps       = float(env("FPS", "25"))
    g_bitrate   = env("BITRATE", "4000k")
    g_transport = env("RTSP_TRANSPORT", "tcp")
    g_template  = env("OUTPUT_FILENAME", "{name}_{datetime}")
    g_outdir    = Path(env("OUTPUT_DIR", "./recordings"))
    g_reconnect = env_bool("AUTO_RECONNECT", "true")
    g_delay     = env_int("RECONNECT_DELAY", 5)
    g_duration  = env_int("DURATION", 0)
    g_gpu       = env("GPU_TYPE", "nvidia").lower()
    g_codec     = env("CODEC", "h264").lower()

    now = datetime.now()
    cameras = []
    for n in sorted(cam_ids):
        url  = env(f"CAM_{n}_URL")
        name = env(f"CAM_{n}_NAME", f"cam{n}")
        if not url:
            print(f"{YELLOW}[WARN] CAM_{n}_URL kosong, dilewati.{RESET}")
            continue

        w   = env_int(f"CAM_{n}_RESOLUTION_WIDTH",  g_w)
        h   = env_int(f"CAM_{n}_RESOLUTION_HEIGHT", g_h)
        fps = float(env(f"CAM_{n}_FPS", str(g_fps)))
        bit = env(f"CAM_{n}_BITRATE", g_bitrate)

        filename = g_template.format(
            name=name,
            datetime=now.strftime("%Y%m%d_%H%M%S"),
            date=now.strftime("%Y%m%d"),
            time=now.strftime("%H%M%S"),
        )
        if not filename.endswith(".mp4"):
            filename += ".mp4"

        g_outdir.mkdir(parents=True, exist_ok=True)

        cameras.append({
            "id":          n,
            "name":        name,
            "url":         url,
            "width":       w,
            "height":      h,
            "fps":         fps,
            "bitrate":     bit,
            "transport":   g_transport,
            "output_path": str(g_outdir / filename),
            "reconnect":   g_reconnect,
            "delay":       g_delay,
            "duration":    g_duration,
            "gpu":         g_gpu,
            "codec":       g_codec,
        })

    return cameras


# ─── Build FFmpeg Command ─────────────────────────────────────────────────────
def build_ffmpeg_cmd(cam, duration_override=None):
    """
    Bangun perintah FFmpeg untuk satu kamera dengan GPU encode.
    Decode → (GPU scale jika perlu) → GPU encode → MP4 output.
    """
    gpu    = cam["gpu"]
    codec  = cam["codec"]
    w, h   = cam["width"], cam["height"]
    fps    = cam["fps"]
    bitrate= cam["bitrate"]
    dur    = duration_override if duration_override is not None else cam["duration"]

    encoder, enc_flags = GPU_ENCODER.get(
        (gpu, codec),
        GPU_ENCODER[("cpu", "h264")]   # fallback
    )
    hw_dec   = HW_DECODER.get(gpu, [])
    scale_fn = HW_SCALE.get(gpu, "scale")

    # Input flags
    input_flags = [
        "-rtsp_transport", cam["transport"],
        "-fflags",         "nobuffer",
        "-flags",          "low_delay",
        "-reorder_queue_size", "0",
        "-i",              cam["url"],
    ]

    # Video filter: scale + fps
    # Untuk NVIDIA scale_cuda dan hevc: -vf scale_cuda=W:H,fps=N
    vf = f"{scale_fn}={w}:{h},fps={fps}"

    # Duration flag
    dur_flags = ["-t", str(dur)] if dur > 0 else []

    cmd = (
        ["ffmpeg", "-y"]
        + hw_dec
        + dur_flags
        + input_flags
        + [
            "-vf",      vf,
            "-c:v",     encoder,
            "-b:v",     bitrate,
        ]
        + enc_flags
        + [
            "-an",              # no audio (bisa hapus kalau mau rekam audio)
            "-f",      "mp4",
            "-movflags", "+faststart",
            cam["output_path"],
        ]
    )

    return cmd


# ─── Recorder Thread ─────────────────────────────────────────────────────────
stop_event = threading.Event()
print_lock  = threading.Lock()

class CameraRecorder(threading.Thread):
    def __init__(self, cam, color, duration_override=None):
        super().__init__(daemon=True)
        self.cam              = cam
        self.color            = color
        self.duration_override= duration_override
        self.process          = None
        self.start_time       = None
        self.frame_count      = 0
        self.status           = "INIT"
        self.error            = None

    def log(self, level, msg):
        tag = {
            "INFO": f"{self.color}[{self.cam['name']}]{RESET}",
            "WARN": f"{YELLOW}[{self.cam['name']}]{RESET}",
            "ERR":  f"{RED}[{self.cam['name']}]{RESET}",
            "OK":   f"{GREEN}[{self.cam['name']}]{RESET}",
        }.get(level, f"[{self.cam['name']}]")
        with print_lock:
            print(f"  {tag} {msg}")

    def run(self):
        attempt = 0
        while not stop_event.is_set():
            attempt += 1
            if attempt > 1:
                self.status = "RECONNECT"
                self.log("WARN", f"Reconnect dalam {self.cam['delay']}s (percobaan ke-{attempt})...")
                for _ in range(self.cam["delay"]):
                    if stop_event.is_set():
                        return
                    time.sleep(1)
                if stop_event.is_set():
                    return

            cmd = build_ffmpeg_cmd(self.cam, self.duration_override)
            self.log("INFO", f"Menghubungkan → {self.cam['url']}")
            self.log("INFO", f"Output        → {self.cam['output_path']}")
            self.log("INFO", f"Encoder       → {GPU_ENCODER.get((self.cam['gpu'], self.cam['codec']), ('libx264',[]))[0]} | {self.cam['width']}x{self.cam['height']} @{self.cam['fps']}fps | {self.cam['bitrate']}")

            try:
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )
                self.start_time  = time.time()
                self.status      = "REC"
                self.frame_count = 0

                # Parse stderr FFmpeg untuk hitung frame
                for line in self.process.stderr:
                    if stop_event.is_set():
                        break
                    # Parse "frame=  123 fps= 25 ..."
                    m = re.search(r"frame=\s*(\d+)", line)
                    if m:
                        self.frame_count = int(m.group(1))

                self.process.wait()
                elapsed = time.time() - self.start_time
                ret     = self.process.returncode

                if ret == 0 or stop_event.is_set():
                    self.status = "DONE"
                    self.log("OK", (
                        f"Selesai | "
                        f"Durasi: {fmt_duration(elapsed)} | "
                        f"Frame: {self.frame_count:,} | "
                        f"Ukuran: {file_size(self.cam['output_path'])}"
                    ))
                    return
                else:
                    self.status = "ERROR"
                    self.log("WARN", f"FFmpeg keluar dengan kode {ret}.")

            except FileNotFoundError:
                self.log("ERR", "ffmpeg tidak ditemukan!")
                stop_event.set()
                return
            except Exception as e:
                self.log("ERR", f"Exception: {e}")

            if not self.cam["reconnect"]:
                self.log("ERR", "AUTO_RECONNECT=false. Berhenti.")
                self.status = "FAILED"
                return

    def stop(self):
        self.status = "STOPPING"
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()


# ─── Status Monitor ───────────────────────────────────────────────────────────
def status_monitor(recorders, global_start):
    """Print baris status semua kamera tiap detik."""
    while not stop_event.is_set():
        time.sleep(1)
        now     = time.time()
        elapsed = now - global_start
        lines   = []
        for r in recorders:
            cam_elapsed = (now - r.start_time) if r.start_time else 0
            icon = {
                "REC":       "🔴",
                "RECONNECT": "🟡",
                "DONE":      "✅",
                "ERROR":     "❌",
                "FAILED":    "💀",
                "STOPPING":  "⏹ ",
                "INIT":      "⏳",
            }.get(r.status, "❓")
            lines.append(
                f"    {r.color}{r.cam['name']:<15}{RESET} "
                f"{icon} {r.status:<10} "
                f"{fmt_duration(cam_elapsed)}  "
                f"frame:{r.frame_count:>6,}"
            )

        with print_lock:
            # Pergi ke awal baris status (ANSI move up)
            up = len(recorders) + 2
            print(f"\033[{up}A", end="")
            print(f"  {BOLD}{'─'*55}{RESET}")
            print(f"  {BOLD}Total elapsed: {fmt_duration(elapsed)}{RESET}  (Ctrl+C untuk stop)")
            print(f"  {'─'*55}")
            for l in lines:
                print(l + "\033[K")   # \033[K = clear to end of line


# ─── Signal Handler ───────────────────────────────────────────────────────────
def handle_signal(sig, frame):
    with print_lock:
        print(f"\n\n{YELLOW}[INFO] Sinyal stop diterima. Menghentikan semua kamera...{RESET}")
    stop_event.set()

signal.signal(signal.SIGINT,  handle_signal)
signal.signal(signal.SIGTERM, handle_signal)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="RTSP Multi-Camera GPU Recorder",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--duration", type=int, default=None,
                        help="Override DURATION dari .env (detik, 0=tanpa batas)")
    parser.add_argument("--list", action="store_true",
                        help="Tampilkan daftar kamera dari .env lalu keluar")
    args = parser.parse_args()

    check_ffmpeg()
    cameras = load_cameras()

    # ── List mode ────────────────────────────────────────────────
    if args.list:
        print(f"\n{BOLD}Kamera yang terdaftar di .env:{RESET}")
        for c in cameras:
            encoder = GPU_ENCODER.get((c["gpu"], c["codec"]), ("libx264",[]))[0]
            print(f"  [{c['id']}] {CYAN}{c['name']:<15}{RESET} "
                  f"{c['url']}")
            print(f"       {c['width']}x{c['height']} @{c['fps']}fps | "
                  f"{c['bitrate']} | encoder: {encoder}")
        print()
        return

    # ── Print header ─────────────────────────────────────────────
    gpu_type = cameras[0]["gpu"]
    codec    = cameras[0]["codec"]
    encoder  = GPU_ENCODER.get((gpu_type, codec), ("libx264",[]))[0]
    duration = args.duration if args.duration is not None else cameras[0]["duration"]

    print(f"\n{'═'*60}")
    print(f"  {BOLD}RTSP Multi-Camera GPU Recorder{RESET}")
    print(f"{'═'*60}")
    print(f"  GPU Type  : {CYAN}{gpu_type.upper()}{RESET}")
    print(f"  Encoder   : {CYAN}{encoder}{RESET}")
    print(f"  Kamera    : {len(cameras)} stream")
    print(f"  Durasi    : {'Tanpa batas (Ctrl+C untuk stop)' if duration == 0 else fmt_duration(duration)}")
    print(f"{'═'*60}\n")

    for i, c in enumerate(cameras):
        color = COLORS[i % len(COLORS)]
        enc   = GPU_ENCODER.get((c["gpu"], c["codec"]), ("libx264",[]))[0]
        print(f"  {color}[{c['name']}]{RESET}")
        print(f"    URL    : {DIM}{c['url']}{RESET}")
        print(f"    Output : {c['output_path']}")
        print(f"    Res    : {c['width']}x{c['height']} @{c['fps']}fps | {c['bitrate']} | {enc}")

    print(f"\n{'─'*60}")

    # ── Start recorders ──────────────────────────────────────────
    global_start = time.time()
    recorders = []
    for i, cam in enumerate(cameras):
        color = COLORS[i % len(COLORS)]
        r = CameraRecorder(cam, color, duration_override=args.duration)
        r.start()
        recorders.append(r)

    # Placeholder baris untuk status monitor
    print()
    print(f"  {'─'*55}")
    print(f"  Total elapsed: 00:00:00")
    print(f"  {'─'*55}")
    for r in recorders:
        print(f"    {r.cam['name']}")

    # Start status thread
    monitor = threading.Thread(target=status_monitor, args=(recorders, global_start), daemon=True)
    monitor.start()

    # Tunggu semua selesai atau stop_event
    try:
        while not stop_event.is_set():
            all_done = all(r.status in ("DONE", "FAILED") for r in recorders)
            if all_done:
                stop_event.set()
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        stop_event.set()

    # Stop semua
    for r in recorders:
        r.stop()
    for r in recorders:
        r.join(timeout=15)

    # Summary
    total = time.time() - global_start
    print(f"\n\n{'═'*60}")
    print(f"  {BOLD}Rekaman selesai — Ringkasan{RESET}")
    print(f"{'═'*60}")
    for r in recorders:
        icon  = "✅" if r.status == "DONE" else "❌"
        elapsed = (time.time() - r.start_time) if r.start_time else 0
        print(f"  {icon} {r.color}{r.cam['name']:<15}{RESET} "
              f"| {fmt_duration(elapsed)} "
              f"| frame: {r.frame_count:,} "
              f"| {file_size(r.cam['output_path'])}")
        print(f"      {DIM}{r.cam['output_path']}{RESET}")
    print(f"{'─'*60}")
    print(f"  Total waktu : {fmt_duration(total)}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
