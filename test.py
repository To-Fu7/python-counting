"""
NVDEC Test Script
Tests if OpenCV's FFmpeg backend can use NVIDIA hardware decoding (h264_cuvid).

Usage:
  python test_nvdec.py                    # uses default test (webcam or sample)
  python test_nvdec.py --source 1.mp4     # test with local file
  python test_nvdec.py --source rtsp://...  # test with RTSP stream
"""

import os
import sys
import time
import argparse
import subprocess
import cv2


def check_ffmpeg_cuvid():
    """Check if system FFmpeg has cuvid support"""
    print("=" * 60)
    print("[1/4] Checking FFmpeg CUVID support...")
    print("=" * 60)
    
    try:
        result = subprocess.run(
            ["ffmpeg", "-decoders"],
            capture_output=True, text=True, timeout=10
        )
        output = result.stdout + result.stderr
        
        cuvid_decoders = [line.strip() for line in output.split('\n') if 'cuvid' in line]
        
        if cuvid_decoders:
            print("  [OK] FFmpeg CUVID decoders found:")
            for d in cuvid_decoders:
                print(f"       {d}")
            return True
        else:
            print("  [FAIL] No CUVID decoders found in FFmpeg!")
            print("         Your FFmpeg was not built with --enable-cuvid")
            return False
            
    except FileNotFoundError:
        print("  [WARN] ffmpeg command not found, skipping system check")
        return None
    except Exception as e:
        print(f"  [WARN] Could not check FFmpeg: {e}")
        return None


def check_opencv_ffmpeg():
    """Check if OpenCV has FFmpeg backend enabled"""
    print()
    print("=" * 60)
    print("[2/4] Checking OpenCV FFmpeg support...")
    print("=" * 60)
    
    print(f"  OpenCV version: {cv2.__version__}")
    
    build_info = cv2.getBuildInformation()
    
    # Check FFmpeg
    ffmpeg_ok = False
    for line in build_info.split('\n'):
        if 'FFMPEG' in line and 'YES' in line:
            ffmpeg_ok = True
            print(f"  [OK] {line.strip()}")
        elif 'FFMPEG' in line:
            print(f"  [INFO] {line.strip()}")
    
    # Check CUDA
    for line in build_info.split('\n'):
        if 'NVIDIA CUDA' in line:
            print(f"  [INFO] {line.strip()}")
        elif 'NVCUVID' in line or 'nvcuvid' in line:
            print(f"  [INFO] {line.strip()}")

    if not ffmpeg_ok:
        print("  [FAIL] OpenCV was NOT built with FFmpeg support!")
        print("         pip-installed opencv-python may bundle FFmpeg without CUDA.")
        print("         You need to build OpenCV from source against CUDA-enabled FFmpeg.")
    
    return ffmpeg_ok


def check_nvidia_gpu():
    """Check if NVIDIA GPU is accessible"""
    print()
    print("=" * 60)
    print("[3/4] Checking NVIDIA GPU...")
    print("=" * 60)
    
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                print(f"  [OK] GPU: {line.strip()}")
            return True
        else:
            print("  [FAIL] nvidia-smi failed")
            return False
    except FileNotFoundError:
        print("  [FAIL] nvidia-smi not found — no NVIDIA driver?")
        return False


def test_decode(source, use_nvdec=True):
    """Test video decoding with or without NVDEC"""
    mode = "NVDEC (GPU)" if use_nvdec else "CPU"
    
    # Set env BEFORE opening VideoCapture
    if use_nvdec:
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'hwaccel;cuda|video_codec;h264_cuvid|rtsp_transport;tcp'
    else:
        os.environ.pop('OPENCV_FFMPEG_CAPTURE_OPTIONS', None)
    
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    
    if not cap.isOpened():
        print(f"  [FAIL] Could not open source with {mode}")
        return None
    
    # Read and time N frames
    num_frames = 200
    times = []
    success_count = 0
    
    for i in range(num_frames):
        t0 = time.perf_counter()
        ret, frame = cap.read()
        t1 = time.perf_counter()
        
        if not ret:
            break
        
        success_count += 1
        times.append(t1 - t0)
    
    cap.release()
    
    if success_count == 0:
        print(f"  [FAIL] No frames read with {mode}")
        return None
    
    avg_ms = (sum(times) / len(times)) * 1000
    fps = 1000 / avg_ms if avg_ms > 0 else 0
    total_s = sum(times)
    
    return {
        "mode": mode,
        "frames": success_count,
        "avg_ms": avg_ms,
        "fps": fps,
        "total_s": total_s
    }


def run_benchmark(source):
    """Run decode benchmark comparing CPU vs NVDEC"""
    print()
    print("=" * 60)
    print(f"[4/4] Benchmarking decode: {source}")
    print("=" * 60)
    
    # Test CPU decode
    print(f"\n  Testing CPU decode (200 frames)...")
    cpu_result = test_decode(source, use_nvdec=False)
    if cpu_result:
        print(f"  [CPU]   {cpu_result['frames']} frames | {cpu_result['avg_ms']:.1f} ms/frame | {cpu_result['fps']:.1f} FPS | {cpu_result['total_s']:.2f}s total")
    
    # Test NVDEC decode
    print(f"\n  Testing NVDEC decode (200 frames)...")
    nvdec_result = test_decode(source, use_nvdec=True)
    if nvdec_result:
        print(f"  [NVDEC] {nvdec_result['frames']} frames | {nvdec_result['avg_ms']:.1f} ms/frame | {nvdec_result['fps']:.1f} FPS | {nvdec_result['total_s']:.2f}s total")
    
    # Compare
    print()
    print("-" * 60)
    if cpu_result and nvdec_result:
        speedup = cpu_result['avg_ms'] / nvdec_result['avg_ms'] if nvdec_result['avg_ms'] > 0 else 0
        
        if speedup > 1.2:
            print(f"  NVDEC is {speedup:.1f}x FASTER than CPU decode")
            print(f"  NVDEC is WORKING!")
        elif speedup > 0.8:
            print(f"  NVDEC and CPU are similar speed ({speedup:.1f}x)")
            print(f"  NVDEC might not be active — FFmpeg may be falling back to CPU.")
            print(f"  Check: ffmpeg -decoders | grep cuvid")
        else:
            print(f"  CPU was faster ({1/speedup:.1f}x) — NVDEC likely not working")
    elif nvdec_result and not cpu_result:
        print("  NVDEC works, CPU failed (unusual)")
    elif cpu_result and not nvdec_result:
        print("  CPU works, NVDEC FAILED")
        print("  NVDEC is NOT working. Likely causes:")
        print("    1. FFmpeg not built with --enable-cuvid")
        print("    2. Missing NVIDIA_DRIVER_CAPABILITIES=video in Docker")
        print("    3. nv-codec-headers version mismatch with driver")
    else:
        print("  Both methods failed — check your video source")
    
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="Test NVDEC hardware video decoding with OpenCV")
    parser.add_argument("--source", type=str, default=None,
                        help="Video source: file path, RTSP URL, or camera index (default: auto)")
    args = parser.parse_args()
    
    print()
    print("###########################################################")
    print("#           NVDEC / h264_cuvid Test Script                #")
    print("###########################################################")
    print()
    
    # Run checks
    cuvid_ok = check_ffmpeg_cuvid()
    ffmpeg_ok = check_opencv_ffmpeg()
    gpu_ok = check_nvidia_gpu()
    
    # Determine source
    source = args.source
    if source is None:
        # Try to find a test video
        for test_file in ['1.mp4', 'test.mp4', 'sample.mp4']:
            if os.path.exists(test_file):
                source = test_file
                break
        
        if source is None:
            print()
            print("[WARN] No video source specified and no test file found.")
            print("       Use: python test_nvdec.py --source <video.mp4 or rtsp://...>")
            print()
            
            # Summary without benchmark
            print("=" * 60)
            print("SUMMARY (no benchmark — no video source)")
            print("=" * 60)
            print(f"  FFmpeg CUVID : {'OK' if cuvid_ok else 'MISSING' if cuvid_ok is not None else 'UNKNOWN'}")
            print(f"  OpenCV FFmpeg: {'OK' if ffmpeg_ok else 'MISSING'}")
            print(f"  NVIDIA GPU   : {'OK' if gpu_ok else 'MISSING'}")
            sys.exit(0)
    
    # Run benchmark
    run_benchmark(source)
    
    # Final summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  FFmpeg CUVID : {'OK' if cuvid_ok else 'MISSING' if cuvid_ok is not None else 'UNKNOWN'}")
    print(f"  OpenCV FFmpeg: {'OK' if ffmpeg_ok else 'MISSING'}")
    print(f"  NVIDIA GPU   : {'OK' if gpu_ok else 'MISSING'}")
    
    if not ffmpeg_ok:
        print()
        print("  To fix: Build OpenCV from source against CUDA-enabled FFmpeg")
        print("  See dockerfile.nvdec for a working Docker build")


if __name__ == "__main__":
    main()