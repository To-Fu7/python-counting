# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Build FFmpeg (with NVDEC/CUVID) + OpenCV from source
# ─────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN sed -i 's|http://archive.ubuntu.com/ubuntu|http://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu|http://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        curl \
        nasm \
        yasm \
        pkg-config \
        python3.10 \
        python3-dev \
        python3-pip \
        python3-numpy \
        libssl-dev \
        libx264-dev \
        libx265-dev \
        libvpx-dev \
        libopus-dev \
        libmp3lame-dev \
        libglib2.0-dev \
        libsm6 \
        libxext6 \
        libxrender-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install nv-codec-headers (required for FFmpeg NVDEC/CUVID support)
RUN git clone --depth 1 --branch n12.2.72.0 \
        https://github.com/FFmpeg/nv-codec-headers.git /tmp/nv-codec-headers && \
    cd /tmp/nv-codec-headers && \
    make install && \
    rm -rf /tmp/nv-codec-headers

# Build FFmpeg 7.1 with CUDA/NVDEC support
RUN git clone --depth 1 --branch n7.1 \
        https://github.com/FFmpeg/FFmpeg.git /tmp/ffmpeg && \
    cd /tmp/ffmpeg && \
    ./configure \
        --prefix=/opt/ffmpeg \
        --enable-shared \
        --disable-static \
        --enable-gpl \
        --enable-nonfree \
        --enable-cuda \
        --enable-cuvid \
        --enable-nvdec \
        --enable-nvenc \
        --enable-libnpp \
        --enable-libx264 \
        --enable-libx265 \
        --enable-libvpx \
        --enable-libopus \
        --enable-libmp3lame \
        --extra-cflags='-I/usr/local/cuda/include' \
        --extra-ldflags='-L/usr/local/cuda/lib64' && \
    make -j$(nproc) && \
    make install && \
    rm -rf /tmp/ffmpeg

ENV PKG_CONFIG_PATH=/opt/ffmpeg/lib/pkgconfig:$PKG_CONFIG_PATH
ENV LD_LIBRARY_PATH=/opt/ffmpeg/lib:$LD_LIBRARY_PATH

# Build OpenCV 4.10 linked against the CUDA-enabled FFmpeg
RUN git clone --depth 1 --branch 4.10.0 \
        https://github.com/opencv/opencv.git /tmp/opencv && \
    mkdir /tmp/opencv/build && \
    cd /tmp/opencv/build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/opencv \
        -DBUILD_SHARED_LIBS=ON \
        -DWITH_FFMPEG=ON \
        -DWITH_CUDA=OFF \
        -DWITH_GTK=OFF \
        -DWITH_QT=OFF \
        -DBUILD_TESTS=OFF \
        -DBUILD_PERF_TESTS=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_DOCS=OFF \
        -DFFMPEG_DIR=/opt/ffmpeg \
        -DPYTHON3_EXECUTABLE=$(which python3.10) \
        -DOPENCV_PYTHON3_INSTALL_PATH=/opt/opencv/python && \
    make -j$(nproc) && \
    make install && \
    rm -rf /tmp/opencv


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Runtime image
# ─────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN sed -i 's|http://archive.ubuntu.com/ubuntu|http://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu|http://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 \
        python3-pip \
        python3-numpy \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libx264-163 \
        libx265-199 \
        libvpx7 \
        libopus0 \
        libmp3lame0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

# Copy FFmpeg and OpenCV builds from builder
COPY --from=builder /opt/ffmpeg /opt/ffmpeg
COPY --from=builder /opt/opencv /opt/opencv

ENV LD_LIBRARY_PATH=/opt/ffmpeg/lib:/opt/opencv/lib:$LD_LIBRARY_PATH
ENV PATH=/opt/ffmpeg/bin:$PATH
ENV PYTHONPATH=/opt/opencv/python:$PYTHONPATH

RUN pip install --upgrade pip setuptools wheel

RUN pip install \
    torch \
    torchvision \
    --index-url https://download.pytorch.org/whl/cu130

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt && \
    pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python 2>/dev/null || true

COPY main.py yolo11m.pt yolo11n.pt ./

EXPOSE 8080

CMD ["python", "main.py"]
