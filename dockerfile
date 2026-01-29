FROM nvidia/cuda:12.5.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN sed -i 's|http://archive.ubuntu.com/ubuntu|http://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu|http://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 \
        python3-pip \
        python3-dev \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        ffmpeg \
        curl \
        git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

RUN ln -sf /usr/bin/python3.10 /usr/bin/python

RUN pip install --upgrade pip setuptools wheel

RUN pip install \
    torch \
    torchvision \
    --index-url https://download.pytorch.org/whl/cu130

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY main.py yolo11m.pt ./

EXPOSE 8080

CMD ["python", "main.py"]

