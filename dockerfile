# Base image with Python (no CUDA)
FROM python:3.10-slim-bullseye

# Set environment variables for non-interactive installs and pip optimizations
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Update and install required system packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        ffmpeg \
        curl \
        git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python build tools
RUN pip install --upgrade pip setuptools wheel

# Set workdir
WORKDIR /app

# Copy requirements first to leverage caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy app files
COPY main.py yolo11n.pt ./

# Expose application port
EXPOSE 8080

# Run the app
CMD ["python", "main.py"]
