# Clarifai Auto-Collect Dockerfile
# Multi-stage build for optimized image size

# Stage 1: Base image with Python and system dependencies
FROM python:3.10-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Build stage for installing Python packages
FROM base AS builder

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Stage 3: Final runtime image
FROM base AS runtime

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create app directory
WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/models \
    /app/face_dataset/images/train \
    /app/face_dataset/images/val \
    /app/face_dataset/labels/train \
    /app/face_dataset/labels/val \
    /app/face_dataset/raw_images \
    /app/face_dataset/windshield_crops \
    /app/face_dataset/visualizations

# Copy application files
COPY main.py .
COPY windshield_detector.py .
COPY clarifai_client.py .
COPY dataset_manager.py .
COPY utils.py .
COPY config.yaml .

# Copy models directory into the image
# Place your YOLO models in models/ directory before building
# Models will be baked into the image
COPY models /app/models/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run the application
CMD ["python", "main.py"]
