# Multi-stage Dockerfile for Mask R-CNN Fire Detection API

# Stage 1: Builder
FROM python:3.9-slim as builder

WORKDIR /build

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libopencv-dev \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/models/fire_detection_model.h5 \
    FLASK_APP=app.py

# Create app user
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libopencv-core4.2 \
    libopencv-imgproc4.2 \
    libopencv-highgui4.2 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy application files
COPY --chown=appuser:appuser flask_app/ .

# Set PATH
ENV PATH=/home/appuser/.local/bin:$PATH

# Create models directory
RUN mkdir -p /models && chown -R appuser:appuser /models

# Switch to app user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/health')" || exit 1

# Expose port
EXPOSE 5000

# Run application
CMD ["uwsgi", "--ini", "uwsgi.ini"]
