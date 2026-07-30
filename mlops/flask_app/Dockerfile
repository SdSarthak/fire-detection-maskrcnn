# Container image for the Mask R-CNN fire detection API.
# Build from the repository root so the segmentation package can be copied in:
#   docker build -t fire-detection -f mlops/flask_app/Dockerfile .

# ---------------------------------------------------------------- build stage
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY mlops/flask_app/requirements.txt .

# CPU-only torch wheels keep the image roughly 2 GB smaller than the CUDA build.
RUN pip install --user --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

# -------------------------------------------------------------- runtime stage
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/models/fire_detection_model.pt \
    SEGMENTATION_ROOT=/app/segmentation \
    SCORE_THRESHOLD=0.5 \
    DEVICE=cpu \
    PATH=/home/appuser/.local/bin:$PATH

# opencv-python-headless still needs libglib for its imgproc bindings.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 appuser \
    && mkdir -p /models \
    && chown -R appuser:appuser /models

WORKDIR /app

COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local
COPY --chown=appuser:appuser mlops/flask_app/ /app/
COPY --chown=appuser:appuser segmentation/src/ /app/segmentation/src/

USER appuser

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:5000/health || exit 1

CMD ["uwsgi", "--ini", "uwsgi.ini"]
