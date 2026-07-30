"""
Flask application serving the Mask R-CNN fire detection model.

Endpoints
    GET  /health        liveness - always 200
    GET  /ready         readiness - 503 until the model is loaded
    GET  /model_info    model metadata
    GET  /metrics       Prometheus exposition format
    POST /predict       single image  (form field: file)
    POST /batch_predict several images (form field: files)
"""
from __future__ import annotations

import base64
import os
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from flask import Flask, jsonify, request

# --------------------------------------------------------------------------- #
# Make the segmentation package importable both in-repo and inside the image
# (the Dockerfile copies segmentation/ next to this file).
# --------------------------------------------------------------------------- #
def _segmentation_root() -> Optional[Path]:
    here = Path(__file__).resolve().parent
    candidates = []
    configured = os.environ.get('SEGMENTATION_ROOT')
    if configured:
        candidates.append(Path(configured))
    candidates += [here / 'segmentation', here.parents[1] / 'segmentation']
    for candidate in candidates:
        if (candidate / 'src' / 'predictor.py').exists():
            return candidate
    return None


_SEGMENTATION_ROOT = _segmentation_root()
if _SEGMENTATION_ROOT and str(_SEGMENTATION_ROOT) not in sys.path:
    sys.path.insert(0, str(_SEGMENTATION_ROOT))

from src.predictor import FirePredictor  # noqa: E402  (path set up above)

# --------------------------------------------------------------------------- #
# Configuration (all overridable through the environment)
# --------------------------------------------------------------------------- #
MODEL_PATH = os.environ.get('MODEL_PATH', '/models/fire_detection_model.pt')
DEVICE = os.environ.get('DEVICE', 'auto')
SCORE_THRESHOLD = float(os.environ.get('SCORE_THRESHOLD', '0.5'))
MASK_THRESHOLD = float(os.environ.get('MASK_THRESHOLD', '0.5'))
MAX_CONTENT_LENGTH_MB = float(os.environ.get('MAX_CONTENT_LENGTH_MB', '16'))
MAX_BATCH_SIZE = int(os.environ.get('MAX_BATCH_SIZE', '16'))
MAX_CONTENT_LENGTH = int(MAX_CONTENT_LENGTH_MB * 1024 * 1024)

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'tif', 'tiff', 'bmp'}
MODEL_VERSION = os.environ.get('MODEL_VERSION', 'unknown')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
_predictor: Optional[FirePredictor] = None
_load_error: Optional[str] = None
_load_lock = threading.Lock()

# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
_metrics_lock = threading.Lock()
_metrics = {
    'requests_total': 0,
    'errors_total': 0,
    'images_processed_total': 0,
    'fire_detections_total': 0,
    'inference_seconds_total': 0.0,
}
_STARTED_AT = time.time()


def _record(**increments) -> None:
    with _metrics_lock:
        for key, value in increments.items():
            _metrics[key] = _metrics.get(key, 0) + value


def get_predictor() -> Optional[FirePredictor]:
    """Return the loaded predictor, loading it on first use.

    Returns ``None`` when the checkpoint is missing or unreadable; the error is
    kept in ``_load_error`` and surfaced by /health and /ready.
    """
    global _predictor, _load_error
    if _predictor is not None:
        return _predictor

    with _load_lock:
        if _predictor is not None:
            return _predictor
        try:
            app.logger.info('Loading model from %s', MODEL_PATH)
            _predictor = FirePredictor.from_checkpoint(
                MODEL_PATH, device=DEVICE,
                score_threshold=SCORE_THRESHOLD,
                mask_threshold=MASK_THRESHOLD)
            _load_error = None
            app.logger.info('Model loaded (classes: %s)', _predictor.class_names)
        except Exception as exc:
            _load_error = f'{type(exc).__name__}: {exc}'
            app.logger.error('Failed to load model from %s: %s', MODEL_PATH, _load_error)
        return _predictor


def set_predictor(predictor: Optional[FirePredictor]) -> None:
    """Inject a predictor directly. Used by tests and by warm-up code."""
    global _predictor, _load_error
    _predictor = predictor
    _load_error = None if predictor is not None else _load_error


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def allowed_file(filename: str) -> bool:
    """Check the upload carries a supported image extension."""
    return '.' in (filename or '') and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def decode_image(raw: bytes) -> np.ndarray:
    """Decode uploaded bytes into an RGB uint8 array without touching disk."""
    if not raw:
        raise ValueError('Empty file')
    buffer = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Could not decode image; the file may be corrupt')
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def encode_overlay(predictor: FirePredictor, image: np.ndarray, result) -> Optional[str]:
    """Render the mask overlay and return it as a base64 PNG data string."""
    overlay = predictor.render_overlay(image, result)
    success, encoded = cv2.imencode('.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    if not success:
        return None
    return base64.b64encode(encoded.tobytes()).decode('ascii')


def wants_overlay() -> bool:
    """Whether the caller asked for the annotated image in the response."""
    flag = request.args.get('overlay', request.form.get('overlay', 'false'))
    return str(flag).strip().lower() in {'1', 'true', 'yes', 'on'}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_single(predictor: FirePredictor, raw: bytes, filename: str,
                overlay: bool) -> dict:
    """Decode, predict and shape the JSON payload for one uploaded image."""
    image = decode_image(raw)
    started = time.time()
    result = predictor.predict(image)
    elapsed = time.time() - started

    payload = FirePredictor.serializable(result)
    payload['filename'] = filename
    payload['inference_ms'] = round(elapsed * 1000, 2)
    payload['status'] = 'success'
    if overlay:
        payload['overlay_png_base64'] = encode_overlay(predictor, image, result)

    _record(images_processed_total=1,
            inference_seconds_total=elapsed,
            fire_detections_total=result['num_detections'])
    return payload


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.route('/health', methods=['GET'])
def health():
    """Liveness probe. Always 200 while the process can serve requests."""
    _record(requests_total=1)
    return jsonify({
        'status': 'healthy',
        'model_loaded': _predictor is not None,
        'model_path': MODEL_PATH,
        'uptime_seconds': round(time.time() - _STARTED_AT, 1),
        'timestamp': _now(),
    }), 200


@app.route('/ready', methods=['GET'])
def ready():
    """Readiness probe. 503 until the model is actually usable."""
    _record(requests_total=1)
    predictor = get_predictor()
    if predictor is None:
        return jsonify({
            'status': 'not_ready',
            'error': _load_error or f'Model not found at {MODEL_PATH}',
            'timestamp': _now(),
        }), 503
    return jsonify({'status': 'ready', 'timestamp': _now()}), 200


@app.route('/model_info', methods=['GET'])
def model_info():
    """Describe the loaded model."""
    _record(requests_total=1)
    predictor = get_predictor()
    if predictor is None:
        _record(errors_total=1)
        return jsonify({'error': _load_error or 'Model not loaded',
                        'status': 'error'}), 503

    config = predictor.config
    return jsonify({
        'model_name': 'Mask R-CNN Fire Detection',
        'architecture': f'maskrcnn_{config.BACKBONE}_fpn',
        'version': MODEL_VERSION,
        'framework': 'pytorch',
        'classes': predictor.class_names,
        'score_threshold': predictor.score_threshold,
        'mask_threshold': predictor.mask_threshold,
        'input_size_range': [config.IMAGE_MIN_DIM, config.IMAGE_MAX_DIM],
        'max_detections': config.DETECTION_MAX_INSTANCES,
        'supported_formats': sorted(ALLOWED_EXTENSIONS),
        'max_file_size_mb': MAX_CONTENT_LENGTH_MB,
        'max_batch_size': MAX_BATCH_SIZE,
        'status': 'loaded',
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """Segment fire regions in a single uploaded image."""
    _record(requests_total=1)
    predictor = get_predictor()
    if predictor is None:
        _record(errors_total=1)
        return jsonify({'error': _load_error or 'Model not loaded',
                        'status': 'error'}), 503

    if 'file' not in request.files:
        _record(errors_total=1)
        return jsonify({'error': "No file part in request (expected form field 'file')",
                        'status': 'error'}), 400

    uploaded = request.files['file']
    if not uploaded.filename:
        _record(errors_total=1)
        return jsonify({'error': 'No file selected', 'status': 'error'}), 400

    if not allowed_file(uploaded.filename):
        _record(errors_total=1)
        return jsonify({
            'error': f'File type not allowed. Allowed types: '
                     f'{", ".join(sorted(ALLOWED_EXTENSIONS))}',
            'status': 'error',
        }), 400

    try:
        payload = _run_single(predictor, uploaded.read(),
                              uploaded.filename, wants_overlay())
        return jsonify(payload), 200
    except ValueError as exc:
        _record(errors_total=1)
        return jsonify({'error': str(exc), 'status': 'error'}), 400
    except Exception as exc:
        _record(errors_total=1)
        app.logger.error('Prediction failed: %s', exc)
        traceback.print_exc()
        return jsonify({'error': 'Prediction failed', 'detail': str(exc),
                        'status': 'error'}), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Segment fire regions in several uploaded images."""
    _record(requests_total=1)
    predictor = get_predictor()
    if predictor is None:
        _record(errors_total=1)
        return jsonify({'error': _load_error or 'Model not loaded',
                        'status': 'error'}), 503

    uploads = request.files.getlist('files')
    if not uploads:
        _record(errors_total=1)
        return jsonify({'error': "No files in request (expected form field 'files')",
                        'status': 'error'}), 400

    if len(uploads) > MAX_BATCH_SIZE:
        _record(errors_total=1)
        return jsonify({
            'error': f'Too many files: {len(uploads)} (maximum {MAX_BATCH_SIZE})',
            'status': 'error',
        }), 400

    overlay = wants_overlay()
    results: List[dict] = []
    succeeded = 0

    for uploaded in uploads:
        filename = uploaded.filename or 'unnamed'
        if not allowed_file(filename):
            results.append({'filename': filename,
                            'error': 'File type not allowed', 'status': 'error'})
            continue
        try:
            results.append(_run_single(predictor, uploaded.read(), filename, overlay))
            succeeded += 1
        except Exception as exc:
            app.logger.warning('Batch item %s failed: %s', filename, exc)
            results.append({'filename': filename, 'error': str(exc), 'status': 'error'})

    failed = len(results) - succeeded
    if failed:
        _record(errors_total=failed)

    return jsonify({
        'results': results,
        'total': len(results),
        'succeeded': succeeded,
        'failed': failed,
        'fire_detected': sum(1 for r in results if r.get('is_fire')),
        'timestamp': _now(),
    }), 200


@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus exposition endpoint scraped by the GKE deployment."""
    with _metrics_lock:
        snapshot = dict(_metrics)

    lines = [
        '# HELP fire_detection_requests_total Total HTTP requests handled.',
        '# TYPE fire_detection_requests_total counter',
        f'fire_detection_requests_total {snapshot["requests_total"]}',
        '# HELP fire_detection_errors_total Total failed requests.',
        '# TYPE fire_detection_errors_total counter',
        f'fire_detection_errors_total {snapshot["errors_total"]}',
        '# HELP fire_detection_images_processed_total Images run through the model.',
        '# TYPE fire_detection_images_processed_total counter',
        f'fire_detection_images_processed_total {snapshot["images_processed_total"]}',
        '# HELP fire_detection_instances_total Fire instances detected.',
        '# TYPE fire_detection_instances_total counter',
        f'fire_detection_instances_total {snapshot["fire_detections_total"]}',
        '# HELP fire_detection_inference_seconds_total Cumulative inference time.',
        '# TYPE fire_detection_inference_seconds_total counter',
        f'fire_detection_inference_seconds_total {snapshot["inference_seconds_total"]:.6f}',
        '# HELP fire_detection_model_loaded Whether the model is loaded (1) or not (0).',
        '# TYPE fire_detection_model_loaded gauge',
        f'fire_detection_model_loaded {1 if _predictor is not None else 0}',
        '# HELP fire_detection_uptime_seconds Process uptime.',
        '# TYPE fire_detection_uptime_seconds gauge',
        f'fire_detection_uptime_seconds {time.time() - _STARTED_AT:.1f}',
        '',
    ]
    return '\n'.join(lines), 200, {'Content-Type': 'text/plain; version=0.0.4'}


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found', 'status': 'error'}), 404


@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'error': 'Method not allowed', 'status': 'error'}), 405


@app.errorhandler(413)
def request_entity_too_large(error):
    _record(errors_total=1)
    return jsonify({
        'error': f'File too large. Maximum size: {MAX_CONTENT_LENGTH_MB:.0f}MB',
        'status': 'error',
    }), 413


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


# Warm the model at import time so uWSGI/gunicorn workers are ready to serve.
get_predictor()


if __name__ == '__main__':
    print('=' * 60)
    print('Mask R-CNN fire detection API')
    print('=' * 60)
    print(f'Model path: {MODEL_PATH}')
    print(f'Model loaded: {_predictor is not None}')
    if _predictor is None:
        print(f'  {_load_error or "checkpoint missing"}')
        print('  The API will start anyway; /ready stays 503 until a model loads.')
    print('Endpoints: /health /ready /model_info /metrics /predict /batch_predict')
    print('=' * 60)
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', '5000')), debug=False)
