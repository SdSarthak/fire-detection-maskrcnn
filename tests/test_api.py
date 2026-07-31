"""Tests for the Flask serving layer, with a stubbed predictor."""
from __future__ import annotations

import io

import cv2
import numpy as np
import pytest

import app as flask_app
from conftest import make_fire_image
from src.config import FireDetectionConfig


class StubPredictor:
    """Stands in for FirePredictor so the API can be tested without weights."""

    def __init__(self, detections=1):
        self.config = FireDetectionConfig()
        self.class_names = list(self.config.CLASS_NAMES)
        self.score_threshold = 0.5
        self.mask_threshold = 0.5
        self.detections = detections
        self.calls = []

    def predict(self, image):
        self.calls.append(image)
        height, width = image.shape[:2]
        masks = np.zeros((self.detections, height, width), dtype=bool)
        detections = []
        for index in range(self.detections):
            masks[index, 2:8, 2:8] = True
            detections.append({
                'class_id': 1,
                'class_name': 'fire',
                'score': 0.91,
                'bounding_box': {'x1': 2.0, 'y1': 2.0, 'x2': 8.0, 'y2': 8.0},
                'mask_area_px': 36,
                'mask_area_ratio': 0.1,
                'polygons': [[[2, 2], [8, 2], [8, 8], [2, 8]]],
            })
        return {
            'detections': detections,
            'num_detections': self.detections,
            'is_fire': self.detections > 0,
            'confidence': 0.91 if self.detections else 0.0,
            'fire_area_ratio': 0.1,
            'fire_area_px': 36,
            'image_size': {'height': int(height), 'width': int(width)},
            'score_threshold': self.score_threshold,
            'masks': masks,
        }

    def render_overlay(self, image, result):
        return image.copy()


def png_bytes(image=None) -> bytes:
    image = make_fire_image() if image is None else image
    ok, encoded = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    assert ok
    return encoded.tobytes()


def upload(name='fire.png', data=None):
    return (io.BytesIO(data if data is not None else png_bytes()), name)


@pytest.fixture
def client():
    flask_app.app.config['TESTING'] = True
    with flask_app.app.test_client() as test_client:
        yield test_client
    flask_app.set_predictor(None)


@pytest.fixture
def loaded(client, monkeypatch):
    stub = StubPredictor()
    flask_app.set_predictor(stub)
    monkeypatch.setattr(flask_app, 'get_predictor', lambda: stub)
    return stub


@pytest.fixture
def unloaded(client, monkeypatch):
    flask_app.set_predictor(None)
    monkeypatch.setattr(flask_app, 'get_predictor', lambda: None)


def test_health_is_up_even_without_a_model(client, unloaded):
    response = client.get('/health')

    assert response.status_code == 200
    assert response.json['status'] == 'healthy'
    assert response.json['model_loaded'] is False


def test_ready_is_503_without_a_model(client, unloaded):
    response = client.get('/ready')

    assert response.status_code == 503
    assert response.json['status'] == 'not_ready'


def test_ready_is_200_with_a_model(client, loaded):
    response = client.get('/ready')

    assert response.status_code == 200
    assert response.json['status'] == 'ready'


def test_model_info_describes_the_loaded_model(client, loaded):
    response = client.get('/model_info')

    assert response.status_code == 200
    assert response.json['framework'] == 'pytorch'
    assert 'fire' in response.json['classes']
    assert response.json['architecture'].startswith('maskrcnn_')


def test_model_info_is_503_without_a_model(client, unloaded):
    assert client.get('/model_info').status_code == 503


def test_predict_returns_detections(client, loaded):
    response = client.post('/predict', data={'file': upload()},
                           content_type='multipart/form-data')

    assert response.status_code == 200
    body = response.json
    assert body['status'] == 'success'
    assert body['is_fire'] is True
    assert body['num_detections'] == 1
    assert body['filename'] == 'fire.png'
    assert 'masks' not in body           # raw arrays must never reach the client
    assert body['inference_ms'] >= 0
    assert body['detections'][0]['class_name'] == 'fire'


def test_predict_can_return_an_overlay(client, loaded):
    response = client.post('/predict?overlay=true', data={'file': upload()},
                           content_type='multipart/form-data')

    assert response.status_code == 200
    assert isinstance(response.json['overlay_png_base64'], str)
    assert len(response.json['overlay_png_base64']) > 0


def test_predict_without_a_model_is_503(client, unloaded):
    response = client.post('/predict', data={'file': upload()},
                           content_type='multipart/form-data')
    assert response.status_code == 503


def test_predict_without_a_file_is_400(client, loaded):
    response = client.post('/predict', data={}, content_type='multipart/form-data')

    assert response.status_code == 400
    assert 'file' in response.json['error']


def test_predict_rejects_unsupported_extensions(client, loaded):
    response = client.post('/predict', data={'file': upload('notes.txt')},
                           content_type='multipart/form-data')

    assert response.status_code == 400
    assert 'not allowed' in response.json['error']


def test_predict_rejects_a_corrupt_image(client, loaded):
    response = client.post('/predict', data={'file': upload('broken.png', b'nope')},
                           content_type='multipart/form-data')

    assert response.status_code == 400
    assert 'decode' in response.json['error']


def test_predict_rejects_an_empty_filename(client, loaded):
    response = client.post('/predict', data={'file': upload('')},
                           content_type='multipart/form-data')
    assert response.status_code == 400


def test_batch_predict_reports_per_file_status(client, loaded):
    response = client.post(
        '/batch_predict',
        data={'files': [upload('a.png'), upload('b.png'), upload('c.txt')]},
        content_type='multipart/form-data')

    assert response.status_code == 200
    body = response.json
    assert body['total'] == 3
    assert body['succeeded'] == 2
    assert body['failed'] == 1
    assert body['fire_detected'] == 2
    assert body['results'][2]['status'] == 'error'


def test_batch_predict_without_files_is_400(client, loaded):
    response = client.post('/batch_predict', data={},
                           content_type='multipart/form-data')
    assert response.status_code == 400


def test_batch_predict_enforces_the_batch_limit(client, loaded, monkeypatch):
    monkeypatch.setattr(flask_app, 'MAX_BATCH_SIZE', 2)

    response = client.post(
        '/batch_predict',
        data={'files': [upload('a.png'), upload('b.png'), upload('c.png')]},
        content_type='multipart/form-data')

    assert response.status_code == 400
    assert 'Too many files' in response.json['error']


def test_oversized_upload_is_rejected(client, loaded, monkeypatch):
    monkeypatch.setitem(flask_app.app.config, 'MAX_CONTENT_LENGTH', 1024)

    response = client.post('/predict',
                           data={'file': upload('big.png', b'0' * 4096)},
                           content_type='multipart/form-data')

    assert response.status_code == 413
    assert response.json['status'] == 'error'


def test_metrics_are_prometheus_formatted(client, loaded):
    client.post('/predict', data={'file': upload()},
                content_type='multipart/form-data')

    response = client.get('/metrics')

    assert response.status_code == 200
    assert response.headers['Content-Type'].startswith('text/plain')
    body = response.get_data(as_text=True)
    assert '# TYPE fire_detection_requests_total counter' in body
    assert 'fire_detection_images_processed_total' in body
    assert 'fire_detection_model_loaded 1' in body


def test_unknown_route_returns_json_404(client, loaded):
    response = client.get('/does-not-exist')

    assert response.status_code == 404
    assert response.json['status'] == 'error'


def test_wrong_method_returns_json_405(client, loaded):
    response = client.get('/predict')

    assert response.status_code == 405
    assert response.json['status'] == 'error'


def test_decode_image_round_trips_rgb():
    image = make_fire_image()

    decoded = flask_app.decode_image(png_bytes(image))

    np.testing.assert_array_equal(decoded, image)


def test_decode_image_rejects_empty_payload():
    with pytest.raises(ValueError, match='Empty'):
        flask_app.decode_image(b'')


@pytest.mark.parametrize('filename, expected', [
    ('photo.jpg', True), ('photo.JPEG', True), ('scan.tiff', True),
    ('doc.pdf', False), ('noextension', False), ('', False),
])
def test_allowed_file(filename, expected):
    assert flask_app.allowed_file(filename) is expected


# --------------------------------------------------------------------------- #
# Decoded-size limits and environment parsing (Pass 2)
# --------------------------------------------------------------------------- #
def test_decode_rejects_a_decompression_bomb(monkeypatch):
    """A tiny upload can decode to an enormous array; the body limit misses it."""
    bomb = np.zeros((2000, 2000, 3), dtype=np.uint8)
    encoded = png_bytes(bomb)
    assert len(encoded) < 100_000  # a few kB on the wire

    monkeypatch.setattr(flask_app, 'MAX_IMAGE_PIXELS', 1_000_000)
    with pytest.raises(ValueError, match='too large'):
        flask_app.decode_image(encoded)


def test_decode_accepts_an_image_inside_the_pixel_limit(monkeypatch):
    monkeypatch.setattr(flask_app, 'MAX_IMAGE_PIXELS', 1_000_000)
    decoded = flask_app.decode_image(png_bytes())
    assert decoded.shape == (48, 64, 3)


def test_oversized_image_is_a_400_not_a_500(client, loaded, monkeypatch):
    monkeypatch.setattr(flask_app, 'MAX_IMAGE_PIXELS', 100)

    response = client.post('/predict', data={'file': upload()},
                           content_type='multipart/form-data')

    assert response.status_code == 400
    assert 'too large' in response.json['error']


def test_decode_rejects_empty_upload():
    with pytest.raises(ValueError, match='Empty file'):
        flask_app.decode_image(b'')


def test_decode_rejects_non_image_bytes():
    with pytest.raises(ValueError, match='corrupt'):
        flask_app.decode_image(b'this is definitely not a png')


@pytest.mark.parametrize('raw,cast,kwargs', [
    ('half', float, {}),
    ('', int, {}),
    ('12.5', int, {}),
])
def test_env_number_rejects_unparseable_values(monkeypatch, raw, cast, kwargs):
    monkeypatch.setenv('FIRE_TEST_VAR', raw)
    if raw == '':
        assert flask_app._env_number('FIRE_TEST_VAR', 7, cast, **kwargs) == 7
        return
    with pytest.raises(ValueError, match='FIRE_TEST_VAR'):
        flask_app._env_number('FIRE_TEST_VAR', 7, cast, **kwargs)


def test_env_number_enforces_bounds(monkeypatch):
    monkeypatch.setenv('FIRE_TEST_VAR', '5')
    with pytest.raises(ValueError, match='must be <= 1'):
        flask_app._env_number('FIRE_TEST_VAR', 0.5, float, 0.0, 1.0)
    with pytest.raises(ValueError, match='must be >= 10'):
        flask_app._env_number('FIRE_TEST_VAR', 0.5, float, 10.0)


def test_env_number_falls_back_to_the_default_when_unset(monkeypatch):
    monkeypatch.delenv('FIRE_TEST_VAR', raising=False)
    assert flask_app._env_number('FIRE_TEST_VAR', 3, int) == 3


# --------------------------------------------------------------------------- #
# Model load back-off (Pass 2)
# --------------------------------------------------------------------------- #
def test_failed_model_load_is_not_retried_on_every_request(client, monkeypatch):
    attempts = []

    def explode(*args, **kwargs):
        attempts.append(1)
        raise FileNotFoundError('no checkpoint')

    flask_app.set_predictor(None)
    monkeypatch.setattr(flask_app, '_load_error', None)
    monkeypatch.setattr(flask_app.FirePredictor, 'from_checkpoint',
                        staticmethod(explode))
    monkeypatch.setattr(flask_app, 'MODEL_LOAD_RETRY_SECONDS', 3600.0)

    for _ in range(5):
        assert flask_app.get_predictor() is None

    assert len(attempts) == 1
    assert '/ready' and client.get('/ready').status_code == 503
    flask_app._next_load_attempt = 0.0


def test_model_load_is_retried_once_the_backoff_expires(client, monkeypatch):
    attempts = []

    def explode(*args, **kwargs):
        attempts.append(1)
        raise FileNotFoundError('no checkpoint')

    flask_app.set_predictor(None)
    monkeypatch.setattr(flask_app.FirePredictor, 'from_checkpoint',
                        staticmethod(explode))
    monkeypatch.setattr(flask_app, 'MODEL_LOAD_RETRY_SECONDS', 0.0)

    flask_app.get_predictor()
    flask_app.get_predictor()

    assert len(attempts) == 2
    flask_app._next_load_attempt = 0.0


def test_set_predictor_clears_the_backoff():
    flask_app._next_load_attempt = float('inf')
    stub = StubPredictor()
    flask_app.set_predictor(stub)
    assert flask_app._next_load_attempt == 0.0
    assert flask_app.get_predictor() is stub
    flask_app.set_predictor(None)


def test_model_info_advertises_the_pixel_limit(client, loaded):
    assert client.get('/model_info').json['max_image_pixels'] == \
        flask_app.MAX_IMAGE_PIXELS


# --------------------------------------------------------------------------- #
# Failure isolation in batch mode (Pass 2)
# --------------------------------------------------------------------------- #
def test_batch_keeps_going_when_one_file_is_corrupt(client, loaded):
    response = client.post('/batch_predict', data={
        'files': [upload('good.png'), upload('bad.png', b'not an image')],
    }, content_type='multipart/form-data')

    assert response.status_code == 200
    body = response.json
    assert body['succeeded'] == 1 and body['failed'] == 1
    statuses = {r['filename']: r['status'] for r in body['results']}
    assert statuses == {'good.png': 'success', 'bad.png': 'error'}


def test_unexpected_batch_failure_does_not_leak_internals(client, loaded,
                                                          monkeypatch):
    def boom(image):
        raise RuntimeError('/secret/path/to/model.pt is missing')

    monkeypatch.setattr(loaded, 'predict', boom)

    response = client.post('/batch_predict', data={'files': [upload()]},
                           content_type='multipart/form-data')

    assert response.status_code == 200
    assert response.json['results'][0]['error'] == 'Prediction failed'
    assert 'secret' not in response.get_data(as_text=True)


def test_unexpected_single_failure_does_not_leak_internals(client, loaded,
                                                           monkeypatch):
    def boom(image):
        raise RuntimeError('/secret/path/to/model.pt is missing')

    monkeypatch.setattr(loaded, 'predict', boom)

    response = client.post('/predict', data={'file': upload()},
                           content_type='multipart/form-data')

    assert response.status_code == 500
    assert 'secret' not in response.get_data(as_text=True)
