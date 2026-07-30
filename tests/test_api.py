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
