"""Tests for the Cloud Run Function that rolls new images out to GKE."""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip('functions_framework')
pytest.importorskip('kubernetes')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'mlops' / 'cloud_functions'))

import main as cloud_function  # noqa: E402


class FakeCloudEvent:
    def __init__(self, data):
        self.data = data


class FakeAppsApi:
    """Captures the patch instead of talking to a real cluster."""

    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def patch_namespaced_deployment(self, name, namespace, body):
        if self.fail:
            raise RuntimeError('cluster unreachable')
        self.calls.append({'name': name, 'namespace': namespace, 'body': body})
        return body


def pubsub_event(payload) -> FakeCloudEvent:
    encoded = base64.b64encode(json.dumps(payload).encode('utf-8')).decode('ascii')
    return FakeCloudEvent({'message': {'data': encoded}})


@pytest.fixture
def settings(monkeypatch):
    monkeypatch.setenv('GCP_PROJECT_ID', 'demo-project')
    monkeypatch.setenv('GKE_CLUSTER', 'demo-cluster')
    monkeypatch.setenv('DEPLOYMENT_NAME', 'fire-detection')
    monkeypatch.setenv('CONTAINER_NAME', 'fire-detection')
    monkeypatch.setenv('NAMESPACE', 'production')
    return cloud_function.get_settings()


def test_settings_come_from_the_environment(settings):
    assert settings['project_id'] == 'demo-project'
    assert settings['cluster'] == 'demo-cluster'
    assert settings['namespace'] == 'production'
    assert settings['zone'] == 'us-central1-a'  # documented default


def test_missing_project_id_is_reported(monkeypatch):
    monkeypatch.delenv('GCP_PROJECT_ID', raising=False)
    monkeypatch.delenv('GCP_PROJECT', raising=False)

    with pytest.raises(RuntimeError, match='GCP_PROJECT_ID'):
        cloud_function.get_settings()


def test_decodes_base64_pubsub_payload():
    message = cloud_function.decode_pubsub_message(
        pubsub_event({'image_uri': 'repo/img:abc', 'build_id': '42'}))

    assert message['image_uri'] == 'repo/img:abc'
    assert message['build_id'] == '42'


def test_decodes_plain_json_payload():
    event = FakeCloudEvent({'message': {'data': json.dumps({'image_uri': 'repo/img:1'})}})

    assert cloud_function.decode_pubsub_message(event)['image_uri'] == 'repo/img:1'


def test_rejects_a_message_without_data():
    with pytest.raises(ValueError, match='no data'):
        cloud_function.decode_pubsub_message(FakeCloudEvent({'message': {}}))


def test_rejects_a_non_json_payload():
    event = FakeCloudEvent({'message': {'data': base64.b64encode(b'not json').decode()}})

    with pytest.raises(ValueError, match='not valid JSON'):
        cloud_function.decode_pubsub_message(event)


def test_patch_targets_the_configured_container(settings):
    api = FakeAppsApi()

    assert cloud_function.update_gke_deployment('repo/img:abc', settings, apps_api=api)

    call = api.calls[0]
    assert call['name'] == 'fire-detection'
    assert call['namespace'] == 'production'
    container = call['body']['spec']['template']['spec']['containers'][0]
    assert container == {'name': 'fire-detection', 'image': 'repo/img:abc'}


def test_patch_failure_is_reported_not_raised(settings):
    assert cloud_function.update_gke_deployment(
        'repo/img:abc', settings, apps_api=FakeAppsApi(fail=True)) is False


def test_handler_updates_on_a_valid_message(settings, monkeypatch):
    api = FakeAppsApi()
    real_update = cloud_function.update_gke_deployment
    monkeypatch.setattr(cloud_function, 'update_gke_deployment',
                        lambda uri, cfg=None, apps_api=None: real_update(uri, settings, api))

    body, status = cloud_function.update_deployment.__wrapped__(
        pubsub_event({'image_uri': 'repo/img:abc', 'build_id': '7'}))

    assert status == 200
    assert body['status'] == 'success'
    assert body['build_id'] == '7'
    assert len(api.calls) == 1


def test_handler_rejects_a_message_without_an_image(settings):
    body, status = cloud_function.update_deployment.__wrapped__(
        pubsub_event({'build_id': '7'}))

    assert status == 400
    assert 'image_uri' in body['message']


def test_handler_reports_a_malformed_message(settings):
    body, status = cloud_function.update_deployment.__wrapped__(
        FakeCloudEvent({'message': {}}))

    assert status == 400
    assert body['status'] == 'error'


def test_handler_reports_a_failed_rollout(settings, monkeypatch):
    monkeypatch.setattr(cloud_function, 'update_gke_deployment',
                        lambda *args, **kwargs: False)

    body, status = cloud_function.update_deployment.__wrapped__(
        pubsub_event({'image_uri': 'repo/img:abc'}))

    assert status == 500
    assert body['status'] == 'error'


# --------------------------------------------------------------------------- #
# Malformed payloads (Pass 2)
# --------------------------------------------------------------------------- #
def test_json_array_payload_is_rejected_not_crashed():
    """A JSON array is valid JSON but has no .get(); this used to 500."""
    with pytest.raises(ValueError, match='JSON object'):
        cloud_function.decode_pubsub_message(pubsub_event(['image:latest']))


def test_json_scalar_payload_is_rejected():
    with pytest.raises(ValueError, match='JSON object'):
        cloud_function.decode_pubsub_message(pubsub_event('image:latest'))


def test_handler_returns_400_for_an_array_payload(settings):
    body, status = cloud_function.update_deployment(pubsub_event([1, 2, 3]))
    assert status == 400
    assert body['status'] == 'error'


@pytest.mark.parametrize('value', [
    123,
    ['gcr.io/p/i:1'],
    {'uri': 'gcr.io/p/i:1'},
    'gcr.io/p/i:1 --privileged',
    'gcr.io/p/i:1\nsomething-else',
    '   ',
])
def test_implausible_image_uris_are_rejected(value):
    with pytest.raises(ValueError):
        cloud_function.validate_image_uri(value)


def test_overlong_image_uri_is_rejected():
    with pytest.raises(ValueError, match='implausibly long'):
        cloud_function.validate_image_uri('gcr.io/p/' + 'a' * 600)


def test_valid_image_uri_is_trimmed_and_returned():
    assert cloud_function.validate_image_uri('  gcr.io/p/i:sha256-abc  ') == \
        'gcr.io/p/i:sha256-abc'


def test_handler_rejects_a_non_string_image_uri(settings):
    body, status = cloud_function.update_deployment(
        pubsub_event({'image_uri': 42, 'build_id': 'b1'}))

    assert status == 400
    assert 'must be a string' in body['message']


def test_handler_rejects_an_image_uri_with_whitespace(settings, monkeypatch):
    patched = []
    monkeypatch.setattr(cloud_function, 'update_gke_deployment',
                        lambda *a, **k: patched.append(a) or True)

    body, status = cloud_function.update_deployment(
        pubsub_event({'image_uri': 'gcr.io/p/i:1 evil'}))

    assert status == 400
    assert patched == []
