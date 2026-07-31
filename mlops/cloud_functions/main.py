"""
Cloud Run Function that rolls a new image out to the GKE deployment.

Cloud Build publishes a Pub/Sub message once the container image lands in
Artifact Registry; this function receives it and patches the deployment so the
cluster performs a rolling update.

Every setting comes from the environment - nothing about the target project is
hardcoded:

    GCP_PROJECT_ID   GCP project that owns the cluster (required)
    GKE_CLUSTER      cluster name              (default: fire-detection-cluster)
    GKE_ZONE         cluster zone              (default: us-central1-a)
    DEPLOYMENT_NAME  deployment to patch       (default: fire-detection)
    CONTAINER_NAME   container inside the pod  (default: fire-detection)
    NAMESPACE        Kubernetes namespace      (default: default)
"""
from __future__ import annotations

import base64
import json
import logging
import os
from typing import Any, Dict, Optional, Tuple

import functions_framework
from kubernetes import client as k8s_client
from kubernetes import config as k8s_config

logging.basicConfig(level=os.environ.get('LOG_LEVEL', 'INFO'))
logger = logging.getLogger(__name__)

try:  # Cloud Logging is available in the runtime but not in unit tests.
    import google.cloud.logging

    google.cloud.logging.Client().setup_logging()
except Exception as exc:  # pragma: no cover - depends on the runtime
    logger.debug('Cloud Logging not configured: %s', exc)


def _setting(name: str, default: Optional[str] = None, required: bool = False) -> str:
    value = os.environ.get(name, default)
    if required and not value:
        raise RuntimeError(f'Required environment variable {name} is not set')
    return value or ''


def get_settings() -> Dict[str, str]:
    """Read the deployment target from the environment."""
    return {
        'project_id': _setting('GCP_PROJECT_ID', os.environ.get('GCP_PROJECT'), required=True),
        'cluster': _setting('GKE_CLUSTER', 'fire-detection-cluster'),
        'zone': _setting('GKE_ZONE', 'us-central1-a'),
        'deployment': _setting('DEPLOYMENT_NAME', 'fire-detection'),
        'container': _setting('CONTAINER_NAME', 'fire-detection'),
        'namespace': _setting('NAMESPACE', 'default'),
    }


def decode_pubsub_message(cloud_event) -> Dict[str, Any]:
    """Extract the JSON payload from a Pub/Sub push CloudEvent."""
    data = getattr(cloud_event, 'data', None) or {}
    message = data.get('message') or {}
    payload = message.get('data')

    if payload is None:
        raise ValueError('Pub/Sub message carried no data field')

    if isinstance(payload, (bytes, bytearray)):
        decoded = bytes(payload)
    else:
        # Pub/Sub base64 encodes the body; fall back to the raw string when a
        # test or an emulator delivers plain JSON.
        try:
            decoded = base64.b64decode(payload, validate=True)
        except Exception:
            decoded = str(payload).encode('utf-8')

    try:
        body = json.loads(decoded.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f'Pub/Sub message was not valid JSON: {exc}') from exc

    # A JSON array or scalar is valid JSON but has no .get(); without this the
    # handler would raise AttributeError and the function would 500 instead of
    # rejecting the message.
    if not isinstance(body, dict):
        raise ValueError(
            'Pub/Sub message must be a JSON object with an "image_uri" field; '
            f'got {type(body).__name__}')
    return body


def validate_image_uri(value: Any) -> str:
    """Check the image reference before it is written into a Deployment patch."""
    if not isinstance(value, str):
        raise ValueError(
            f'image_uri must be a string, got {type(value).__name__}')
    uri = value.strip()
    if not uri:
        raise ValueError('image_uri is empty')
    if len(uri) > 512:
        raise ValueError(f'image_uri is implausibly long ({len(uri)} characters)')
    if any(character.isspace() for character in uri):
        raise ValueError('image_uri must not contain whitespace')
    return uri


def load_kubernetes_config() -> None:
    """Authenticate against the cluster, in-cluster first then local kubeconfig."""
    try:
        k8s_config.load_incluster_config()
        logger.info('Loaded in-cluster Kubernetes configuration')
    except k8s_config.ConfigException:
        k8s_config.load_kube_config()
        logger.info('Loaded local kubeconfig')


def update_gke_deployment(image_uri: str, settings: Optional[Dict[str, str]] = None,
                          apps_api=None) -> bool:
    """Patch the deployment's container image and trigger a rolling update.

    Args:
        image_uri: Fully qualified image reference to roll out.
        settings: Target description; read from the environment when omitted.
        apps_api: Injected ``AppsV1Api`` (used by tests).

    Returns:
        True when the patch was accepted.
    """
    settings = settings or get_settings()

    if apps_api is None:
        load_kubernetes_config()
        apps_api = k8s_client.AppsV1Api()

    logger.info('Patching deployment %s/%s in cluster %s (%s) to %s',
                settings['namespace'], settings['deployment'],
                settings['cluster'], settings['zone'], image_uri)

    patch = {
        'spec': {
            'template': {
                'spec': {
                    'containers': [
                        {'name': settings['container'], 'image': image_uri}
                    ]
                }
            }
        }
    }

    try:
        apps_api.patch_namespaced_deployment(
            name=settings['deployment'],
            namespace=settings['namespace'],
            body=patch,
        )
    except Exception as exc:
        logger.error('Failed to patch deployment %s: %s', settings['deployment'], exc)
        return False

    logger.info('Deployment %s now runs %s', settings['deployment'], image_uri)
    return True


@functions_framework.cloud_event
def update_deployment(cloud_event) -> Tuple[Dict[str, Any], int]:
    """Entrypoint: handle one Pub/Sub notification about a new image."""
    try:
        message = decode_pubsub_message(cloud_event)
    except ValueError as exc:
        logger.error('Malformed Pub/Sub message: %s', exc)
        return {'status': 'error', 'message': str(exc)}, 400

    build_id = message.get('build_id')
    raw_image_uri = message.get('image_uri')
    logger.info('Received build %s for image %s', build_id, raw_image_uri)

    if raw_image_uri is None or raw_image_uri == '':
        logger.error('Message is missing image_uri: %s', message)
        return {'status': 'error', 'message': 'Missing image_uri'}, 400

    try:
        image_uri = validate_image_uri(raw_image_uri)
    except ValueError as exc:
        logger.error('Rejecting image_uri %r: %s', raw_image_uri, exc)
        return {'status': 'error', 'message': str(exc)}, 400

    try:
        settings = get_settings()
    except RuntimeError as exc:
        logger.error('Function is misconfigured: %s', exc)
        return {'status': 'error', 'message': str(exc)}, 500

    if update_gke_deployment(image_uri, settings):
        return {
            'status': 'success',
            'message': f'Deployment updated with {image_uri}',
            'image_uri': image_uri,
            'build_id': build_id,
        }, 200

    return {'status': 'error',
            'message': f'Failed to update deployment with {image_uri}'}, 500
