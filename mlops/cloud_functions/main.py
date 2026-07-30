"""
Cloud Run Function to handle Pub/Sub messages and update GKE deployments
Triggered when new Docker image is pushed to Artifact Registry
"""
import json
import functions_framework
import google.cloud.container_v1 as container_v1
import google.cloud.logging
import logging
from google.api_core import retry
from google.auth import default

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize GCP clients
client_logging = google.cloud.logging.Client()
client_logging.setup_logging()

# GKE configuration
PROJECT_ID = "your-project-id"
CLUSTER_NAME = "fire-detection-cluster"
CLUSTER_ZONE = "us-central1-a"
DEPLOYMENT_NAME = "fire-detection"
NAMESPACE = "default"


@functions_framework.cloud_event
def update_deployment(cloud_event):
    """
    Cloud Function triggered by Pub/Sub message
    Updates GKE deployment with new image
    
    Args:
        cloud_event: Cloud Event containing Pub/Sub message
    """
    try:
        # Parse Pub/Sub message
        pubsub_message = cloud_event.data["message"]["data"]
        
        if isinstance(pubsub_message, str):
            message_json = json.loads(pubsub_message)
        else:
            import base64
            message_json = json.loads(base64.b64decode(pubsub_message))
        
        logger.info(f"Received message: {message_json}")
        
        # Extract image information
        image_uri = message_json.get('image_uri')
        image_digest = message_json.get('image_digest')
        build_id = message_json.get('build_id')
        
        if not image_uri:
            logger.error("No image_uri in message")
            return "Error: Missing image_uri", 400
        
        logger.info(f"Updating deployment with image: {image_uri}")
        
        # Update GKE deployment
        success = update_gke_deployment(image_uri)
        
        if success:
            logger.info(f"Successfully updated deployment with {image_uri}")
            return {
                'status': 'success',
                'message': f"Deployment updated with {image_uri}",
                'image_uri': image_uri,
                'build_id': build_id
            }, 200
        else:
            logger.error("Failed to update deployment")
            return {
                'status': 'error',
                'message': 'Failed to update deployment'
            }, 500
            
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}", exc_info=True)
        return {
            'status': 'error',
            'message': str(e)
        }, 500


def update_gke_deployment(image_uri):
    """
    Update GKE deployment with new image
    
    Args:
        image_uri: Full URI of new Docker image
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create Kubernetes API client
        credentials, project = default()
        client = container_v1.ClusterManagerClient(credentials=credentials)
        
        # Get cluster
        cluster_path = f"projects/{PROJECT_ID}/zones/{CLUSTER_ZONE}/clusters/{CLUSTER_NAME}"
        
        logger.info(f"Accessing cluster: {cluster_path}")
        
        # Get cluster credentials (simplified)
        # In production, use the Kubernetes Python client
        from kubernetes import client as k8s_client, config
        
        # Load cluster config
        try:
            config.load_incluster_config()
        except:
            # Fallback for local testing
            config.load_kube_config()
        
        # Create API client
        v1_apps = k8s_client.AppsV1Api()
        
        # Get current deployment
        try:
            deployment = v1_apps.read_namespaced_deployment(
                name=DEPLOYMENT_NAME,
                namespace=NAMESPACE
            )
        except Exception as e:
            logger.error(f"Failed to read deployment: {e}")
            return False
        
        # Update image
        if deployment.spec.template.spec.containers:
            deployment.spec.template.spec.containers[0].image = image_uri
        
        # Apply update
        try:
            v1_apps.patch_namespaced_deployment(
                name=DEPLOYMENT_NAME,
                namespace=NAMESPACE,
                body=deployment
            )
            logger.info(f"Deployment patched successfully with image: {image_uri}")
            return True
        except Exception as e:
            logger.error(f"Failed to patch deployment: {e}")
            return False
            
    except Exception as e:
        logger.error(f"Error updating deployment: {str(e)}", exc_info=True)
        return False
