# MLOps Module Documentation

Complete MLOps pipeline for deploying the Mask R-CNN fire detection model on Google Cloud Platform.

## Module Overview

The MLOps module implements an end-to-end CI/CD pipeline:

1. **Code Commit** → GitHub
2. **Build Trigger** → Cloud Build
3. **Image Creation** → Docker
4. **Registry Storage** → Artifact Registry
5. **Event Publishing** → Cloud Pub/Sub
6. **Auto Deployment** → Cloud Run Function
7. **Orchestration** → Google Kubernetes Engine (GKE)
8. **Load Balancing** → Kubernetes Service

## Directory Structure

```
mlops/
├── flask_app/
│   ├── app.py              # Flask REST API application
│   ├── Dockerfile          # Multi-stage Docker build
│   ├── uwsgi.ini           # uWSGI server configuration
│   ├── requirements.txt    # Python dependencies
│   └── uploads/            # Temporary image uploads
│
├── cloudbuild/
│   └── cloudbuild.yaml    # Cloud Build CI/CD pipeline definition
│
├── gke/
│   └── deployment.yaml    # Kubernetes deployment manifests
│
├── cloud_functions/
│   ├── main.py            # Auto-deployment Cloud Function
│   └── requirements.txt    # Function dependencies
│
├── DEPLOYMENT_GUIDE.md     # Complete deployment instructions
└── README.md               # This file
```

## Quick Start

### 1. Local Testing

```bash
# Build Docker image
docker build -t fire-detection:local -f mlops/flask_app/Dockerfile .

# Run container
docker run -p 5000:5000 \
  -e MODEL_PATH=/models/fire_detection_model.h5 \
  fire-detection:local

# Test API
curl http://localhost:5000/health
```

### 2. GCP Deployment

```bash
# Setup GCP project
gcloud projects create fire-detection-mlops
export PROJECT_ID=$(gcloud config get-value project)

# Enable required APIs
gcloud services enable cloudbuild.googleapis.com artifactregistry.googleapis.com

# Create Artifact Registry
gcloud artifacts repositories create fire-detection \
  --repository-format=docker --location=us-central1

# Create GKE cluster
gcloud container clusters create fire-detection-cluster \
  --zone us-central1-a --num-nodes 3

# Deploy
kubectl apply -f mlops/gke/deployment.yaml
```

## Components

### Flask Application (`app.py`)

REST API for model inference:

#### Architecture
- **Framework**: Flask 2.0+
- **Server**: uWSGI
- **Port**: 5000

#### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check endpoint |
| GET | `/model_info` | Model information |
| POST | `/predict` | Single image prediction |
| POST | `/batch_predict` | Multiple image prediction |

#### Health Check
```bash
curl http://localhost:5000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2024-01-29T10:30:00"
}
```

#### Single Image Prediction
```bash
curl -X POST -F "file=@fire_image.jpg" \
  http://localhost:5000/predict
```

Response:
```json
{
  "predicted_class": "Fire",
  "confidence": 0.95,
  "is_fire": true,
  "bounding_box": {
    "x1": 100,
    "y1": 150,
    "x2": 300,
    "y2": 400
  },
  "filename": "fire_image.jpg",
  "original_dimensions": {"height": 480, "width": 640},
  "status": "success"
}
```

#### Batch Prediction
```bash
curl -X POST -F "files=@fire1.jpg" -F "files=@fire2.jpg" \
  http://localhost:5000/batch_predict
```

Response:
```json
{
  "results": [
    {"predicted_class": "Fire", "confidence": 0.95, "filename": "fire1.jpg", "status": "success"},
    {"predicted_class": "Background", "confidence": 0.88, "filename": "fire2.jpg", "status": "success"}
  ],
  "total": 2,
  "timestamp": "2024-01-29T10:30:00"
}
```

#### Model Information
```bash
curl http://localhost:5000/model_info
```

### Docker Configuration (`Dockerfile`)

Multi-stage Docker build:

**Build Stage**
- Compiles Python dependencies
- Installs OpenCV and system libraries

**Runtime Stage**
- Minimal image size
- Non-root user (`appuser`)
- Health checks
- Security best practices

#### Building Docker Image

```bash
# Build with specific tag
docker build -t fire-detection:v1.0 -f mlops/flask_app/Dockerfile .

# Build for production
docker build \
  --build-arg PYTHON_VERSION=3.9 \
  -t fire-detection:latest \
  -f mlops/flask_app/Dockerfile .
```

#### Docker Security Features

1. **Multi-stage build**: Reduces final image size
2. **Non-root user**: Runs as `appuser` (UID 1000)
3. **Read-only filesystem**: Limited attack surface
4. **Health checks**: Automatic restart on failure
5. **Minimal base image**: Uses `python:3.9-slim`

#### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| MODEL_PATH | `/models/fire_detection_model.h5` | Path to trained model |
| PYTHONUNBUFFERED | `1` | Enable stdout logging |
| PYTHONDONTWRITEBYTECODE | `1` | Skip bytecode generation |
| FLASK_APP | `app.py` | Flask application module |

### uWSGI Configuration (`uwsgi.ini`)

Production application server:

```ini
socket = 0.0.0.0:5000      # Listen on port 5000
master = True              # Enable master process
processes = 4              # Worker processes
threads = 2                # Threads per worker
worker_reload_time = 120   # Reload timeout
http_timeout = 300         # Request timeout
```

#### Performance Tuning

```ini
# For high traffic
processes = 8              # More workers
threads = 4                # More threads
buffer_size = 65536        # Larger buffer

# For limited resources
processes = 2              # Fewer workers
threads = 1                # Single thread
buffer_size = 16384        # Smaller buffer
```

### Cloud Build Pipeline (`cloudbuild.yaml`)

Automated CI/CD pipeline:

#### Build Steps

1. **Build Docker Image**
   ```
   docker build -t IMAGE_URI .
   ```

2. **Push to Artifact Registry**
   ```
   docker push IMAGE_URI
   ```

3. **Download Model from GCS**
   ```
   gsutil cp gs://bucket/model.h5 .
   ```

4. **Publish Pub/Sub Message**
   - Publishes image metadata
   - Triggers Cloud Run Function

5. **Update GKE Deployment**
   - Updates deployment image
   - Starts rolling update

#### Build Configuration

```yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'IMAGE_URI', '.']
  
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'IMAGE_URI']
```

#### Substitution Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| _REGION | `us-central1` | GCP region |
| _ARTIFACT_REPO | `fire-detection` | Artifact Registry name |
| _MODEL_GCS_PATH | `gs://bucket/model.h5` | Model storage path |
| _GKE_CLUSTER | `fire-detection-cluster` | Cluster name |
| _GKE_ZONE | `us-central1-a` | Cluster zone |

### Kubernetes Deployment (`deployment.yaml`)

Kubernetes manifests for GKE:

#### Components

1. **Deployment**
   - 3 replicas (configurable)
   - Rolling update strategy
   - Resource limits and requests
   - Health probes

2. **Service**
   - LoadBalancer type
   - Port 80 → 5000 (internal)
   - Session affinity

3. **HorizontalPodAutoscaler**
   - Scale 3-10 replicas
   - CPU threshold: 70%
   - Memory threshold: 80%

4. **PersistentVolumeClaim**
   - 5GB storage
   - ReadOnlyMany access mode

5. **ServiceAccount**
   - RBAC configuration
   - Pod permissions

#### Resource Requirements

```yaml
resources:
  requests:
    memory: "2Gi"
    cpu: "1"
  limits:
    memory: "4Gi"
    cpu: "2"
```

#### Health Probes

```yaml
readinessProbe:
  httpGet:
    path: /health
    port: 5000
  initialDelaySeconds: 30
  periodSeconds: 10

livenessProbe:
  httpGet:
    path: /health
    port: 5000
  initialDelaySeconds: 60
  periodSeconds: 30
```

### Cloud Function (`main.py`)

Serverless function for automatic deployment:

#### Triggers
- Cloud Pub/Sub topic messages
- Triggered on successful build

#### Functionality
1. Parse Pub/Sub message
2. Extract image URI
3. Update GKE deployment
4. Log operation result

#### Configuration

```python
PROJECT_ID = "your-project-id"
CLUSTER_NAME = "fire-detection-cluster"
CLUSTER_ZONE = "us-central1-a"
DEPLOYMENT_NAME = "fire-detection"
NAMESPACE = "default"
```

#### Event Flow

```
Build Complete
    ↓
Pub/Sub Message
    ↓
Cloud Function Triggered
    ↓
Update Kubernetes Deployment
    ↓
RollingUpdate Starts
    ↓
New Pods Launched
    ↓
Old Pods Terminated
    ↓
Zero Downtime Deployment
```

## Deployment Workflow

### 1. Code Changes

Developer makes changes to:
- Model code (`segmentation/`)
- API code (`flask_app/app.py`)
- Configuration files

### 2. GitHub Commit

```bash
git add .
git commit -m "Update model or API"
git push origin main
```

### 3. Cloud Build Trigger

Webhook automatically:
- Clones repository
- Builds Docker image
- Runs tests (if configured)
- Pushes to Artifact Registry

### 4. Pub/Sub Message

Build completion publishes:
```json
{
  "image_uri": "us-central1-docker.pkg.dev/PROJECT/fire-detection/fire-detection:abc123",
  "image_digest": "sha256:...",
  "build_id": "abc123def456"
}
```

### 5. Cloud Function

Function updates:
- Kubernetes deployment image
- Triggers rolling update
- Maintains availability

### 6. Rolling Update

Kubernetes performs:
- Starts new pods with new image
- Waits for readiness
- Terminates old pods
- Verifies health

## Monitoring and Logging

### Cloud Build Logs

```bash
# List recent builds
gcloud builds list --limit 5

# View build details
gcloud builds log BUILD_ID --stream

# Download build logs
gcloud builds log BUILD_ID > build.log
```

### GKE Pod Logs

```bash
# Stream pod logs
kubectl logs -f deployment/fire-detection

# View specific pod
kubectl logs POD_NAME

# View previous logs
kubectl logs POD_NAME --previous

# View all pods in deployment
kubectl logs -l app=fire-detection --tail=100
```

### Cloud Logging

```bash
# Query application logs
gcloud logging read "resource.type=k8s_container" \
  --limit 50 --format json

# Create log sink
gcloud logging sinks create SINK_NAME \
  storage.googleapis.com/LOG_BUCKET
```

## Performance Tuning

### API Performance

```python
# Enable caching
@app.route('/model_info', methods=['GET'])
@lru_cache(maxsize=1)
def model_info():
    ...

# Batch processing
# Use /batch_predict for multiple images

# Async processing (advanced)
from celery import Celery
celery = Celery(app.name)
```

### Container Performance

```dockerfile
# Use multi-stage to reduce image size
FROM python:3.9-slim as builder

# Final stage
FROM python:3.9-slim
COPY --from=builder /app /app
```

### Kubernetes Performance

```yaml
# Resource optimization
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "2Gi"
    cpu: "1"

# Pod disruption budget
podDisruptionBudget:
  minAvailable: 1
```

## Security Best Practices

### Application Security

1. **Input Validation**
   ```python
   if not allowed_file(filename):
       return error_response()
   ```

2. **File Size Limits**
   ```python
   MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
   ```

3. **Error Handling**
   ```python
   try:
       # Process
   except Exception as e:
       # Log and return generic error
   ```

### Container Security

1. **Non-root User**
   ```dockerfile
   USER appuser
   ```

2. **Read-only Filesystem**
   ```dockerfile
   RUN chmod 555 /app
   ```

3. **Health Checks**
   ```dockerfile
   HEALTHCHECK CMD curl -f http://localhost:5000/health
   ```

### Kubernetes Security

1. **Network Policies**
   ```yaml
   networkPolicy:
     policyTypes:
     - Ingress
     - Egress
   ```

2. **RBAC**
   ```yaml
   serviceAccount:
     name: fire-detection-sa
   ```

3. **Pod Security Policy**
   ```yaml
   securityContext:
     runAsNonRoot: true
     readOnlyRootFilesystem: false
   ```

## Troubleshooting

### Build Fails

```bash
# Check build logs
gcloud builds log BUILD_ID --stream

# Verify Dockerfile
docker build -f mlops/flask_app/Dockerfile .

# Test locally
docker run --rm fire-detection:latest
```

### Deployment Not Updating

```bash
# Check Cloud Function logs
gcloud functions describe update-gke-deployment

# Verify Pub/Sub messages
gcloud pubsub subscriptions pull fire-detection-builds-sub

# Check GKE events
kubectl get events
kubectl describe deployment fire-detection
```

### Pod Crashes

```bash
# Check pod status
kubectl describe pod POD_NAME

# View logs
kubectl logs POD_NAME

# Check resource availability
kubectl top nodes
kubectl top pods

# Try different resource requests
# Edit deployment and adjust resources
```

### API Errors

```bash
# Test health endpoint
curl -v http://SERVICE_IP:5000/health

# Check model path
kubectl exec POD_NAME -- ls -la /models/

# Verify file permissions
kubectl exec POD_NAME -- stat /models/fire_detection_model.h5
```

## Cost Optimization

### GCP Services Cost

| Service | Optimization |
|---------|----------------|
| Cloud Build | Use spot instances, cache docker layers |
| Artifact Registry | Clean up old images |
| GKE | Use preemptible nodes, auto-scaling |
| Cloud Run | Use only when needed |
| Pub/Sub | Archive old messages |

### Implementation

```bash
# Delete old images
gcloud artifacts docker images list \
  --repository=fire-detection \
  --location=us-central1 \
  --format="value(image)" | \
  xargs -I {} gcloud artifacts docker images delete {}

# Use preemptible nodes
gcloud container node-pools create preemptible-pool \
  --cluster=fire-detection-cluster \
  --preemptible

# Set resource quotas
kubectl set quota QUOTA_NAME \
  --hard=pods=10,limits.cpu=10,limits.memory=20Gi
```

## Next Steps

1. **Setup GCP**: Follow DEPLOYMENT_GUIDE.md
2. **Configure CI/CD**: Setup Cloud Build trigger
3. **Deploy Model**: Push code to GitHub
4. **Monitor**: Watch Cloud Build and GKE
5. **Test API**: Use curl or Postman
6. **Optimize**: Tune resources based on metrics
7. **Scale**: Adjust HPA thresholds
8. **Secure**: Implement security measures

## References

- [GCP Cloud Build](https://cloud.google.com/build/docs)
- [GCP Artifact Registry](https://cloud.google.com/artifact-registry/docs)
- [GCP GKE](https://cloud.google.com/kubernetes-engine/docs)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Docker Documentation](https://docs.docker.com/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)

---

**Module Version**: 1.0.0
**Last Updated**: January 29, 2024
