# Mask R-CNN Fire Detection - MLOps Deployment Guide

This guide provides step-by-step instructions for deploying the Mask R-CNN fire detection model using Google Cloud Platform services.

## Architecture Overview

The MLOps pipeline consists of five main stages:

1. **Local Development** - Model training and testing locally
2. **Source Control** - GitHub repository with CI/CD triggers
3. **Build Stage** - Cloud Build creates Docker images
4. **Registry** - Artifact Registry stores container images
5. **Deployment** - GKE clusters run the containerized application
6. **Automation** - Cloud Run Functions handle Pub/Sub triggers

## Prerequisites

Before starting, ensure you have:

- Google Cloud Platform (GCP) account with billing enabled
- `gcloud` CLI installed and configured
- `kubectl` CLI installed
- Docker installed (for local testing)
- GitHub account with the repository
- Sufficient GCP project quota for:
  - Cloud Build
  - Artifact Registry
  - Google Kubernetes Engine (GKE)
  - Cloud Run Functions
  - Cloud Pub/Sub

## Step 1: Setup GCP Project

### 1.1 Create GCP Project

```bash
gcloud projects create fire-detection-mlops --set-as-default
gcloud config set project fire-detection-mlops
```

### 1.2 Enable Required APIs

```bash
gcloud services enable \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  container.googleapis.com \
  run.googleapis.com \
  pubsub.googleapis.com \
  cloudkms.googleapis.com \
  storage-api.googleapis.com
```

### 1.3 Set Environment Variables

```bash
export PROJECT_ID=$(gcloud config get-value project)
export REGION=us-central1
export ZONE=us-central1-a
export ARTIFACT_REPO=fire-detection
export GKE_CLUSTER=fire-detection-cluster
```

## Step 2: Setup Cloud Storage

### 2.1 Create GCS Buckets

```bash
# Model storage
gsutil mb -l $REGION gs://${PROJECT_ID}-models/

# Build artifacts
gsutil mb -l $REGION gs://${PROJECT_ID}-builds/

# Application logs
gsutil mb -l $REGION gs://${PROJECT_ID}-logs/
```

### 2.2 Upload Model

```bash
gsutil cp segmentation/weights/fire_detection_model.h5 \
  gs://${PROJECT_ID}-models/fire_detection_model.h5

# Set appropriate permissions
gsutil iam ch serviceAccount:cloud-build@${PROJECT_ID}.iam.gserviceaccount.com:objectViewer \
  gs://${PROJECT_ID}-models/
```

## Step 3: Setup Artifact Registry

### 3.1 Create Docker Repository

```bash
gcloud artifacts repositories create $ARTIFACT_REPO \
  --repository-format=docker \
  --location=$REGION \
  --description="Fire Detection Model Containers"
```

### 3.2 Configure Docker Authentication

```bash
gcloud auth configure-docker ${REGION}-docker.pkg.dev
```

## Step 4: Setup Cloud Build

### 4.1 Create Cloud Build Service Account

```bash
# Create service account
gcloud iam service-accounts create cloud-build-sa \
  --display-name="Cloud Build Service Account"

# Grant necessary roles
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member=serviceAccount:cloud-build-sa@${PROJECT_ID}.iam.gserviceaccount.com \
  --role=roles/artifactregistry.writer

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member=serviceAccount:cloud-build-sa@${PROJECT_ID}.iam.gserviceaccount.com \
  --role=roles/container.developer

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member=serviceAccount:cloud-build-sa@${PROJECT_ID}.iam.gserviceaccount.com \
  --role=roles/pubsub.publisher

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member=serviceAccount:cloud-build-sa@${PROJECT_ID}.iam.gserviceaccount.com \
  --role=roles/storage.objectViewer
```

### 4.2 Setup Cloud Build Trigger

```bash
# From GCP Console:
# 1. Go to Cloud Build > Triggers
# 2. Create New Trigger
# 3. Connect GitHub repository
# 4. Set build configuration to cloudbuild.yaml
# 5. Configure substitution variables:
#    - _REGION=us-central1
#    - _ARTIFACT_REPO=fire-detection
#    - _MODEL_GCS_PATH=gs://${PROJECT_ID}-models/fire_detection_model.h5
#    - _GKE_CLUSTER=fire-detection-cluster
#    - _GKE_ZONE=us-central1-a
```

## Step 5: Setup GKE Cluster

### 5.1 Create GKE Cluster

```bash
gcloud container clusters create $GKE_CLUSTER \
  --zone $ZONE \
  --num-nodes 3 \
  --machine-type n1-standard-2 \
  --enable-stackdriver-kubernetes \
  --addons HttpLoadBalancing,HttpsLoadBalancing \
  --enable-autorepair \
  --enable-autoupgrade \
  --enable-autoscaling \
  --min-nodes 3 \
  --max-nodes 10 \
  --release-channel regular
```

### 5.2 Get Cluster Credentials

```bash
gcloud container clusters get-credentials $GKE_CLUSTER --zone $ZONE
```

### 5.3 Create Persistent Volume

```bash
# Create storage class
kubectl apply -f - <<EOF
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: standard
provisioner: pd.csi.storage.gke.io
parameters:
  type: pd-standard
EOF

# Create persistent volume claim
kubectl apply -f mlops/gke/deployment.yaml
```

### 5.4 Deploy Application

```bash
# Update deployment.yaml with your project ID
sed -i "s/PROJECT_ID/$PROJECT_ID/g" mlops/gke/deployment.yaml

# Deploy
kubectl apply -f mlops/gke/deployment.yaml

# Verify deployment
kubectl get deployments
kubectl get pods -l app=fire-detection
kubectl get svc fire-detection-service
```

## Step 6: Setup Cloud Pub/Sub

### 6.1 Create Pub/Sub Topic

```bash
gcloud pubsub topics create fire-detection-builds

gcloud pubsub subscriptions create fire-detection-builds-sub \
  --topic=fire-detection-builds
```

### 6.2 Grant Cloud Build Permissions

```bash
# Get Cloud Build service account
export CLOUD_BUILD_SA=$(gcloud projects describe $PROJECT_ID \
  --format='value(projectNumber)')@cloudbuild.gserviceaccount.com

# Grant publish permission
gcloud pubsub topics add-iam-policy-binding fire-detection-builds \
  --member=serviceAccount:$CLOUD_BUILD_SA \
  --role=roles/pubsub.publisher
```

## Step 7: Setup Cloud Run Function

### 7.1 Deploy Cloud Run Function

```bash
gcloud functions deploy update-gke-deployment \
  --runtime python39 \
  --trigger-topic fire-detection-builds \
  --entry-point update_deployment \
  --source mlops/cloud_functions/ \
  --memory 512MB \
  --timeout 540 \
  --set-env-vars PROJECT_ID=$PROJECT_ID,CLUSTER_NAME=$GKE_CLUSTER,CLUSTER_ZONE=$ZONE
```

### 7.2 Grant Cloud Run Function Permissions

```bash
export CLOUD_RUN_SA=$(gcloud projects describe $PROJECT_ID \
  --format='value(projectNumber)')@cloudbuild.gserviceaccount.com

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member=serviceAccount:$CLOUD_RUN_SA \
  --role=roles/container.developer
```

## Step 8: Test the Pipeline

### 8.1 Local Testing

```bash
# Build Docker image locally
docker build -t fire-detection:local -f mlops/flask_app/Dockerfile .

# Run container
docker run -p 5000:5000 -e MODEL_PATH=/models/fire_detection_model.h5 fire-detection:local
```

### 8.2 Test API Endpoints

```bash
# Health check
curl -X GET http://localhost:5000/health

# Model info
curl -X GET http://localhost:5000/model_info

# Single image prediction
curl -X POST -F "file=@sample_fire.jpg" http://localhost:5000/predict

# Batch prediction
curl -X POST -F "files=@fire1.jpg" -F "files=@fire2.jpg" http://localhost:5000/batch_predict
```

### 8.3 Test GKE Deployment

```bash
# Forward port to local machine
kubectl port-forward svc/fire-detection-service 8080:80

# Test API
curl -X GET http://localhost:8080/health
```

## Step 9: Trigger CI/CD Pipeline

### 9.1 Push Code to GitHub

```bash
git add .
git commit -m "Deploy Mask R-CNN fire detection model"
git push origin main
```

This will automatically trigger:
1. Cloud Build to build Docker image
2. Push image to Artifact Registry
3. Publish message to Pub/Sub
4. Cloud Run Function to update GKE

### 9.2 Monitor Build

```bash
gcloud builds list --limit 5

# View build details
gcloud builds log <BUILD_ID> --stream
```

## Step 10: Monitor Deployment

### 10.1 View Pod Logs

```bash
# Get pod logs
kubectl logs -l app=fire-detection --tail=100 -f

# Check deployment status
kubectl describe deployment fire-detection

# Check HPA status
kubectl get hpa fire-detection-hpa
```

### 10.2 Setup Cloud Logging

```bash
# View application logs
gcloud logging read "resource.type=k8s_container AND resource.labels.pod_name=~'^fire-detection.*'" \
  --limit 50 \
  --format json
```

## Step 11: Production Considerations

### 11.1 Enable Network Policies

```bash
kubectl apply -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: fire-detection-netpol
spec:
  podSelector:
    matchLabels:
      app: fire-detection
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector: {}
  egress:
  - to:
    - namespaceSelector: {}
EOF
```

### 11.2 Setup Ingress

```bash
kubectl apply -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: fire-detection-ingress
  annotations:
    kubernetes.io/ingress.class: gce
spec:
  rules:
  - host: api.example.com
    http:
      paths:
      - path: /*
        pathType: ImplementationSpecific
        backend:
          service:
            name: fire-detection-service
            port:
              number: 80
EOF
```

### 11.3 Setup Monitoring

```bash
# Enable Monitoring API
gcloud services enable monitoring.googleapis.com

# Create uptime check
gcloud monitoring uptime-checks create fire-detection-health \
  --display-name="Fire Detection API Health" \
  --http-check-path=/health
```

## Step 12: Cleanup (When Done)

```bash
# Delete GKE cluster
gcloud container clusters delete $GKE_CLUSTER --zone $ZONE

# Delete Artifact Registry
gcloud artifacts repositories delete $ARTIFACT_REPO --location $REGION

# Delete storage buckets
gsutil -m rm -r gs://${PROJECT_ID}-models/
gsutil -m rm -r gs://${PROJECT_ID}-builds/
gsutil -m rm -r gs://${PROJECT_ID}-logs/

# Delete Cloud Run Functions
gcloud functions delete update-gke-deployment --region $REGION

# Delete Pub/Sub resources
gcloud pubsub subscriptions delete fire-detection-builds-sub
gcloud pubsub topics delete fire-detection-builds

# Delete project (if needed)
gcloud projects delete $PROJECT_ID
```

## Troubleshooting

### Build Failed

```bash
# Check build logs
gcloud builds log <BUILD_ID> --stream

# Check Cloud Build permissions
gcloud projects get-iam-policy $PROJECT_ID \
  --flatten=bindings[].members \
  --filter=bindings.role:roles/artifactregistry.writer
```

### Deployment Not Updating

```bash
# Check Cloud Run Function logs
gcloud functions describe update-gke-deployment --gen2

# Check Pub/Sub messages
gcloud pubsub subscriptions pull fire-detection-builds-sub --auto-ack

# Check GKE events
kubectl describe deployment fire-detection
kubectl get events
```

### Pod Not Starting

```bash
# Check pod details
kubectl describe pod <POD_NAME>

# Check logs
kubectl logs <POD_NAME>

# Check resource availability
kubectl top nodes
kubectl top pods
```

## API Documentation

### Health Check

```http
GET /health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2024-01-29T10:30:00"
}
```

### Single Image Prediction

```http
POST /predict
Content-Type: multipart/form-data

file: <image_file>
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
  "original_dimensions": {
    "height": 480,
    "width": 640
  },
  "status": "success"
}
```

### Batch Prediction

```http
POST /batch_predict
Content-Type: multipart/form-data

files: <image_file_1>
files: <image_file_2>
...
```

Response:
```json
{
  "results": [
    {
      "predicted_class": "Fire",
      "confidence": 0.95,
      "filename": "fire1.jpg",
      "status": "success"
    },
    {
      "predicted_class": "Background",
      "confidence": 0.88,
      "filename": "fire2.jpg",
      "status": "success"
    }
  ],
  "total": 2,
  "timestamp": "2024-01-29T10:30:00"
}
```

### Model Information

```http
GET /model_info
```

Response:
```json
{
  "model_name": "Mask R-CNN Fire Detection",
  "version": "1.0.0",
  "input_shape": "Variable (images)",
  "output_shapes": {
    "classification": "N x 2",
    "bounding_box": "N x 8",
    "segmentation": "N x H x W x 2"
  },
  "supported_formats": ["jpg", "jpeg", "png", "tiff", "tif"],
  "max_file_size": "16MB",
  "status": "loaded"
}
```

## Cost Optimization

1. **Use Committed Use Discounts** for GKE nodes
2. **Enable autoscaling** to scale down during off-peak hours
3. **Use preemptible nodes** for non-critical workloads
4. **Monitor Cloud Build quotas** to avoid unnecessary builds
5. **Archive old logs** to Cloud Storage
6. **Use Cloud Storage for models** instead of container image

## Security Best Practices

1. **Enable Workload Identity** for GKE pods
2. **Use Network Policies** to restrict pod communication
3. **Enable Pod Security Policies**
4. **Scan container images** for vulnerabilities
5. **Use Secret Manager** for sensitive data
6. **Enable Cloud Audit Logs**
7. **Use private GKE clusters** when possible
8. **Implement RBAC** properly

## References

- [GCP Cloud Build Documentation](https://cloud.google.com/build/docs)
- [GCP Artifact Registry Documentation](https://cloud.google.com/artifact-registry/docs)
- [GCP GKE Documentation](https://cloud.google.com/kubernetes-engine/docs)
- [GCP Cloud Run Functions Documentation](https://cloud.google.com/functions/docs)
- [GCP Pub/Sub Documentation](https://cloud.google.com/pubsub/docs)
