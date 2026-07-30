# Project Setup and Quick Start Guide

Complete guide to get started with the Mask R-CNN Fire Detection project.

## System Requirements

### Minimum Requirements
- **OS**: Windows 10+, macOS 10.15+, or Linux
- **Python**: 3.9 or higher
- **RAM**: 8GB minimum (16GB recommended)
- **GPU**: Optional but recommended for training (NVIDIA with CUDA support)
- **Storage**: 20GB for models and datasets

### For GCP Deployment
- Google Cloud Platform account with billing enabled
- `gcloud` CLI installed
- `kubectl` CLI installed
- Docker installed

## Installation

### 1. Clone or Download Project

```bash
# Navigate to your workspace
cd "c:\Users\sarth\OneDrive\Desktop\Projects\Mask R CNN"
```

### 2. Run Setup Script

**On Windows:**
```bash
setup.bat
```

**On macOS/Linux:**
```bash
chmod +x setup.sh
./setup.sh
```

### 3. Manual Setup (If Script Fails)

```bash
# Navigate to segmentation folder
cd segmentation

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create directories
mkdir -p data/train data/val data/test data/annotations
mkdir -p weights outputs

# Navigate back
cd ..
```

## Project Structure Overview

```
Mask R CNN/
│
├── segmentation/               # Phase 1: Model Development
│   ├── src/
│   │   ├── config.py          # Model configuration
│   │   ├── dataset.py         # Data loading
│   │   └── model.py           # Model architecture
│   ├── data/
│   │   ├── train/             # Training images (place here)
│   │   ├── val/               # Validation images (place here)
│   │   ├── test/              # Test images (place here)
│   │   └── annotations/       # VGG annotations (place here)
│   ├── weights/               # Trained models (generated)
│   ├── outputs/               # Inference results (generated)
│   ├── train.py               # Training script
│   ├── infer.py               # Inference script
│   ├── requirements.txt        # Python dependencies
│   └── README.md              # Detailed documentation
│
├── mlops/                      # Phase 2: MLOps Deployment
│   ├── flask_app/
│   │   ├── app.py             # REST API
│   │   ├── Dockerfile         # Container definition
│   │   └── uwsgi.ini          # Server config
│   ├── cloudbuild/
│   │   └── cloudbuild.yaml    # CI/CD pipeline
│   ├── gke/
│   │   └── deployment.yaml    # Kubernetes config
│   ├── cloud_functions/
│   │   └── main.py            # Auto-deployment function
│   ├── DEPLOYMENT_GUIDE.md    # GCP setup guide
│   └── README.md              # Detailed documentation
│
├── README.md                   # Project overview
├── setup.bat                   # Windows setup
├── setup.sh                    # Linux/macOS setup
├── .gitignore                  # Git ignore file
└── QUICKSTART.md               # This file
```

## Phase 1: Model Development

### Preparing Training Data

1. **Collect Fire Images**
   - Gather 20+ fire detection images
   - Supported formats: JPG, PNG, TIFF
   - Organize images:
     ```
     segmentation/data/train/     (20 images)
     segmentation/data/val/       (10 images)
     segmentation/data/test/      (5+ images)
     ```

2. **Create Annotations**
   - Go to [VGG Annotator](https://www.robots.ox.ac.uk/~vgg/software/via/via_demo.html)
   - Load your images
   - Draw polygons around fire regions
   - Mark regions with attribute `object: fire`
   - Export as JSON file
   - Save as:
     ```
     segmentation/data/annotations/train_annotations.json
     segmentation/data/annotations/val_annotations.json
     ```

### Training the Model

```bash
# Navigate to segmentation folder
cd segmentation

# Activate virtual environment (if not already activated)
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Run training
python train.py
```

**Training Output:**
```
============================================================
Mask R-CNN Fire Detection - Training Pipeline
============================================================

Step 1: Download Pre-trained Weights
  ✓ Weights configuration initialized

Step 2: Load Dataset
  ✓ Training samples: 20
  ✓ Validation samples: 10

Step 3: Build Model
  ✓ Model architecture created
  ✓ Model compiled

Step 4: Train Model
  Starting training...
  Epochs: 50
  
  [Training progress...]
  
Step 5: Save Model
  ✓ Model saved to weights/fire_detection_model.h5

============================================================
Training Complete!
============================================================
```

**Training Tips:**
- First training will download pre-trained weights (~500MB)
- Training time: 1-4 hours depending on GPU
- Monitor GPU memory: `nvidia-smi`
- Reduce `IMAGE_MAX_DIM` in config.py if out of memory

### Running Inference

```bash
# Run inference on test images
python infer.py
```

**Inference Output:**
```
============================================================
Mask R-CNN Fire Detection - Inference Pipeline
============================================================

Step 1: Load Model
  ✓ Model loaded from weights/fire_detection_model.h5

Step 2: Load Test Data
  ✓ Found 5 test images

Step 3: Run Inference
  Processing 1/5: fire_test_1.jpg
    ✓ Confidence: 0.9523
    ✓ Predicted: Fire
    ✓ Saved visualization to outputs/prediction_fire_test_1.jpg

  [Additional results...]

============================================================
Inference Complete!
============================================================
```

### Evaluating Results

```bash
# Results saved in:
segmentation/outputs/

# View predictions:
# - prediction_<image_name>.png files
# - Visualization shows original, predicted class, and mask
```

## Phase 2: MLOps Deployment

### Local Testing with Docker

```bash
# Build Docker image
docker build -t fire-detection:local -f mlops/flask_app/Dockerfile .

# Run container locally
docker run -p 5000:5000 \
  -e MODEL_PATH=/models/fire_detection_model.h5 \
  --mount type=bind,source="$(pwd)/segmentation/weights",target=/models \
  fire-detection:local
```

### Testing API

```bash
# Health check
curl http://localhost:5000/health

# Model info
curl http://localhost:5000/model_info

# Single image prediction
curl -X POST -F "file=@test_image.jpg" \
  http://localhost:5000/predict

# Batch prediction
curl -X POST -F "files=@image1.jpg" -F "files=@image2.jpg" \
  http://localhost:5000/batch_predict
```

### GCP Deployment

#### Step 1: Setup GCP Project

```bash
# Create project
gcloud projects create fire-detection-mlops --set-as-default
export PROJECT_ID=$(gcloud config get-value project)

# Enable required APIs
gcloud services enable \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  container.googleapis.com \
  run.googleapis.com \
  pubsub.googleapis.com
```

#### Step 2: Create Artifact Registry

```bash
gcloud artifacts repositories create fire-detection \
  --repository-format=docker \
  --location=us-central1

gcloud auth configure-docker us-central1-docker.pkg.dev
```

#### Step 3: Create GKE Cluster

```bash
gcloud container clusters create fire-detection-cluster \
  --zone us-central1-a \
  --num-nodes 3 \
  --machine-type n1-standard-2

gcloud container clusters get-credentials fire-detection-cluster \
  --zone us-central1-a
```

#### Step 4: Deploy to Kubernetes

```bash
# Update project ID in deployment.yaml
sed -i "s/PROJECT_ID/$PROJECT_ID/g" mlops/gke/deployment.yaml

# Deploy
kubectl apply -f mlops/gke/deployment.yaml

# Verify deployment
kubectl get deployments
kubectl get pods
kubectl get svc fire-detection-service
```

#### Step 5: Setup CI/CD (Cloud Build)

```bash
# Create Pub/Sub topic
gcloud pubsub topics create fire-detection-builds

# Create subscription
gcloud pubsub subscriptions create fire-detection-builds-sub \
  --topic=fire-detection-builds

# Deploy Cloud Function for auto-update
gcloud functions deploy update-gke-deployment \
  --runtime python39 \
  --trigger-topic fire-detection-builds \
  --entry-point update_deployment \
  --source mlops/cloud_functions/
```

#### Step 6: Enable CI/CD Trigger

From GCP Console:
1. Go to **Cloud Build > Triggers**
2. Click **Create Trigger**
3. Connect to GitHub repository
4. Set build configuration to `cloudbuild.yaml`
5. Configure substitution variables:
   ```
   _REGION=us-central1
   _ARTIFACT_REPO=fire-detection
   _GKE_CLUSTER=fire-detection-cluster
   _GKE_ZONE=us-central1-a
   ```

### Monitoring Deployment

```bash
# View Cloud Build logs
gcloud builds list --limit 5
gcloud builds log <BUILD_ID> --stream

# View GKE logs
kubectl logs -f deployment/fire-detection

# Check pod status
kubectl describe pods

# View service IP
kubectl get svc fire-detection-service

# Test deployed API
kubectl port-forward svc/fire-detection-service 8080:80
curl http://localhost:8080/health
```

## API Usage Examples

### Using Python Requests

```python
import requests

# Health check
response = requests.get('http://localhost:5000/health')
print(response.json())

# Single image prediction
with open('fire_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/predict',
        files={'file': f}
    )
    result = response.json()
    print(f"Fire detected: {result['is_fire']}")
    print(f"Confidence: {result['confidence']:.2%}")

# Batch prediction
files = [
    ('files', open('fire1.jpg', 'rb')),
    ('files', open('fire2.jpg', 'rb')),
    ('files', open('fire3.jpg', 'rb'))
]
response = requests.post('http://localhost:5000/batch_predict', files=files)
results = response.json()
for result in results['results']:
    print(f"{result['filename']}: {result['predicted_class']}")
```

### Using cURL

```bash
# Single prediction
curl -X POST \
  -F "file=@fire_image.jpg" \
  http://localhost:5000/predict

# Batch prediction
curl -X POST \
  -F "files=@fire1.jpg" \
  -F "files=@fire2.jpg" \
  http://localhost:5000/batch_predict
```

## Troubleshooting

### Training Issues

**Problem**: Out of memory error
```
Solution: Reduce IMAGE_MAX_DIM in segmentation/src/config.py
```

**Problem**: Annotation file not found
```
Solution: Ensure annotations.json is in segmentation/data/annotations/
         Check filename in load_images_from_dir() call
```

**Problem**: Model not converging
```
Solution: Increase TRAIN_EPOCHS in config.py
         Try lower LEARNING_RATE (e.g., 0.0001)
         Verify annotation quality
```

### Docker Issues

**Problem**: Build fails
```bash
# Debug
docker build --no-cache -f mlops/flask_app/Dockerfile .
# Check for typos in Dockerfile
# Verify all files exist
```

**Problem**: Container won't start
```bash
# Check logs
docker logs <container_id>
# Verify MODEL_PATH environment variable
# Check volume mounts exist
```

### Kubernetes Issues

**Problem**: Pods not starting
```bash
# Check pod status
kubectl describe pod <pod_name>
# View logs
kubectl logs <pod_name>
# Check resource availability
kubectl top nodes
```

**Problem**: Service not accessible
```bash
# Verify service
kubectl get svc
# Check load balancer IP
kubectl get svc fire-detection-service
# Port forward for testing
kubectl port-forward svc/fire-detection-service 8080:80
```

## Common Tasks

### Update Model on GCP

```bash
# 1. Train new model locally
cd segmentation
python train.py

# 2. Upload to GCS
gsutil cp segmentation/weights/fire_detection_model.h5 \
  gs://$PROJECT_ID-models/

# 3. Trigger build (push to GitHub)
git add .
git commit -m "Update model"
git push origin main

# 4. Monitor deployment
gcloud builds list --limit 1
kubectl get pods -w
```

### Scale Up/Down

```bash
# Scale manually
kubectl scale deployment fire-detection --replicas=5

# View current replicas
kubectl get hpa fire-detection-hpa

# Edit HPA settings
kubectl edit hpa fire-detection-hpa
```

### View Logs

```bash
# Cloud Build
gcloud builds log <BUILD_ID> --stream

# GKE Pods
kubectl logs -f pod/<POD_NAME>

# All pods in deployment
kubectl logs -l app=fire-detection --tail=100 -f

# Cloud Logging
gcloud logging read "resource.type=k8s_container" --limit 50
```

### Cleanup Resources

```bash
# Delete Kubernetes deployment
kubectl delete -f mlops/gke/deployment.yaml

# Delete GKE cluster
gcloud container clusters delete fire-detection-cluster --zone us-central1-a

# Delete Artifact Registry
gcloud artifacts repositories delete fire-detection --location=us-central1

# Delete GCS buckets
gsutil -m rm -r gs://$PROJECT_ID-models/
gsutil -m rm -r gs://$PROJECT_ID-builds/

# Delete Cloud Function
gcloud functions delete update-gke-deployment --region us-central1

# Delete Pub/Sub resources
gcloud pubsub subscriptions delete fire-detection-builds-sub
gcloud pubsub topics delete fire-detection-builds
```

## Performance Benchmarks

### Expected Performance

| Metric | Value |
|--------|-------|
| Model Training Time | 1-4 hours (GPU) |
| Inference Time | 50-200ms per image |
| Model Size | 100-150MB |
| RAM Usage | 2-4GB |
| GPU Memory | 4-6GB |

### Optimization Tips

1. **Training**
   - Use GPU: Reduces training time by 10x
   - Increase batch size: Better GPU utilization
   - Use mixed precision: 2x faster with minimal accuracy loss

2. **Inference**
   - Use batch prediction: Process multiple images
   - Enable caching: Reduce model loading time
   - Use TensorRT: 2-3x inference speedup

3. **Deployment**
   - Use preemptible instances: 70% cost reduction
   - Set resource limits: Better pod distribution
   - Enable autoscaling: Handle traffic spikes

## Next Steps

1. ✅ **Setup Complete**: Project structure is ready
2. 📊 **Prepare Data**: Gather and annotate fire images
3. 🧠 **Train Model**: Run `python train.py` in segmentation/
4. 🔍 **Validate**: Run `python infer.py` to test predictions
5. 🐳 **Containerize**: Build Docker image for deployment
6. ☁️ **Deploy**: Follow GCP deployment guide
7. 📈 **Monitor**: Track model performance in production
8. 🚀 **Scale**: Optimize based on usage patterns

## Documentation Files

- **README.md** - Project overview and architecture
- **segmentation/README.md** - Model development guide
- **mlops/README.md** - Deployment infrastructure guide
- **mlops/DEPLOYMENT_GUIDE.md** - Complete GCP setup steps
- **QUICKSTART.md** - This file

## Support Resources

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [GCP Documentation](https://cloud.google.com/docs)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Docker Documentation](https://docs.docker.com/)
- [Flask Documentation](https://flask.palletsprojects.com/)

## Cost Estimates

| Service | Monthly Cost |
|---------|--------------|
| GKE (3 nodes) | $150-200 |
| Cloud Build | $0-30 (based on usage) |
| Artifact Registry | $10-50 |
| Cloud Run Function | $0-20 |
| **Total** | **$160-300** |

Use [GCP Cost Calculator](https://cloud.google.com/products/calculator) for accurate estimates.

---

**Version**: 1.0.0
**Last Updated**: January 29, 2024

For detailed information, refer to the README.md and module-specific documentation files.
