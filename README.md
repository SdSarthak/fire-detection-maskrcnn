# Fire Detection Image Segmentation with Mask R-CNN

Complete project for fire detection using Mask R-CNN model with image segmentation. This project includes model training, inference, and MLOps deployment on Google Cloud Platform.

## Project Structure

```
Mask R CNN/
├── segmentation/                 # Model training and inference
│   ├── src/
│   │   ├── config.py            # Configuration management
│   │   ├── dataset.py           # Data loading and preprocessing
│   │   └── model.py             # Model architecture and training
│   ├── data/
│   │   ├── train/               # Training images
│   │   ├── val/                 # Validation images
│   │   ├── test/                # Test images
│   │   └── annotations/         # VGG Annotator JSON files
│   ├── weights/                 # Pre-trained and trained weights
│   ├── outputs/                 # Inference results and visualizations
│   ├── train.py                 # Training script
│   ├── infer.py                 # Inference script
│   ├── requirements.txt          # Python dependencies
│   └── README.md                # Segmentation module documentation
│
└── mlops/                        # MLOps deployment infrastructure
    ├── flask_app/
    │   ├── app.py               # Flask REST API
    │   ├── Dockerfile           # Container image definition
    │   ├── uwsgi.ini            # uWSGI server configuration
    │   └── requirements.txt      # Flask dependencies
    ├── cloudbuild/
    │   └── cloudbuild.yaml      # Cloud Build CI/CD pipeline
    ├── gke/
    │   └── deployment.yaml      # Kubernetes deployment configuration
    ├── cloud_functions/
    │   ├── main.py              # Cloud Run Function for auto-deployment
    │   └── requirements.txt      # Cloud Function dependencies
    ├── DEPLOYMENT_GUIDE.md       # Complete deployment instructions
    └── README.md                # MLOps module documentation
```

## Business Overview

Fire detection is critical for early warning systems in forests, buildings, and industrial facilities. This project demonstrates an automated machine learning pipeline for real-time fire detection using image segmentation with Mask R-CNN.

### Key Features

- **Real-time Fire Detection**: Process images to detect fire locations
- **Image Segmentation**: Use Mask R-CNN for precise fire region identification
- **Automated MLOps**: Complete CI/CD pipeline on Google Cloud Platform
- **Scalable Deployment**: Kubernetes-based containerized application
- **REST API**: Easy integration with external systems
- **Model Versioning**: Automatic model updates via CI/CD

## Technologies Used

### Model Training
- **TensorFlow**: Deep learning framework
- **Keras**: High-level neural network API
- **Mask R-CNN**: Instance segmentation architecture
- **OpenCV**: Image processing

### Deployment
- **Docker**: Container virtualization
- **Google Cloud Build**: CI/CD automation
- **Artifact Registry**: Container image registry
- **Google Kubernetes Engine (GKE)**: Container orchestration
- **Cloud Run Functions**: Serverless compute
- **Cloud Pub/Sub**: Event messaging
- **Flask**: Web framework
- **uWSGI**: Application server

## Project Phases

### Phase 1: Model Development (segmentation/)

This phase covers building and training the fire detection model:

1. **Dependencies Installation**: Install required Python packages
2. **Data Preparation**: Load and preprocess training data
3. **Annotation Creation**: Create masks using VGG Annotator
4. **Model Building**: Build Mask R-CNN architecture
5. **Model Training**: Train on annotated fire images
6. **Model Evaluation**: Validate on test data
7. **Inference**: Run predictions on new images

**Quick Start for Training:**

```bash
cd segmentation
pip install -r requirements.txt
python train.py
```

**Quick Start for Inference:**

```bash
cd segmentation
python infer.py
```

### Phase 2: MLOps Deployment (mlops/)

This phase automates deployment on Google Cloud Platform:

1. **Local Development**: Test application locally
2. **Containerization**: Create Docker image
3. **Cloud Build**: Automated build pipeline
4. **Image Registry**: Store images in Artifact Registry
5. **Kubernetes Deployment**: Deploy to GKE
6. **Auto-Scaling**: Handle traffic with HPA
7. **Monitoring**: Track model performance

**Deployment Steps:**

```bash
# See DEPLOYMENT_GUIDE.md for detailed instructions
# Quick overview:

# 1. Setup GCP
gcloud projects create fire-detection-mlops
gcloud services enable cloudbuild.googleapis.com artifactregistry.googleapis.com

# 2. Create Artifact Registry
gcloud artifacts repositories create fire-detection \
  --repository-format=docker --location=us-central1

# 3. Create GKE Cluster
gcloud container clusters create fire-detection-cluster \
  --zone us-central1-a --num-nodes 3

# 4. Deploy Application
kubectl apply -f mlops/gke/deployment.yaml

# 5. Test API
curl http://your-service-ip/health
```

## Data Preparation

### Image Requirements
- Format: JPG, PNG, TIFF
- Size: Any resolution (auto-resized to 256x256)
- Content: Images containing fire scenes

### Annotation Process

1. **Use VGG Annotator**: https://www.robots.ox.ac.uk/~vgg/software/via/
2. **Create Masks**: Draw polygons around fire regions
3. **Export JSON**: Save annotations as `via_project.json`
4. **Organize Data**:
   - Place images in `segmentation/data/train/`, `val/`, `test/`
   - Place annotations in `segmentation/data/annotations/`

### Data Structure
```
data/
├── train/
│   ├── fire_image_1.jpg
│   ├── fire_image_2.png
│   └── ...
├── val/
│   ├── fire_val_1.jpg
│   └── ...
├── test/
│   └── test_image.jpg
└── annotations/
    ├── train_annotations.json
    ├── val_annotations.json
    └── via_project.json
```

## Model Architecture

### Mask R-CNN Components

1. **Backbone**: ResNet50 for feature extraction
2. **Region Proposal Network (RPN)**: Generates region proposals
3. **ROI Align**: Extracts fixed-size features from proposals
4. **Detection Head**: Classification and bounding box regression
5. **Mask Head**: Generates segmentation masks

### Configuration

Model training is controlled by `src/config.py`:

```python
class FireDetectionConfig:
    NUM_CLASSES = 2  # Background + Fire
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 512
    TRAIN_EPOCHS = 50
    LEARNING_RATE = 0.001
    DETECTION_MIN_CONFIDENCE = 0.7
```

## API Endpoints

### Health Check
```http
GET /health
```

### Model Information
```http
GET /model_info
```

### Single Image Prediction
```http
POST /predict
Content-Type: multipart/form-data
Body: file=<image>
```

### Batch Prediction
```http
POST /batch_predict
Content-Type: multipart/form-data
Body: files=<image1>&files=<image2>&...
```

## Performance Metrics

The model is evaluated on:
- **Accuracy**: Classification accuracy for fire/no-fire
- **Precision**: True positive rate of fire detections
- **Recall**: Detection of all actual fires
- **F1 Score**: Harmonic mean of precision and recall
- **mAP (mean Average Precision)**: Detection quality
- **IoU (Intersection over Union)**: Segmentation quality

## Deployment Architecture

```
Developer/ML Engineer
        ↓
   GitHub Commit
        ↓
  Cloud Build (Trigger)
        ↓
  Build Docker Image → Artifact Registry
        ↓
 Pub/Sub Message (Image Ready)
        ↓
 Cloud Run Function (Listen)
        ↓
  Update GKE Deployment
        ↓
  Rolling Update → New Pods Start
        ↓
 Load Balancer Routes Traffic
        ↓
    Flask API (uWSGI)
        ↓
     End Users
```

## Continuous Integration/Deployment

### Build Process
1. Code pushed to GitHub
2. Cloud Build automatically triggered
3. Dependencies installed
4. Docker image built
5. Image pushed to Artifact Registry
6. Pub/Sub message published
7. Cloud Run Function receives notification
8. GKE deployment updated with new image
9. Rolling update ensures zero downtime

### Monitoring
- Cloud Build logs: `gcloud builds log <ID>`
- GKE pod logs: `kubectl logs -f deployment/fire-detection`
- Cloud Logging: View all application logs

## Cost Estimation

Using GCP services incurs costs. Estimated monthly costs:

| Service | Estimated Cost |
|---------|----------------|
| GKE Cluster (3 nodes) | $150-200 |
| Cloud Build | $0.003 per build minute |
| Artifact Registry | $0.10 per GB/month storage |
| Cloud Run Functions | Pay per invocation |
| **Total Estimate** | **$200-300/month** |

Use [GCP Cost Calculator](https://cloud.google.com/products/calculator) for accurate estimates.

## Security Considerations

1. **Image Scanning**: Scan Docker images for vulnerabilities
2. **Network Policies**: Restrict pod-to-pod communication
3. **RBAC**: Implement role-based access control
4. **Secrets Management**: Use GCP Secret Manager
5. **Audit Logging**: Enable Cloud Audit Logs
6. **SSL/TLS**: Use HTTPS for API communication
7. **Pod Security**: Run containers as non-root

## Troubleshooting

### Training Issues
- Ensure annotation files are in correct JSON format
- Check image file formats are supported
- Verify sufficient GPU memory (if using GPU)

### Deployment Issues
- Check Cloud Build logs for build failures
- Verify GKE cluster is running: `gcloud container clusters list`
- Check pod logs: `kubectl logs <pod-name>`
- Verify service is accessible: `kubectl get svc`

### Model Issues
- Ensure model weights file is uploaded to GCS
- Check MODEL_PATH environment variable in Kubernetes
- Verify model is not corrupted: `file model.h5`

## Contributing

To contribute improvements:

1. Create a feature branch
2. Make changes and test locally
3. Commit with descriptive messages
4. Push to GitHub to trigger CI/CD
5. Monitor deployment in Cloud Build

## References

- [Mask R-CNN Paper](https://arxiv.org/abs/1703.06870)
- [Matterport Mask R-CNN Implementation](https://github.com/matterport/Mask_RCNN)
- [GCP Documentation](https://cloud.google.com/docs)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)

## Next Steps

1. **Prepare Data**: Gather and annotate fire detection images
2. **Train Model**: Run training script with your data
3. **Test Locally**: Use inference script for validation
4. **Setup GCP**: Follow deployment guide
5. **Deploy**: Push to GitHub to trigger CI/CD
6. **Monitor**: Watch Cloud Build and GKE metrics
7. **Optimize**: Tune hyperparameters based on metrics

## License

This project is provided as-is for educational and research purposes.

## Support

For issues and questions:
- Check troubleshooting section above
- Review GCP documentation
- Check GitHub issues
- Contact project maintainer

---

**Last Updated**: January 29, 2024
**Version**: 1.0.0
