# Deployment guide

Step-by-step setup for running the fire detection API on Google Cloud with an
automated build-and-rollout pipeline.

Prerequisites: `gcloud`, `kubectl` and `docker` installed, a billing-enabled
GCP project, and a trained checkpoint from `segmentation/train.py`.

Set your values once and reuse them:

```bash
export PROJECT_ID="your-gcp-project-id"
export REGION="us-central1"
export ZONE="us-central1-a"
export CLUSTER="fire-detection-cluster"
export REPO="fire-detection"
export TOPIC="fire-detection-image-ready"
export MODEL_BUCKET="${PROJECT_ID}-fire-detection-models"

gcloud config set project "$PROJECT_ID"
```

## 1. Enable the APIs

```bash
gcloud services enable \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  container.googleapis.com \
  run.googleapis.com \
  cloudfunctions.googleapis.com \
  pubsub.googleapis.com \
  storage.googleapis.com \
  eventarc.googleapis.com
```

## 2. Upload the trained model

```bash
gcloud storage buckets create "gs://${MODEL_BUCKET}" --location="$REGION"
gcloud storage cp segmentation/weights/fire_detection_model.pt \
  "gs://${MODEL_BUCKET}/fire_detection_model.pt"
```

The pod's init container fetches this object at startup, so publishing a new
model is a `gcloud storage cp` followed by a `kubectl rollout restart` — no
image rebuild needed.

## 3. Create the Artifact Registry repository

```bash
gcloud artifacts repositories create "$REPO" \
  --repository-format=docker \
  --location="$REGION" \
  --description="Fire detection container images"

gcloud auth configure-docker "${REGION}-docker.pkg.dev"
```

## 4. Create the GKE cluster

```bash
gcloud container clusters create "$CLUSTER" \
  --zone="$ZONE" \
  --num-nodes=3 \
  --machine-type=e2-standard-4 \
  --workload-pool="${PROJECT_ID}.svc.id.goog" \
  --enable-autoscaling --min-nodes=1 --max-nodes=6 \
  --enable-ip-alias

gcloud container clusters get-credentials "$CLUSTER" --zone="$ZONE"
```

Workload Identity is what lets the init container read from Cloud Storage
without a service account key file on disk.

## 5. Bind the Kubernetes service account to a Google service account

```bash
gcloud iam service-accounts create fire-detection-sa \
  --display-name="Fire detection workload"

gcloud storage buckets add-iam-policy-binding "gs://${MODEL_BUCKET}" \
  --member="serviceAccount:fire-detection-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role=roles/storage.objectViewer

# Create the Kubernetes objects first so the KSA exists.
sed "s/PROJECT_ID/${PROJECT_ID}/g" mlops/gke/deployment.yaml | kubectl apply -f -

kubectl annotate serviceaccount fire-detection-sa \
  --namespace default \
  iam.gke.io/gcp-service-account="fire-detection-sa@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud iam service-accounts add-iam-policy-binding \
  "fire-detection-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role=roles/iam.workloadIdentityUser \
  --member="serviceAccount:${PROJECT_ID}.svc.id.goog[default/fire-detection-sa]"
```

Point the ConfigMap at your bucket:

```bash
kubectl create configmap fire-detection-config \
  --from-literal="model_gcs_uri=gs://${MODEL_BUCKET}/fire_detection_model.pt" \
  --dry-run=client -o yaml | kubectl apply -f -
```

## 6. Build and push the first image manually

```bash
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/fire-detection"

docker build -t "${IMAGE}:bootstrap" -f mlops/flask_app/Dockerfile .
docker push "${IMAGE}:bootstrap"

kubectl set image deployment/fire-detection "fire-detection=${IMAGE}:bootstrap"
kubectl rollout status deployment/fire-detection --timeout=10m
```

## 7. Verify

```bash
kubectl get pods -l app=fire-detection
kubectl logs -l app=fire-detection --tail=50

EXTERNAL_IP=$(kubectl get svc fire-detection-service \
  -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

curl "http://${EXTERNAL_IP}/health"
curl "http://${EXTERNAL_IP}/ready"
curl "http://${EXTERNAL_IP}/model_info"
curl -F "file=@wildfire.jpg" "http://${EXTERNAL_IP}/predict"
```

`/ready` returning 503 with `"error": "Model checkpoint not found..."` means the
init container did not place the file — check `kubectl logs <pod> -c fetch-model`.

## 8. Wire up the Pub/Sub rollout function

```bash
gcloud pubsub topics create "$TOPIC"

gcloud functions deploy fire-detection-rollout \
  --gen2 \
  --runtime=python311 \
  --region="$REGION" \
  --source=mlops/cloud_functions \
  --entry-point=update_deployment \
  --trigger-topic="$TOPIC" \
  --set-env-vars="GCP_PROJECT_ID=${PROJECT_ID},GKE_CLUSTER=${CLUSTER},GKE_ZONE=${ZONE},DEPLOYMENT_NAME=fire-detection,CONTAINER_NAME=fire-detection,NAMESPACE=default"
```

Grant the function's runtime service account `roles/container.developer` so it
can patch deployments:

```bash
FUNCTION_SA=$(gcloud functions describe fire-detection-rollout --region="$REGION" \
  --gen2 --format='value(serviceConfig.serviceAccountEmail)')

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${FUNCTION_SA}" \
  --role=roles/container.developer
```

Test it without a build:

```bash
gcloud pubsub topics publish "$TOPIC" \
  --message="{\"image_uri\":\"${IMAGE}:bootstrap\",\"build_id\":\"manual\"}"

gcloud functions logs read fire-detection-rollout --region="$REGION" --limit=20
```

## 9. Connect Cloud Build to GitHub

```bash
gcloud builds triggers create github \
  --name=fire-detection-main \
  --repo-owner=SdSarthak \
  --repo-name=fire-detection-maskrcnn \
  --branch-pattern='^main$' \
  --build-config=mlops/cloudbuild/cloudbuild.yaml \
  --substitutions="_REGION=${REGION},_ARTIFACT_REPO=${REPO},_GKE_CLUSTER=${CLUSTER},_GKE_ZONE=${ZONE},_PUBSUB_TOPIC=${TOPIC}"
```

The Cloud Build service account needs three roles:

```bash
BUILD_SA="$(gcloud projects describe "$PROJECT_ID" \
  --format='value(projectNumber)')@cloudbuild.gserviceaccount.com"

for ROLE in roles/container.developer roles/artifactregistry.writer roles/pubsub.publisher; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${BUILD_SA}" --role="$ROLE"
done
```

Push to `main` and watch it run:

```bash
gcloud builds list --limit=5
gcloud builds log "$(gcloud builds list --limit=1 --format='value(id)')"
```

## Operations

**Roll out a retrained model** (no image rebuild):

```bash
gcloud storage cp segmentation/weights/fire_detection_model.pt \
  "gs://${MODEL_BUCKET}/fire_detection_model.pt"
kubectl rollout restart deployment/fire-detection
kubectl rollout status deployment/fire-detection
```

**Roll back a bad deploy:**

```bash
kubectl rollout undo deployment/fire-detection
```

**Scale:**

```bash
kubectl get hpa fire-detection-hpa            # autoscales on CPU 70% / memory 80%
kubectl scale deployment/fire-detection --replicas=5
```

**Watch:**

```bash
kubectl top pods -l app=fire-detection
kubectl logs -f deployment/fire-detection
curl "http://${EXTERNAL_IP}/metrics"
```

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Pod stuck `Init:0/1` | `fetch-model` cannot read the bucket. Check Workload Identity annotation and the ConfigMap URI |
| `/ready` stays 503 | Checkpoint missing or built by an incompatible torchvision. Check `kubectl logs <pod>` for the load error |
| Probes fail but the app logs look fine | `uwsgi.ini` must use `http-socket`, not `socket` |
| `OOMKilled` | Raise the memory limit above 4Gi, or lower `processes` in `uwsgi.ini` |
| Build fails on `COPY segmentation/` | Build from the repository root, not from `mlops/flask_app/` |
| Function returns 400 | The Pub/Sub message has no `image_uri` field |
| Function returns 500 | `GCP_PROJECT_ID` unset, or the runtime SA lacks `roles/container.developer` |

## Security checklist

- Containers run as UID 1000, non-root, with a read-only root filesystem and
  all Linux capabilities dropped.
- No credentials in the image or the repository — Workload Identity everywhere.
- `MAX_CONTENT_LENGTH_MB` and `MAX_BATCH_SIZE` bound request cost.
- Put an HTTPS load balancer or Ingress with a managed certificate in front of
  the Service before exposing it publicly; the plain LoadBalancer above is
  HTTP-only and is meant for testing.
- Enable Artifact Registry vulnerability scanning and Cloud Audit Logs.

## Tear down

```bash
gcloud container clusters delete "$CLUSTER" --zone="$ZONE"
gcloud functions delete fire-detection-rollout --region="$REGION" --gen2
gcloud pubsub topics delete "$TOPIC"
gcloud artifacts repositories delete "$REPO" --location="$REGION"
gcloud storage rm -r "gs://${MODEL_BUCKET}"
```
