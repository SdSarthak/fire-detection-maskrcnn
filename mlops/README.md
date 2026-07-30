# MLOps module

Everything needed to serve the trained model and keep it deployed.

```
mlops/
├── flask_app/
│   ├── app.py            REST API
│   ├── Dockerfile        multi-stage CPU image
│   ├── uwsgi.ini         production server config
│   └── requirements.txt
├── cloudbuild/
│   └── cloudbuild.yaml   build -> push -> announce -> roll out -> verify
├── gke/
│   └── deployment.yaml   Deployment, ConfigMap, Service, PDB, HPA, ServiceAccount
├── cloud_functions/
│   ├── main.py           Pub/Sub -> patch the GKE deployment
│   └── requirements.txt
└── DEPLOYMENT_GUIDE.md   step by step GCP setup
```

## Pipeline

```
git push
   -> Cloud Build trigger
      -> docker build (repo root as context)
      -> push :SHORT_SHA and :latest to Artifact Registry
      -> publish {image_uri, build_id, commit_sha} to Pub/Sub
         -> Cloud Run Function patches deployment/fire-detection
      -> kubectl set image (fallback if the function is down)
      -> kubectl rollout status (fails the build if the rollout stalls)
   -> rolling update, maxUnavailable 0
```

The Pub/Sub path and the direct `kubectl set image` step both set the same
image on the same deployment, so running both is idempotent — whichever lands
first wins and the second is a no-op.

## API

| Method | Path | Notes |
|---|---|---|
| GET | `/health` | Liveness. 200 whenever the process is serving |
| GET | `/ready` | Readiness. 503 until the checkpoint is loaded |
| GET | `/model_info` | Architecture, classes, thresholds, limits |
| GET | `/metrics` | Prometheus text format |
| POST | `/predict` | Form field `file`. `?overlay=true` adds a base64 PNG |
| POST | `/batch_predict` | Form field `files`, capped by `MAX_BATCH_SIZE` |

Splitting liveness from readiness matters here: loading Mask R-CNN takes tens of
seconds, and a single `/health` probe used for both would either restart the pod
mid-load or route traffic to a pod that cannot answer.

```bash
curl http://localhost:5000/health
curl -F "file=@wildfire.jpg" http://localhost:5000/predict
curl -F "files=@a.jpg" -F "files=@b.jpg" http://localhost:5000/batch_predict
```

Uploads are decoded straight from memory with `cv2.imdecode`; nothing is
written to disk, so the container runs with a read-only root filesystem and
there is no temp-file cleanup path to leak.

### Configuration

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_PATH` | `/models/fire_detection_model.pt` | Checkpoint location |
| `SEGMENTATION_ROOT` | auto-detected | Where the `src/` package lives |
| `DEVICE` | `auto` | `auto` / `cpu` / `cuda` |
| `SCORE_THRESHOLD` | `0.5` | Minimum detection confidence |
| `MASK_THRESHOLD` | `0.5` | Soft-mask binarisation cut-off |
| `MAX_CONTENT_LENGTH_MB` | `16` | Request body limit (returns 413) |
| `MAX_BATCH_SIZE` | `16` | Images per `/batch_predict` call |
| `MODEL_VERSION` | `unknown` | Reported by `/model_info` |
| `PORT` | `5000` | Dev server only; uWSGI binds via `uwsgi.ini` |

### Metrics

`/metrics` exposes counters for requests, errors, images processed, fire
instances detected and cumulative inference seconds, plus gauges for uptime and
whether the model is loaded. The pod template carries the matching
`prometheus.io/*` annotations.

## Local Docker

```bash
# from the repository root -- the build context must include segmentation/
docker build -t fire-detection -f mlops/flask_app/Dockerfile .

docker run --rm -p 5000:5000 \
  -v "$PWD/segmentation/weights:/models:ro" \
  -e MODEL_PATH=/models/fire_detection_model.pt \
  fire-detection

curl http://localhost:5000/ready
```

The image installs CPU-only PyTorch wheels, which keeps it roughly 2 GB smaller
than the default CUDA build. For GPU nodes, drop the
`--extra-index-url .../cpu` line from the Dockerfile and request
`nvidia.com/gpu` in the deployment.

## uWSGI

`uwsgi.ini` uses `http-socket`, not `socket`: the Kubernetes probes and the
LoadBalancer speak HTTP, and the uwsgi binary protocol would need an nginx
sidecar to translate it. It also runs with `lazy-apps = true` so each worker
owns its own model copy after fork, and never daemonizes — logs go to stdout
for Cloud Logging.

## Cloud Function

`cloud_functions/main.py` takes its entire target from the environment:
`GCP_PROJECT_ID` (required), `GKE_CLUSTER`, `GKE_ZONE`, `DEPLOYMENT_NAME`,
`CONTAINER_NAME`, `NAMESPACE`. It patches only the container image, letting
Kubernetes perform the rolling update, and returns 4xx for a malformed message
versus 5xx for a failed rollout so Pub/Sub retries only what is worth retrying.

## Cost

| Service | Rough monthly cost |
|---|---|
| GKE cluster, 3 e2-standard-4 nodes | $150-200 |
| Cloud Build | $0.003 per build-minute |
| Artifact Registry | $0.10 per GB stored |
| Cloud Run Functions | Per invocation, effectively free at this volume |
| Cloud Storage (model) | Cents |

An Autopilot cluster or a single-node zonal cluster cuts the largest line item
substantially for a demo deployment.
