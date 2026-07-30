# Fire Detection with Mask R-CNN

Instance segmentation of fire regions in images, plus the MLOps pipeline that
serves the model on Google Cloud.

The model is a real Mask R-CNN — ResNet50 + FPN backbone, region proposal
network, ROI Align, box head and mask head — built on `torchvision` and
fine-tuned from COCO weights. It outputs one mask, box and confidence score per
fire instance, not a single image-level label.

```
Mask R CNN/
├── segmentation/            model training and inference
│   ├── src/
│   │   ├── config.py        dataclass config, overridable via FIRE_* env vars
│   │   ├── dataset.py       VIA annotation parsing -> torch Dataset
│   │   ├── model.py         model construction, training loop, evaluation
│   │   ├── metrics.py       IoU / Dice / precision / recall / AP
│   │   └── predictor.py     inference wrapper shared by the CLI and the API
│   ├── train.py             training entrypoint
│   ├── infer.py             inference entrypoint
│   ├── data/                train/ val/ test/ annotations/ (git-ignored)
│   ├── weights/             checkpoints (git-ignored)
│   └── outputs/             overlays and predictions.json (git-ignored)
├── mlops/
│   ├── flask_app/           REST API, Dockerfile, uWSGI config
│   ├── cloudbuild/          Cloud Build CI/CD pipeline
│   ├── gke/                 Kubernetes manifests
│   └── cloud_functions/     Pub/Sub -> GKE rollout function
└── tests/                   pytest suite (118 tests)
```

## Quick start

```bash
git clone https://github.com/SdSarthak/fire-detection-maskrcnn.git
cd fire-detection-maskrcnn

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r segmentation/requirements.txt
pip install -r requirements-dev.txt   # only needed to run the tests

cp .env.example .env              # then edit
```

`setup.sh` (or `setup.bat` on Windows) does the same thing in one step.

For a GPU build of PyTorch, follow the selector at
<https://pytorch.org/get-started/locally/> instead of the pinned CPU wheels.

## 1. Prepare data

1. Put images in `segmentation/data/train/`, `val/` and `test/`.
2. Annotate the fire regions with the
   [VGG Image Annotator](https://www.robots.ox.ac.uk/~vgg/software/via/).
   Draw a polygon per fire region and set a region attribute to `fire`.
3. Export the project as JSON to
   `segmentation/data/annotations/train_annotations.json` (and
   `val_annotations.json`).

Both VIA export layouts are supported (flat, and the newer
`_via_img_metadata` wrapper), as are polygon, polyline, rect, circle and
ellipse shapes — non-polygon shapes are converted automatically.

Images without a usable annotation are skipped and reported, so a typo in a
filename never silently trains on an empty mask.

## 2. Train

```bash
cd segmentation
python train.py --epochs 30
```

Useful flags:

| Flag | Meaning |
|---|---|
| `--data-dir` / `--annotations-dir` | Point at a dataset outside the repo |
| `--epochs`, `--batch-size`, `--lr` | Override the schedule |
| `--pretrained coco\|imagenet\|none` | Starting weights (default `coco`) |
| `--device auto\|cpu\|cuda` | Force a device |
| `--class-filter fire,flame` | Which region attributes count as fire |
| `--output` | Checkpoint path |

Each epoch reports the four Mask R-CNN losses plus validation loss, precision,
recall, F1, mean IoU and AP@0.5. The checkpoint with the best validation F1 is
kept, and the full history is written to `outputs/training_history.json`.

A checkpoint stores its own configuration, so inference rebuilds the exact
architecture it was trained with — there is no way to load weights into a
mismatched model.

## 3. Run inference

```bash
python infer.py --input data/test --score-threshold 0.5
```

Writes an annotated overlay per image to `outputs/` and an aggregated
`outputs/predictions.json`:

```json
{
  "filename": "wildfire_03.jpg",
  "is_fire": true,
  "confidence": 0.94,
  "num_detections": 2,
  "fire_area_ratio": 0.187,
  "detections": [
    {
      "class_name": "fire",
      "score": 0.94,
      "bounding_box": {"x1": 210.4, "y1": 96.1, "x2": 512.0, "y2": 388.7},
      "mask_area_px": 41230,
      "polygons": [[[212, 98], [508, 104], "..."]]
    }
  ]
}
```

Masks are returned as simplified polygons rather than bitmaps, so the payload
stays small enough to log or forward to another service.

## 4. Serve the API

```bash
export MODEL_PATH=segmentation/weights/fire_detection_model.pt
python mlops/flask_app/app.py
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness. Always 200 while the process is up |
| `GET /ready` | Readiness. 503 until the model is loaded |
| `GET /model_info` | Architecture, classes, thresholds, limits |
| `GET /metrics` | Prometheus counters and gauges |
| `POST /predict` | One image, form field `file` |
| `POST /batch_predict` | Several images, form field `files` |

```bash
curl -F "file=@wildfire.jpg" http://localhost:5000/predict
curl -F "file=@wildfire.jpg" "http://localhost:5000/predict?overlay=true"   # + base64 PNG
```

The API never writes uploads to disk: bytes are decoded straight into memory,
which removes the temp-file cleanup path entirely and keeps the container's
root filesystem read-only.

## 5. Deploy

```bash
docker build -t fire-detection -f mlops/flask_app/Dockerfile .
docker run -p 5000:5000 -v "$PWD/segmentation/weights:/models:ro" fire-detection
```

On Google Cloud the flow is: push to GitHub → Cloud Build builds and pushes the
image to Artifact Registry → a Pub/Sub message fires → a Cloud Run Function
patches the GKE deployment → rolling update with zero downtime.

See [`mlops/DEPLOYMENT_GUIDE.md`](mlops/DEPLOYMENT_GUIDE.md) for the full setup
and [`mlops/README.md`](mlops/README.md) for how the pieces fit together.

## Configuration

Every field of `FireDetectionConfig` can be set with a `FIRE_<FIELD>`
environment variable, and the serving app reads its own `MODEL_PATH`,
`DEVICE`, `SCORE_THRESHOLD` and friends. See `.env.example` for the full list.

```bash
FIRE_LEARNING_RATE=0.001 FIRE_TRAIN_EPOCHS=50 python train.py
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

118 tests covering VIA parsing and mask rasterisation, the metrics, the torch
dataset, model construction and a real training step, checkpoint round-trips,
the Flask API and the Cloud Function. The model tests run on CPU with
randomly initialised weights and 64x48 images, so the whole suite finishes in
about 15 seconds without downloading anything.

## Metrics

Evaluation matches predicted masks to ground truth greedily by descending
score, with each ground-truth instance claimed at most once, then reports:

- **Precision / recall / F1** at an IoU threshold (default 0.5)
- **Mean IoU** and **mean Dice** over matched pairs
- **AP@0.5** with all-point interpolation

## Notes and limitations

- Fire is a single class. Adding smoke means extending `CLASS_NAMES` and
  `NUM_CLASSES`, then retraining.
- Inference is CPU-friendly but not real-time; a 512px image takes roughly a
  second per core. Use a GPU node pool for video-rate throughput.
- Model weights, datasets and `.env` are git-ignored. Ship checkpoints through
  Cloud Storage, not the repository.

## References

- [Mask R-CNN](https://arxiv.org/abs/1703.06870) (He et al., 2017)
- [torchvision detection models](https://pytorch.org/vision/stable/models.html#object-detection-instance-segmentation-and-person-keypoint-detection)
- [VGG Image Annotator](https://www.robots.ox.ac.uk/~vgg/software/via/)

## License

MIT
