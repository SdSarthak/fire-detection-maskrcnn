# Segmentation module

Training and inference for the fire detection Mask R-CNN.

```
segmentation/
├── src/
│   ├── config.py      FireDetectionConfig / InferenceConfig
│   ├── dataset.py     VIA parsing, mask rasterisation, torch Dataset
│   ├── model.py       model construction, training, evaluation, checkpoints
│   ├── metrics.py     IoU / Dice / precision / recall / AP (pure NumPy)
│   └── predictor.py   FirePredictor: inference + overlay rendering
├── train.py           training entrypoint
├── infer.py           inference entrypoint
├── data/              train/ val/ test/ annotations/   (git-ignored)
├── weights/           checkpoints                      (git-ignored)
└── outputs/           overlays, predictions.json       (git-ignored)
```

## Install

```bash
pip install -r requirements.txt
```

CPU wheels are fine for inference and small fine-tuning runs. For GPU training
install the CUDA build of PyTorch from <https://pytorch.org/get-started/locally/>.

## Architecture

`build_model()` assembles `torchvision.models.detection.maskrcnn_resnet50_fpn`
and then replaces the two COCO heads:

| Stage | Component |
|---|---|
| Backbone | ResNet50 + Feature Pyramid Network (5 levels) |
| Proposals | RPN with anchors from `RPN_ANCHOR_SCALES` x `RPN_ANCHOR_RATIOS` |
| ROI features | `MultiScaleRoIAlign`, 7x7 for boxes, 14x14 for masks |
| Detection head | `FastRCNNPredictor` sized to `NUM_CLASSES` |
| Mask head | `MaskRCNNPredictor` sized to `NUM_CLASSES` |

`PRETRAINED_WEIGHTS` selects the starting point:

- `coco` (default) — full Mask R-CNN COCO weights, heads reinitialised
- `imagenet` — ImageNet backbone only
- `none` — random initialisation (used by the tests, no downloads)

`TRAINABLE_BACKBONE_LAYERS` (default 3) controls how many ResNet stages stay
unfrozen. It is only meaningful when pre-trained weights are loaded.

## Configuration

`FireDetectionConfig` is a dataclass. Fields can be set three ways, in
increasing priority: defaults → `FIRE_<FIELD>` environment variables → explicit
CLI flags.

```python
from src.config import FireDetectionConfig

config = FireDetectionConfig.from_env(TRAIN_EPOCHS=10)
config.display()
```

| Field | Default | Meaning |
|---|---|---|
| `IMAGE_MIN_DIM` / `IMAGE_MAX_DIM` | 256 / 512 | Detector rescales inputs into this range |
| `NUM_CLASSES` | 2 | Background + fire; must match `CLASS_NAMES` |
| `LEARNING_RATE` | 0.005 | SGD, with momentum 0.9 and weight decay 5e-4 |
| `TRAIN_EPOCHS` | 30 | Epochs |
| `BATCH_SIZE` | 2 | Images per step |
| `LR_STEP_SIZE` / `LR_GAMMA` | 10 / 0.1 | Step decay schedule |
| `GRAD_CLIP_NORM` | 10.0 | Gradient clipping; 0 disables |
| `HORIZONTAL_FLIP_PROB` | 0.5 | Training augmentation |
| `DETECTION_MIN_CONFIDENCE` | 0.7 | Score threshold |
| `DETECTION_NMS_THRESHOLD` | 0.3 | Box NMS |
| `MASK_BINARY_THRESHOLD` | 0.5 | Soft-mask cut-off |
| `EVAL_IOU_THRESHOLD` | 0.5 | IoU for counting a true positive |
| `PRETRAINED_WEIGHTS` | `coco` | Starting weights |
| `DEVICE` | `auto` | `auto` / `cpu` / `cuda` |

Invalid combinations raise at construction time — a `NUM_CLASSES` that does not
match `CLASS_NAMES`, `IMAGE_MIN_DIM > IMAGE_MAX_DIM`, or an unknown
`PRETRAINED_WEIGHTS` value.

## Data

Annotate with the [VGG Image Annotator](https://www.robots.ox.ac.uk/~vgg/software/via/):
one polygon per fire region, region attribute set to `fire`, then
*Annotation → Export Annotations (as json)*.

```json
{
  "fire_1.jpg102400": {
    "filename": "fire_1.jpg",
    "regions": [
      {
        "shape_attributes": {
          "name": "polygon",
          "all_points_x": [100, 150, 200, 100],
          "all_points_y": [100, 80, 120, 150]
        },
        "region_attributes": {"object": "fire"}
      }
    ]
  }
}
```

`parse_via_annotations` handles both the flat layout above and the newer
`{"_via_img_metadata": {...}}` wrapper, regions stored as either a list or an
index-keyed object, and rect/circle/ellipse shapes (converted to polygons).
Degenerate shapes — fewer than three points, zero width, empty rasterisation —
are dropped rather than becoming empty training targets.

```python
from src.dataset import FireSegmentationDataset

dataset = FireSegmentationDataset(
    images_dir='data/train',
    annotations='data/annotations/train_annotations.json',
    class_filter=['fire'],
    horizontal_flip_prob=0.5,
)
image, target = dataset[0]     # CHW float tensor, {'boxes','labels','masks',...}
print(dataset.unannotated)     # images the annotation file does not cover
```

## Training

```bash
python train.py --epochs 30
python train.py --data-dir /data/fire --pretrained imagenet --device cuda
python train.py --pretrained none --epochs 1        # offline smoke run
```

Per epoch the script prints `loss_classifier`, `loss_box_reg`, `loss_mask`,
`loss_objectness` and `loss_rpn_box_reg`, then validation loss and the full
metric set. The best validation F1 wins the checkpoint; with no validation
split the lowest training loss wins instead.

A diverging loss aborts with the offending components and a pointer at
`LEARNING_RATE` rather than silently producing NaN weights.

The first epoch uses a linear learning-rate warm-up, which matters because
freshly initialised detection heads produce large early gradients.

`outputs/training_history.json` is rewritten atomically after every epoch, so a
crash at epoch 27 of 30 still leaves the first 27 on disk.

### Reproducibility

`--seed` (default 42) seeds `random`, `numpy` and torch, seeds every CUDA
device, and drives a dedicated generator for the training `DataLoader` so the
shuffle order does not depend on anything else drawing from the global RNG.
DataLoader workers are seeded through `worker_init_fn`. Add `--deterministic`
to also pin cuDNN to deterministic kernels (slower, GPU only).

### Split hygiene

Training aborts if any image basename appears in both `data/train` and
`data/val` - otherwise every reported `val_f1` would be a training score.
Duplicate basenames *within* one split raise a warning, because annotations are
keyed by basename and the duplicates would share labels.

## Inference

```bash
python infer.py --input data/test
python infer.py --input photo.jpg --score-threshold 0.4 --json results.json
python infer.py --input data/test --no-overlay
```

```python
from src.predictor import FirePredictor

predictor = FirePredictor.from_checkpoint('weights/fire_detection_model.pt')
result, image = predictor.predict_file('photo.jpg')

print(result['is_fire'], result['confidence'], result['fire_area_ratio'])
predictor.save_overlay(image, result, 'outputs/overlay.png')

payload = FirePredictor.serializable(result)   # drops the raw mask array
```

`result['masks']` is the raw `(N, H, W)` boolean stack; `serializable()` strips
it so the rest can be JSON encoded. Each detection also carries simplified
`polygons`, which is the compact way to persist a mask.

## Evaluation

`SegmentationEvaluator` matches predictions to ground truth greedily by
descending score, claiming each ground-truth instance at most once, so two
overlapping predictions on one fire count as one true positive and one false
positive rather than two hits.

```python
from src.metrics import SegmentationEvaluator

evaluator = SegmentationEvaluator(iou_threshold=0.5)
evaluator.add(pred_masks, scores, gt_masks)
print(evaluator.compute())
# {'precision': ..., 'recall': ..., 'f1': ..., 'mean_iou': ...,
#  'mean_dice': ..., 'ap@0.5': ..., 'true_positives': ..., ...}
```

## Troubleshooting

**"No annotated training images found"** — the filenames in the VIA JSON must
match the files on disk exactly, including extension and case.

**Out of memory** — lower `BATCH_SIZE` to 1 and `IMAGE_MAX_DIM` to 320, or
reduce `TRAINABLE_BACKBONE_LAYERS` to 1.

**Loss will not fall** — with only a few dozen images, start from `coco`
weights and keep `TRAINABLE_BACKBONE_LAYERS` low; training the whole backbone
on a small dataset overfits within a handful of epochs.

**No detections at inference** — `DETECTION_MIN_CONFIDENCE` defaults to 0.7.
Pass `--score-threshold 0.3` to see what the model is actually proposing.
