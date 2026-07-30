"""
Mask R-CNN model for fire detection.

Builds a genuine Mask R-CNN (ResNet50 + FPN backbone, region proposal
network, ROI Align, box head and mask head) from torchvision and adapts the
detection/mask predictors to the fire classes, then provides the training and
evaluation loops.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torchvision
from torch import nn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign

from .config import FireDetectionConfig
from .metrics import SegmentationEvaluator

# FPN exposes four feature maps to the ROI heads (the fifth is RPN-only).
_ROI_FEATMAP_NAMES = ['0', '1', '2', '3']


def resolve_device(device: str = 'auto') -> torch.device:
    """Turn ``'auto' | 'cpu' | 'cuda'`` into a concrete :class:`torch.device`."""
    if device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device)


def _detection_weights(config: FireDetectionConfig):
    """Select the torchvision weight enums implied by ``PRETRAINED_WEIGHTS``."""
    from torchvision.models import ResNet50_Weights
    from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights

    if config.PRETRAINED_WEIGHTS == 'coco':
        return MaskRCNN_ResNet50_FPN_Weights.COCO_V1, ResNet50_Weights.IMAGENET1K_V1
    if config.PRETRAINED_WEIGHTS == 'imagenet':
        return None, ResNet50_Weights.IMAGENET1K_V1
    return None, None


def build_model(config: Optional[FireDetectionConfig] = None) -> nn.Module:
    """Build the Mask R-CNN model described by ``config``.

    Pre-trained COCO weights are loaded first (when requested) and the box and
    mask predictors are then swapped for freshly initialised heads sized to
    ``config.NUM_CLASSES``, which is the standard fine-tuning recipe.
    """
    config = config or FireDetectionConfig()
    if config.BACKBONE != 'resnet50':
        raise ValueError(f'Unsupported backbone: {config.BACKBONE!r} (expected resnet50)')

    detection_weights, backbone_weights = _detection_weights(config)

    anchor_sizes = tuple((size,) for size in config.RPN_ANCHOR_SCALES)
    anchor_generator = AnchorGenerator(
        sizes=anchor_sizes,
        aspect_ratios=(tuple(config.RPN_ANCHOR_RATIOS),) * len(anchor_sizes),
    )

    # torchvision only honours (and only warns about) this when weights are loaded.
    trainable_layers = (config.TRAINABLE_BACKBONE_LAYERS
                        if (detection_weights or backbone_weights) else None)

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights=detection_weights,
        weights_backbone=backbone_weights,
        trainable_backbone_layers=trainable_layers,
        min_size=config.IMAGE_MIN_DIM,
        max_size=config.IMAGE_MAX_DIM,
        rpn_anchor_generator=anchor_generator,
        rpn_nms_thresh=config.RPN_NMS_THRESHOLD,
        rpn_batch_size_per_image=config.RPN_TRAIN_ANCHORS_PER_IMAGE,
        box_roi_pool=MultiScaleRoIAlign(featmap_names=_ROI_FEATMAP_NAMES,
                                        output_size=config.POOL_SIZE,
                                        sampling_ratio=2),
        mask_roi_pool=MultiScaleRoIAlign(featmap_names=_ROI_FEATMAP_NAMES,
                                         output_size=config.MASK_POOL_SIZE,
                                         sampling_ratio=2),
        box_score_thresh=config.DETECTION_MIN_CONFIDENCE,
        box_nms_thresh=config.DETECTION_NMS_THRESHOLD,
        box_detections_per_img=config.DETECTION_MAX_INSTANCES,
    )

    # Replace the COCO detection head with one sized for our classes.
    box_in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(box_in_features, config.NUM_CLASSES)

    # Replace the COCO mask head likewise.
    mask_in_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        mask_in_features, config.MASK_HIDDEN_LAYER, config.NUM_CLASSES)

    return model


def create_optimizer(model: nn.Module, config: FireDetectionConfig):
    """SGD with momentum over the trainable parameters only."""
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.SGD(params,
                           lr=config.LEARNING_RATE,
                           momentum=config.LEARNING_MOMENTUM,
                           weight_decay=config.WEIGHT_DECAY)


def create_scheduler(optimizer, config: FireDetectionConfig):
    """Step the learning rate down every ``LR_STEP_SIZE`` epochs."""
    return torch.optim.lr_scheduler.StepLR(optimizer,
                                           step_size=config.LR_STEP_SIZE,
                                           gamma=config.LR_GAMMA)


def _warmup_scheduler(optimizer, num_iterations: int):
    """Linear LR warm-up used for the first epoch to stabilise early steps."""
    warmup_iters = min(1000, max(num_iterations - 1, 1))
    start_factor = 1.0 / 1000

    def factor(step: int) -> float:
        if step >= warmup_iters:
            return 1.0
        alpha = step / warmup_iters
        return start_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _to_device(targets: Iterable[Dict[str, torch.Tensor]], device) -> List[Dict[str, torch.Tensor]]:
    return [{k: v.to(device) for k, v in t.items()} for t in targets]


def train_one_epoch(model: nn.Module, optimizer, data_loader, device,
                    epoch: int, config: FireDetectionConfig,
                    print_freq: int = 10) -> Dict[str, float]:
    """Run one training epoch and return the mean of each loss component."""
    model.train()
    warmup = _warmup_scheduler(optimizer, len(data_loader)) if epoch == 0 else None

    totals: Dict[str, float] = {}
    batches = 0
    started = time.time()

    for step, (images, targets) in enumerate(data_loader):
        images = [image.to(device) for image in images]
        targets = _to_device(targets, device)

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        loss_value = float(loss.item())

        if not math.isfinite(loss_value):
            raise RuntimeError(
                f'Loss diverged to {loss_value} at epoch {epoch} step {step}. '
                f'Lower LEARNING_RATE ({config.LEARNING_RATE}) and retry. '
                f'Components: { {k: float(v) for k, v in loss_dict.items()} }'
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if config.GRAD_CLIP_NORM > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
        optimizer.step()
        if warmup is not None:
            warmup.step()

        batches += 1
        totals['loss'] = totals.get('loss', 0.0) + loss_value
        for name, value in loss_dict.items():
            totals[name] = totals.get(name, 0.0) + float(value.item())

        if print_freq and step % print_freq == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f'  epoch {epoch} [{step + 1}/{len(data_loader)}] '
                  f'loss={loss_value:.4f} lr={lr:.6f}')

    if batches == 0:
        raise RuntimeError('Training data loader produced no batches; is data/train empty?')

    averages = {name: total / batches for name, total in totals.items()}
    averages['seconds'] = time.time() - started
    return averages


@torch.no_grad()
def compute_validation_loss(model: nn.Module, data_loader, device) -> float:
    """Mean total loss over the validation split.

    Mask R-CNN only returns losses in training mode, so the module is switched
    to ``train()`` while gradients stay disabled. The backbone uses frozen
    batch norm, so no running statistics are disturbed by this.
    """
    was_training = model.training
    model.train()
    total, batches = 0.0, 0
    try:
        for images, targets in data_loader:
            images = [image.to(device) for image in images]
            targets = _to_device(targets, device)
            loss_dict = model(images, targets)
            total += float(sum(loss_dict.values()).item())
            batches += 1
    finally:
        model.train(was_training)
    return total / batches if batches else float('nan')


@torch.no_grad()
def evaluate(model: nn.Module, data_loader, device,
             config: FireDetectionConfig) -> Dict[str, float]:
    """Score the model on a labelled split using mask IoU matching."""
    model.eval()
    evaluator = SegmentationEvaluator(iou_threshold=config.EVAL_IOU_THRESHOLD)

    for images, targets in data_loader:
        outputs = model([image.to(device) for image in images])
        for output, target in zip(outputs, targets):
            pred_masks = masks_to_numpy(output['masks'], config.MASK_BINARY_THRESHOLD)
            scores = output['scores'].detach().cpu().numpy().tolist()
            gt_masks = target['masks'].detach().cpu().numpy().astype(bool)
            evaluator.add(pred_masks, scores, gt_masks)

    return evaluator.compute()


def masks_to_numpy(masks: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    """Binarise Mask R-CNN's ``(N, 1, H, W)`` soft masks into ``(N, H, W)`` bools."""
    if masks.numel() == 0:
        shape = tuple(masks.shape)
        height = shape[-2] if len(shape) >= 2 else 0
        width = shape[-1] if len(shape) >= 1 else 0
        return np.zeros((0, height, width), dtype=bool)
    array = masks.detach().cpu().numpy()
    if array.ndim == 4:
        array = array[:, 0]
    return array >= threshold


def save_checkpoint(model: nn.Module, path, config: FireDetectionConfig,
                    epoch: int = 0, metrics: Optional[Dict[str, float]] = None) -> Path:
    """Persist weights plus the configuration needed to rebuild the model."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.to_dict(),
        'class_names': list(config.CLASS_NAMES),
        'epoch': int(epoch),
        'metrics': {k: float(v) for k, v in (metrics or {}).items()},
    }, destination)
    return destination


def load_checkpoint(path, map_location='cpu') -> Dict[str, Any]:
    """Load a checkpoint dict, tolerating older PyTorch pickling defaults."""
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f'Model checkpoint not found: {checkpoint_path}')
    try:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    except Exception:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)


def load_model(path, device='auto',
               config_overrides: Optional[Dict[str, Any]] = None
               ) -> Tuple[nn.Module, FireDetectionConfig]:
    """Rebuild a model from a checkpoint and return it in eval mode."""
    torch_device = resolve_device(device)
    checkpoint = load_checkpoint(path, map_location=torch_device)

    config_data = dict(checkpoint.get('config') or {})
    config_data.update(config_overrides or {})
    # Never re-download pre-trained weights when the checkpoint already has them.
    config_data['PRETRAINED_WEIGHTS'] = 'none'
    config = FireDetectionConfig.from_dict(config_data)

    model = build_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(torch_device)
    model.eval()
    return model, config
