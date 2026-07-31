"""
Inference wrapper around the trained Mask R-CNN fire detector.

Shared by the CLI (``infer.py``) and the Flask serving app so both return
byte-for-byte identical results.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from .config import FireDetectionConfig
from .dataset import image_to_tensor, load_image
from .model import build_model, load_model, masks_to_numpy, resolve_device

# BGR-agnostic RGB colours cycled per instance when drawing overlays.
_OVERLAY_COLORS = (
    (255, 64, 0), (255, 170, 0), (255, 0, 128),
    (0, 200, 255), (140, 0, 255), (0, 255, 140),
)

# A 40 kB PNG can decode to 30000x30000 px (~2.7 GB as float32 CHW). Cap the
# decoded size so a hostile or accidental upload cannot exhaust memory.
DEFAULT_MAX_IMAGE_PIXELS = 40_000_000  # ~ 6300 x 6300


def validate_image(image: np.ndarray,
                   max_pixels: int = DEFAULT_MAX_IMAGE_PIXELS) -> np.ndarray:
    """Reject images the inference path cannot safely handle.

    Raises ``ValueError`` for empty, wrongly-shaped or absurdly large arrays.
    The pixel cap matters most on the serving path: HTTP body limits bound the
    *compressed* size, not the decoded one.
    """
    array = np.asarray(image)
    if array.ndim not in (2, 3):
        raise ValueError(
            f'Expected an HW or HWC image, got an array of shape {array.shape}')
    height, width = array.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError(f'Image has a zero dimension: {width}x{height}')
    if max_pixels and height * width > max_pixels:
        raise ValueError(
            f'Image is too large: {width}x{height} = {height * width:,} pixels '
            f'(limit {max_pixels:,}). Downscale it before submitting.')
    return array


def mask_to_polygons(mask: np.ndarray, epsilon_ratio: float = 0.004,
                     max_points: int = 80) -> List[List[List[int]]]:
    """Trace a binary mask into simplified polygons for compact JSON output."""
    binary = np.ascontiguousarray(mask.astype(np.uint8))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons: List[List[List[int]]] = []
    for contour in contours:
        if len(contour) < 3:
            continue
        epsilon = epsilon_ratio * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(approx) < 3:
            continue
        if len(approx) > max_points:
            step = int(np.ceil(len(approx) / max_points))
            approx = approx[::step]
        polygons.append([[int(x), int(y)] for x, y in approx])
    return polygons


class FirePredictor:
    """Run fire instance segmentation on images.

    Args:
        model: A built Mask R-CNN module in eval mode.
        config: The configuration the model was built with.
        device: Device the model lives on.
        score_threshold: Minimum detection confidence to report.
        mask_threshold: Probability cut-off used to binarise soft masks.
    """

    def __init__(self, model, config: FireDetectionConfig, device=None,
                 score_threshold: Optional[float] = None,
                 mask_threshold: Optional[float] = None,
                 max_image_pixels: int = DEFAULT_MAX_IMAGE_PIXELS):
        self.model = model
        self.config = config
        self.device = resolve_device(device) if isinstance(device, str) or device is None \
            else device
        self.score_threshold = float(
            config.DETECTION_MIN_CONFIDENCE if score_threshold is None else score_threshold)
        self.mask_threshold = float(
            config.MASK_BINARY_THRESHOLD if mask_threshold is None else mask_threshold)
        if not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError(
                f'score_threshold must be in [0, 1]; got {self.score_threshold}')
        if not 0.0 <= self.mask_threshold <= 1.0:
            raise ValueError(
                f'mask_threshold must be in [0, 1]; got {self.mask_threshold}')
        self.max_image_pixels = int(max_image_pixels)
        self.class_names = list(config.CLASS_NAMES)
        self.model.eval()

    # ---------------------------------------------------------------- factories
    @classmethod
    def from_checkpoint(cls, path, device: str = 'auto',
                        score_threshold: Optional[float] = None,
                        mask_threshold: Optional[float] = None) -> 'FirePredictor':
        """Load a trained checkpoint from disk."""
        model, config = load_model(path, device=device)
        return cls(model, config, resolve_device(device),
                   score_threshold=score_threshold, mask_threshold=mask_threshold)

    @classmethod
    def untrained(cls, config: Optional[FireDetectionConfig] = None,
                  device: str = 'cpu', **kwargs) -> 'FirePredictor':
        """Build a predictor with randomly initialised heads (tests, smoke runs)."""
        config = config or FireDetectionConfig(PRETRAINED_WEIGHTS='none')
        model = build_model(config)
        torch_device = resolve_device(device)
        model.to(torch_device)
        model.eval()
        return cls(model, config, torch_device, **kwargs)

    # ---------------------------------------------------------------- inference
    def class_name(self, class_id: int) -> str:
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return f'class_{class_id}'

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """Segment a single RGB uint8 image."""
        return self.predict_batch([image])[0]

    @torch.no_grad()
    def predict_batch(self, images: Sequence[np.ndarray]) -> List[Dict[str, Any]]:
        """Segment several RGB images in one forward pass."""
        if not images:
            return []
        images = [validate_image(image, self.max_image_pixels) for image in images]
        tensors = [image_to_tensor(image).to(self.device) for image in images]
        try:
            outputs = self.model(tensors)
        finally:
            # Drop the device copies as soon as the forward pass is done rather
            # than waiting for the whole batch of results to be built.
            del tensors
        return [self._build_result(output, image)
                for output, image in zip(outputs, images)]

    def predict_file(self, image_path) -> Tuple[Dict[str, Any], np.ndarray]:
        """Segment an image from disk; returns ``(result, rgb_image)``."""
        path = Path(image_path)
        image = load_image(path)
        result = self.predict(image)
        result['filename'] = path.name
        return result, image

    def _build_result(self, output: Dict[str, torch.Tensor],
                      image: np.ndarray) -> Dict[str, Any]:
        height, width = image.shape[:2]
        pixels = float(max(height * width, 1))
        scores = output['scores'].detach().cpu().numpy()
        keep = scores >= self.score_threshold

        boxes = output['boxes'].detach().cpu().numpy()[keep]
        labels = output['labels'].detach().cpu().numpy()[keep]
        scores = scores[keep]
        masks = masks_to_numpy(output['masks'], self.mask_threshold)[keep]

        detections: List[Dict[str, Any]] = []
        for index in range(len(scores)):
            mask = masks[index]
            area = int(mask.sum())
            x1, y1, x2, y2 = (float(v) for v in boxes[index])
            detections.append({
                'class_id': int(labels[index]),
                'class_name': self.class_name(int(labels[index])),
                'score': float(scores[index]),
                'bounding_box': {
                    'x1': round(x1, 2), 'y1': round(y1, 2),
                    'x2': round(x2, 2), 'y2': round(y2, 2),
                },
                'mask_area_px': area,
                'mask_area_ratio': round(area / pixels, 6),
                'polygons': mask_to_polygons(mask),
            })

        union = (np.logical_or.reduce(masks) if len(masks)
                 else np.zeros((height, width), dtype=bool))
        fire_pixels = int(union.sum())

        return {
            'detections': detections,
            'num_detections': len(detections),
            'is_fire': len(detections) > 0,
            'confidence': float(scores.max()) if len(scores) else 0.0,
            'fire_area_ratio': round(fire_pixels / pixels, 6),
            'fire_area_px': fire_pixels,
            'image_size': {'height': int(height), 'width': int(width)},
            'score_threshold': self.score_threshold,
            'masks': masks,
        }

    # ------------------------------------------------------------ presentation
    @staticmethod
    def serializable(result: Dict[str, Any]) -> Dict[str, Any]:
        """Strip the raw mask array so the result can be JSON encoded."""
        return {key: value for key, value in result.items() if key != 'masks'}

    def render_overlay(self, image: np.ndarray, result: Dict[str, Any],
                       alpha: float = 0.45) -> np.ndarray:
        """Draw masks, boxes and labels onto a copy of the RGB image."""
        canvas = np.ascontiguousarray(np.asarray(image).copy())
        if canvas.ndim == 2:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
        elif canvas.ndim == 3 and canvas.shape[2] == 4:
            canvas = canvas[:, :, :3]
        if canvas.dtype != np.uint8:
            # A float image in [0, 1] would render as solid black under a plain
            # astype(uint8); rescale it instead.
            canvas = np.asarray(canvas, dtype=np.float64)
            peak = float(np.nanmax(canvas)) if canvas.size else 0.0
            if np.isfinite(peak) and peak <= 1.0:
                canvas = canvas * 255.0
            canvas = np.clip(np.nan_to_num(canvas), 0, 255).astype(np.uint8)
        canvas = np.ascontiguousarray(canvas)

        masks = result.get('masks')
        detections = result.get('detections', [])

        for index, detection in enumerate(detections):
            color = _OVERLAY_COLORS[index % len(_OVERLAY_COLORS)]

            if masks is not None and index < len(masks):
                mask = np.asarray(masks[index]).astype(bool)
                if mask.shape != canvas.shape[:2]:
                    mask = np.zeros(canvas.shape[:2], dtype=bool)
                if mask.any():
                    tint = np.zeros_like(canvas)
                    tint[mask] = color
                    canvas = np.where(mask[:, :, None],
                                      (canvas * (1 - alpha) + tint * alpha).astype(np.uint8),
                                      canvas)

            box = detection['bounding_box']
            x1, y1 = int(box['x1']), int(box['y1'])
            x2, y2 = int(box['x2']), int(box['y2'])
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

            label = f"{detection['class_name']} {detection['score']:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top = max(y1 - text_h - 6, 0)
            cv2.rectangle(canvas, (x1, top), (x1 + text_w + 6, top + text_h + 6), color, -1)
            cv2.putText(canvas, label, (x1 + 3, top + text_h + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        if not detections:
            cv2.putText(canvas, 'no fire detected', (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2, cv2.LINE_AA)

        return canvas

    def save_overlay(self, image: np.ndarray, result: Dict[str, Any],
                     output_path) -> Path:
        """Render an overlay and write it to disk as an image file."""
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        overlay = self.render_overlay(image, result)
        if not cv2.imwrite(str(destination), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)):
            raise IOError(f'Failed to write overlay image: {destination}')
        return destination
