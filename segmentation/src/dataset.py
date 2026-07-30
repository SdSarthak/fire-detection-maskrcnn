"""
Dataset loading for Mask R-CNN fire detection.

Reads images plus polygon annotations exported by the VGG Image Annotator
(VIA) and turns them into the ``(image, target)`` pairs that torchvision's
Mask R-CNN expects:

    image  -- float tensor, CHW, values in [0, 1]
    target -- {'boxes', 'labels', 'masks', 'image_id', 'area', 'iscrowd'}
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}

# Newer VIA exports nest the per-image records under this key.
VIA_METADATA_KEY = '_via_img_metadata'


# --------------------------------------------------------------------------- #
# VIA annotation parsing
# --------------------------------------------------------------------------- #
def _shape_to_polygon(shape: Dict) -> Optional[np.ndarray]:
    """Convert a single VIA ``shape_attributes`` entry into an (N, 2) polygon.

    Polygons/polylines are used directly; rectangles, circles and ellipses are
    rasterised into equivalent polygons so every VIA shape type is usable.
    Returns ``None`` when the shape cannot produce a closed area.
    """
    name = shape.get('name', 'polygon')

    if 'all_points_x' in shape and 'all_points_y' in shape:
        xs = shape['all_points_x']
        ys = shape['all_points_y']
        if len(xs) != len(ys) or len(xs) < 3:
            return None
        return np.stack([np.asarray(xs, dtype=np.float64),
                         np.asarray(ys, dtype=np.float64)], axis=1)

    if name == 'rect' and {'x', 'y', 'width', 'height'} <= set(shape):
        x, y = float(shape['x']), float(shape['y'])
        w, h = float(shape['width']), float(shape['height'])
        if w <= 0 or h <= 0:
            return None
        return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                        dtype=np.float64)

    if name in {'circle', 'ellipse'} and {'cx', 'cy'} <= set(shape):
        cx, cy = float(shape['cx']), float(shape['cy'])
        if name == 'circle':
            rx = ry = float(shape.get('r', 0.0))
        else:
            rx = float(shape.get('rx', 0.0))
            ry = float(shape.get('ry', 0.0))
        if rx <= 0 or ry <= 0:
            return None
        theta = np.linspace(0.0, 2.0 * math.pi, num=32, endpoint=False)
        return np.stack([cx + rx * np.cos(theta), cy + ry * np.sin(theta)], axis=1)

    return None


def _iter_regions(regions) -> List[Dict]:
    """VIA stores regions either as a list or as a dict keyed by index."""
    if isinstance(regions, dict):
        return [r for _, r in sorted(regions.items(), key=lambda kv: str(kv[0]))]
    if isinstance(regions, list):
        return list(regions)
    return []


def _region_matches(region: Dict, class_filter: Optional[Sequence[str]]) -> bool:
    """Keep a region when no filter is given, or when any attribute matches."""
    if not class_filter:
        return True
    wanted = {str(c).lower() for c in class_filter}
    attributes = region.get('region_attributes') or {}
    for value in attributes.values():
        if isinstance(value, str) and value.lower() in wanted:
            return True
        if isinstance(value, dict):  # VIA checkbox attributes
            for key, selected in value.items():
                if selected and str(key).lower() in wanted:
                    return True
    return False


def parse_via_annotations(via_data: Dict,
                          class_filter: Optional[Sequence[str]] = None
                          ) -> Dict[str, List[np.ndarray]]:
    """Parse raw VIA JSON into ``{filename: [polygon, ...]}``.

    Args:
        via_data: Object decoded from a VIA project or annotation export.
        class_filter: Optional region attribute values to keep (case
            insensitive), e.g. ``['fire']``. ``None`` keeps every region.

    Returns:
        Mapping of image filename to a list of ``(N, 2)`` float polygons.
    """
    records = via_data.get(VIA_METADATA_KEY, via_data) if isinstance(via_data, dict) else {}

    parsed: Dict[str, List[np.ndarray]] = {}
    for key, image_data in records.items():
        if not isinstance(image_data, dict):
            continue
        filename = image_data.get('filename')
        if not filename:
            # VIA keys look like "fire_1.jpg12345"; fall back to the key itself.
            filename = str(key)

        polygons: List[np.ndarray] = []
        for region in _iter_regions(image_data.get('regions')):
            if not isinstance(region, dict):
                continue
            if not _region_matches(region, class_filter):
                continue
            polygon = _shape_to_polygon(region.get('shape_attributes') or {})
            if polygon is not None:
                polygons.append(polygon)

        parsed.setdefault(filename, []).extend(polygons)

    return parsed


def load_via_annotations(path,
                         class_filter: Optional[Sequence[str]] = None
                         ) -> Dict[str, List[np.ndarray]]:
    """Load and parse a VIA annotation file from disk."""
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f'Annotation file not found: {json_path}')
    with open(json_path, 'r', encoding='utf-8') as handle:
        return parse_via_annotations(json.load(handle), class_filter=class_filter)


# --------------------------------------------------------------------------- #
# Mask helpers
# --------------------------------------------------------------------------- #
def polygon_to_mask(polygon: np.ndarray, height: int, width: int) -> np.ndarray:
    """Rasterise one polygon into a ``(height, width)`` uint8 binary mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    points = np.asarray(polygon, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 3:
        return mask
    points = np.round(points).astype(np.int32)
    np.clip(points[:, 0], 0, width - 1, out=points[:, 0])
    np.clip(points[:, 1], 0, height - 1, out=points[:, 1])
    cv2.fillPoly(mask, [points], 1)
    return mask


def polygons_to_masks(polygons: Sequence[np.ndarray], height: int,
                      width: int) -> np.ndarray:
    """Rasterise polygons into a ``(N, height, width)`` uint8 mask stack."""
    if not polygons:
        return np.zeros((0, height, width), dtype=np.uint8)
    return np.stack([polygon_to_mask(p, height, width) for p in polygons])


def masks_to_boxes(masks: np.ndarray) -> np.ndarray:
    """Derive tight ``xyxy`` boxes from a ``(N, H, W)`` binary mask stack."""
    if masks.size == 0:
        return np.zeros((0, 4), dtype=np.float32)

    boxes = np.zeros((masks.shape[0], 4), dtype=np.float32)
    for i, mask in enumerate(masks):
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            continue
        # +1 on the far edge so a single-pixel mask still has non-zero area.
        boxes[i] = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    return boxes


def drop_degenerate(masks: np.ndarray, boxes: np.ndarray,
                    min_area: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """Remove instances whose rasterised mask is empty or vanishingly small."""
    if masks.size == 0:
        return masks, boxes
    areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    keep = (areas >= min_area) & (widths > 0) & (heights > 0)
    return masks[keep], boxes[keep]


def list_images(directory) -> List[Path]:
    """Return every supported image file under ``directory``, sorted by name."""
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f'Image directory not found: {root}')
    return sorted(p for p in root.rglob('*')
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def load_image(image_path) -> np.ndarray:
    """Load an image from disk as an RGB uint8 array."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Failed to load image: {image_path}')
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert an HWC uint8 RGB array into a CHW float tensor in [0, 1]."""
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    if array.shape[2] == 4:
        array = array[:, :, :3]
    if array.dtype != np.float32:
        array = array.astype(np.float32)
    if array.max() > 1.0:
        array = array / 255.0
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class FireSegmentationDataset(Dataset):
    """Instance-segmentation dataset backed by images plus VIA polygons.

    Args:
        images_dir: Directory holding the image files for this split.
        annotations: Path to a VIA JSON file, or an already parsed
            ``{filename: [polygon, ...]}`` mapping. ``None`` yields an
            unlabelled dataset (useful for prediction over a folder).
        class_filter: Region attribute values to keep, e.g. ``['fire']``.
        require_annotations: Skip images that have no usable polygons. Turn
            this off to train with explicit negative (fire-free) examples.
        horizontal_flip_prob: Probability of the random horizontal flip
            augmentation. Set to 0 for validation and test splits.
        label: Class index assigned to every instance (1 = fire).
    """

    def __init__(self,
                 images_dir,
                 annotations=None,
                 class_filter: Optional[Sequence[str]] = None,
                 require_annotations: bool = True,
                 horizontal_flip_prob: float = 0.0,
                 label: int = 1,
                 min_instance_area: int = 4):
        self.images_dir = Path(images_dir)
        self.horizontal_flip_prob = float(horizontal_flip_prob)
        self.label = int(label)
        self.min_instance_area = int(min_instance_area)

        if annotations is None:
            self.annotations: Dict[str, List[np.ndarray]] = {}
        elif isinstance(annotations, dict):
            self.annotations = annotations
        else:
            self.annotations = load_via_annotations(annotations, class_filter=class_filter)

        all_images = list_images(self.images_dir)
        if require_annotations:
            self.image_paths = [p for p in all_images if self.annotations.get(p.name)]
        else:
            self.image_paths = all_images

    def __len__(self) -> int:
        return len(self.image_paths)

    @property
    def unannotated(self) -> List[str]:
        """Names of images in the directory that carry no usable polygons."""
        return [p.name for p in list_images(self.images_dir)
                if not self.annotations.get(p.name)]

    def polygons_for(self, filename: str) -> List[np.ndarray]:
        """Polygons registered for an image name (empty list when unlabelled)."""
        return list(self.annotations.get(filename, []))

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        image_path = self.image_paths[index]
        image = load_image(image_path)
        height, width = image.shape[:2]

        masks = polygons_to_masks(self.polygons_for(image_path.name), height, width)
        boxes = masks_to_boxes(masks)
        masks, boxes = drop_degenerate(masks, boxes, min_area=self.min_instance_area)

        if self.horizontal_flip_prob > 0 and torch.rand(1).item() < self.horizontal_flip_prob:
            image = np.ascontiguousarray(image[:, ::-1])
            if masks.size:
                masks = np.ascontiguousarray(masks[:, :, ::-1])
                flipped = boxes.copy()
                flipped[:, 0] = width - boxes[:, 2]
                flipped[:, 2] = width - boxes[:, 0]
                boxes = flipped

        areas = (masks.reshape(masks.shape[0], -1).sum(axis=1)
                 if masks.size else np.zeros((0,), dtype=np.float32))

        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            'labels': torch.full((masks.shape[0],), self.label, dtype=torch.int64),
            'masks': torch.as_tensor(masks, dtype=torch.uint8).reshape(-1, height, width),
            'image_id': torch.tensor([index], dtype=torch.int64),
            'area': torch.as_tensor(areas, dtype=torch.float32).reshape(-1),
            'iscrowd': torch.zeros((masks.shape[0],), dtype=torch.int64),
        }
        return image_to_tensor(image), target


def collate_fn(batch):
    """Keep variable-sized images as a tuple instead of stacking them."""
    return tuple(zip(*batch))
