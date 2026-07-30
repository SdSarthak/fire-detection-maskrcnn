"""Shared fixtures: paths, synthetic fire images and VIA annotation files."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SEGMENTATION_ROOT = REPO_ROOT / 'segmentation'
FLASK_APP_ROOT = REPO_ROOT / 'mlops' / 'flask_app'

for path in (SEGMENTATION_ROOT, FLASK_APP_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def make_fire_image(width: int = 64, height: int = 48,
                    box=(10, 10, 40, 34)) -> np.ndarray:
    """A dark RGB image with one bright orange rectangle standing in for fire."""
    image = np.full((height, width, 3), 30, dtype=np.uint8)
    x1, y1, x2, y2 = box
    image[y1:y2, x1:x2] = (255, 120, 20)
    return image


def rectangle_polygon(x1: int, y1: int, x2: int, y2: int):
    """VIA-style polygon point lists for an axis-aligned rectangle."""
    return {
        'name': 'polygon',
        'all_points_x': [x1, x2, x2, x1],
        'all_points_y': [y1, y1, y2, y2],
    }


def via_project(entries) -> dict:
    """Build a VIA export dict from ``{filename: [shape_attributes, ...]}``."""
    project = {}
    for index, (filename, shapes) in enumerate(entries.items()):
        project[f'{filename}{index}'] = {
            'filename': filename,
            'size': -1,
            'regions': [{'shape_attributes': shape,
                         'region_attributes': {'object': 'fire'}}
                        for shape in shapes],
            'file_attributes': {},
        }
    return project


@pytest.fixture
def image_size():
    return (64, 48)  # width, height


@pytest.fixture
def dataset_dir(tmp_path, image_size):
    """A tiny two-split dataset on disk with matching VIA annotations."""
    width, height = image_size
    data_dir = tmp_path / 'data'
    annotations_dir = data_dir / 'annotations'
    annotations_dir.mkdir(parents=True)

    boxes = {
        'train': {'fire_train_1.jpg': (8, 8, 30, 28),
                  'fire_train_2.jpg': (20, 12, 50, 40)},
        'val': {'fire_val_1.jpg': (12, 10, 36, 32)},
    }

    for split, images in boxes.items():
        split_dir = data_dir / split
        split_dir.mkdir(parents=True)
        shapes = {}
        for filename, box in images.items():
            image = make_fire_image(width, height, box)
            cv2.imwrite(str(split_dir / filename), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            shapes[filename] = [rectangle_polygon(*box)]
        annotation_path = annotations_dir / f'{split}_annotations.json'
        annotation_path.write_text(json.dumps(via_project(shapes)), encoding='utf-8')

    return data_dir
