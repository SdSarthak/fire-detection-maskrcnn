"""Tests for the torch Dataset that feeds Mask R-CNN."""
from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from conftest import make_fire_image
from src.dataset import (FireSegmentationDataset, collate_fn, image_to_tensor,
                         list_images, load_image)


def test_lists_supported_images_only(tmp_path):
    (tmp_path / 'a.jpg').write_bytes(b'x')
    (tmp_path / 'b.PNG').write_bytes(b'x')
    (tmp_path / 'notes.txt').write_text('x')

    names = [p.name for p in list_images(tmp_path)]

    assert names == ['a.jpg', 'b.PNG']


def test_list_images_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        list_images(tmp_path / 'absent')


def test_load_image_returns_rgb(tmp_path):
    path = tmp_path / 'fire.png'
    cv2.imwrite(str(path), np.array([[[0, 0, 255]]], dtype=np.uint8))  # BGR red

    np.testing.assert_array_equal(load_image(path)[0, 0], [255, 0, 0])


def test_load_image_rejects_unreadable_file(tmp_path):
    path = tmp_path / 'broken.jpg'
    path.write_bytes(b'not an image')
    with pytest.raises(ValueError):
        load_image(path)


def test_image_to_tensor_normalises_and_transposes():
    tensor = image_to_tensor(make_fire_image(20, 10))

    assert tensor.shape == (3, 10, 20)
    assert tensor.dtype == torch.float32
    assert 0.0 <= float(tensor.min()) and float(tensor.max()) <= 1.0


def test_image_to_tensor_expands_grayscale():
    assert image_to_tensor(np.zeros((6, 8), dtype=np.uint8)).shape == (3, 6, 8)


def test_image_to_tensor_drops_alpha_channel():
    assert image_to_tensor(np.zeros((6, 8, 4), dtype=np.uint8)).shape == (3, 6, 8)


def test_dataset_yields_maskrcnn_targets(dataset_dir, image_size):
    width, height = image_size
    dataset = FireSegmentationDataset(
        images_dir=dataset_dir / 'train',
        annotations=dataset_dir / 'annotations' / 'train_annotations.json',
        class_filter=['fire'])

    assert len(dataset) == 2

    image, target = dataset[0]

    assert image.shape == (3, height, width)
    assert target['boxes'].shape == (1, 4)
    assert target['masks'].shape == (1, height, width)
    assert target['labels'].tolist() == [1]
    assert target['iscrowd'].tolist() == [0]
    assert float(target['area'][0]) > 0

    x1, y1, x2, y2 = target['boxes'][0].tolist()
    assert x2 > x1 and y2 > y1


def test_dataset_skips_unannotated_images_by_default(dataset_dir):
    extra = dataset_dir / 'train' / 'unlabelled.jpg'
    cv2.imwrite(str(extra), np.zeros((48, 64, 3), dtype=np.uint8))

    annotations = dataset_dir / 'annotations' / 'train_annotations.json'
    strict = FireSegmentationDataset(dataset_dir / 'train', annotations)
    permissive = FireSegmentationDataset(dataset_dir / 'train', annotations,
                                         require_annotations=False)

    assert len(strict) == 2
    assert len(permissive) == 3
    assert strict.unannotated == ['unlabelled.jpg']


def test_dataset_without_annotations_yields_empty_targets(dataset_dir, image_size):
    width, height = image_size
    dataset = FireSegmentationDataset(dataset_dir / 'train', annotations=None,
                                      require_annotations=False)

    _, target = dataset[0]

    assert target['boxes'].shape == (0, 4)
    assert target['masks'].shape == (0, height, width)
    assert target['labels'].numel() == 0


def test_horizontal_flip_keeps_boxes_aligned_with_masks(dataset_dir, image_size):
    width, _ = image_size
    dataset = FireSegmentationDataset(
        images_dir=dataset_dir / 'train',
        annotations=dataset_dir / 'annotations' / 'train_annotations.json',
        horizontal_flip_prob=1.0)

    _, target = dataset[0]
    mask = target['masks'][0].numpy().astype(bool)
    box = target['boxes'][0].tolist()

    ys, xs = np.nonzero(mask)
    assert box[0] == pytest.approx(xs.min(), abs=1)
    assert box[2] == pytest.approx(xs.max() + 1, abs=1)
    assert 0 <= box[0] < box[2] <= width


def test_collate_fn_keeps_variable_sized_images_apart(dataset_dir):
    dataset = FireSegmentationDataset(
        images_dir=dataset_dir / 'train',
        annotations=dataset_dir / 'annotations' / 'train_annotations.json')

    images, targets = collate_fn([dataset[0], dataset[1]])

    assert len(images) == 2 and len(targets) == 2
    assert isinstance(images, tuple)
    assert all(isinstance(image, torch.Tensor) for image in images)
