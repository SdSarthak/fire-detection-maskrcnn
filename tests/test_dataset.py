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


# --------------------------------------------------------------------------- #
# image_to_tensor edge cases (Pass 2)
# --------------------------------------------------------------------------- #
def test_very_dark_uint8_image_is_still_scaled_by_255():
    """Keying normalisation off ``array.max()`` breaks on near-black frames.

    A night-time frame whose brightest pixel is 1 must become 1/255, not 1.0 -
    otherwise the detector sees a pixel 255x brighter than it really is.
    """
    dark = np.zeros((4, 4, 3), dtype=np.uint8)
    dark[0, 0] = 1

    tensor = image_to_tensor(dark)

    assert float(tensor.max()) == pytest.approx(1.0 / 255.0)


def test_all_black_uint8_image_stays_zero():
    tensor = image_to_tensor(np.zeros((4, 4, 3), dtype=np.uint8))
    assert float(tensor.max()) == 0.0
    assert tensor.shape == (3, 4, 4)


def test_uint8_white_maps_to_one():
    tensor = image_to_tensor(np.full((2, 2, 3), 255, dtype=np.uint8))
    assert float(tensor.min()) == pytest.approx(1.0)


def test_single_channel_hwc_image_is_expanded_to_three():
    tensor = image_to_tensor(np.zeros((5, 7, 1), dtype=np.uint8))
    assert tensor.shape == (3, 5, 7)


def test_empty_image_raises_a_clear_error():
    with pytest.raises(ValueError, match='empty image'):
        image_to_tensor(np.zeros((0, 0, 3), dtype=np.uint8))


def test_unsupported_channel_count_raises():
    with pytest.raises(ValueError, match='channel count'):
        image_to_tensor(np.zeros((4, 4, 5), dtype=np.uint8))


def test_five_dimensional_input_raises():
    with pytest.raises(ValueError, match='HW or HWC'):
        image_to_tensor(np.zeros((2, 4, 4, 3), dtype=np.uint8))


def test_float_image_with_nan_raises():
    array = np.zeros((3, 3, 3), dtype=np.float32)
    array[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match='NaN'):
        image_to_tensor(array)


def test_float_image_already_in_unit_range_is_left_alone():
    array = np.full((3, 3, 3), 0.5, dtype=np.float32)
    assert float(image_to_tensor(array).max()) == pytest.approx(0.5)


def test_out_of_range_float_pixels_are_clipped_into_unit_range():
    array = np.full((3, 3, 3), -2.0, dtype=np.float32)
    tensor = image_to_tensor(array)
    assert float(tensor.min()) == 0.0


# --------------------------------------------------------------------------- #
# Dataset argument validation (Pass 2)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('prob', [-0.1, 1.5])
def test_dataset_rejects_out_of_range_flip_probability(dataset_dir, prob):
    with pytest.raises(ValueError, match='horizontal_flip_prob'):
        FireSegmentationDataset(dataset_dir / 'train', horizontal_flip_prob=prob,
                                require_annotations=False)


def test_dataset_rejects_background_label(dataset_dir):
    with pytest.raises(ValueError, match='reserved for background'):
        FireSegmentationDataset(dataset_dir / 'train', label=0,
                                require_annotations=False)


def test_dataset_warns_about_duplicate_basenames(dataset_dir):
    nested = dataset_dir / 'train' / 'nested'
    nested.mkdir()
    cv2.imwrite(str(nested / 'fire_train_1.jpg'),
                np.zeros((48, 64, 3), dtype=np.uint8))

    with pytest.warns(RuntimeWarning, match='duplicate image basename'):
        FireSegmentationDataset(dataset_dir / 'train', require_annotations=False)
