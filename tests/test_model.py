"""End-to-end tests for building, training, saving and reloading the model.

These run on CPU with randomly initialised weights and 64x48 images, so they
exercise the real Mask R-CNN code paths without downloading COCO weights.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from conftest import make_fire_image
from src.config import FireDetectionConfig
from src.dataset import FireSegmentationDataset, collate_fn
from src.model import (build_model, compute_validation_loss, create_optimizer,
                       create_scheduler, evaluate, load_checkpoint, load_model,
                       masks_to_numpy, resolve_device, save_checkpoint,
                       set_seed, train_one_epoch)
from src.predictor import FirePredictor, mask_to_polygons


@pytest.fixture(scope='module')
def tiny_config():
    return FireDetectionConfig(
        PRETRAINED_WEIGHTS='none',
        IMAGE_MIN_DIM=64,
        IMAGE_MAX_DIM=96,
        TRAIN_EPOCHS=1,
        BATCH_SIZE=2,
        LEARNING_RATE=0.001,
        DETECTION_MAX_INSTANCES=10,
        DETECTION_MIN_CONFIDENCE=0.05,
        RPN_TRAIN_ANCHORS_PER_IMAGE=32,
    )


@pytest.fixture(scope='module')
def tiny_model(tiny_config):
    torch.manual_seed(0)
    return build_model(tiny_config)


def loader_for(directory, annotations, config, batch_size=1, flip=0.0):
    dataset = FireSegmentationDataset(directory, annotations,
                                      class_filter=['fire'],
                                      horizontal_flip_prob=flip)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=0, collate_fn=collate_fn)


def test_resolve_device():
    assert resolve_device('cpu').type == 'cpu'
    assert resolve_device('auto').type in {'cpu', 'cuda'}


def test_model_has_the_mask_rcnn_components(tiny_model, tiny_config):
    assert hasattr(tiny_model, 'backbone')
    assert hasattr(tiny_model, 'rpn')
    assert hasattr(tiny_model.roi_heads, 'mask_predictor')

    # Heads are resized to our classes, not COCO's 91.
    assert tiny_model.roi_heads.box_predictor.cls_score.out_features == tiny_config.NUM_CLASSES
    assert tiny_model.roi_heads.mask_predictor.mask_fcn_logits.out_channels == \
        tiny_config.NUM_CLASSES
    assert tiny_model.roi_heads.box_roi_pool.output_size[0] == tiny_config.POOL_SIZE
    assert tiny_model.roi_heads.mask_roi_pool.output_size[0] == tiny_config.MASK_POOL_SIZE


def test_anchor_generator_follows_config(tiny_model, tiny_config):
    sizes = tiny_model.rpn.anchor_generator.sizes
    assert tuple(s[0] for s in sizes) == tiny_config.RPN_ANCHOR_SCALES
    assert tuple(tiny_model.rpn.anchor_generator.aspect_ratios[0]) == \
        tiny_config.RPN_ANCHOR_RATIOS


def test_unsupported_backbone_is_rejected():
    with pytest.raises(ValueError, match='backbone'):
        build_model(FireDetectionConfig(BACKBONE='vgg16', PRETRAINED_WEIGHTS='none'))


def test_training_step_produces_all_four_losses(tiny_model, tiny_config, dataset_dir):
    loader = loader_for(dataset_dir / 'train',
                        dataset_dir / 'annotations' / 'train_annotations.json',
                        tiny_config, batch_size=2)
    optimizer = create_optimizer(tiny_model, tiny_config)

    losses = train_one_epoch(tiny_model, optimizer, loader,
                             torch.device('cpu'), epoch=0, config=tiny_config,
                             print_freq=0)

    for component in ('loss_classifier', 'loss_box_reg', 'loss_mask', 'loss_objectness'):
        assert component in losses
        assert np.isfinite(losses[component])
    assert losses['loss'] > 0


def test_scheduler_decays_the_learning_rate(tiny_model, tiny_config):
    config = FireDetectionConfig(PRETRAINED_WEIGHTS='none', LR_STEP_SIZE=1, LR_GAMMA=0.5,
                                 LEARNING_RATE=0.01)
    optimizer = create_optimizer(tiny_model, config)
    scheduler = create_scheduler(optimizer, config)

    scheduler.step()

    assert optimizer.param_groups[0]['lr'] == pytest.approx(0.005)


def test_validation_loss_leaves_the_model_in_eval_mode(tiny_model, tiny_config, dataset_dir):
    loader = loader_for(dataset_dir / 'val',
                        dataset_dir / 'annotations' / 'val_annotations.json',
                        tiny_config)
    tiny_model.eval()

    value = compute_validation_loss(tiny_model, loader, torch.device('cpu'))

    assert np.isfinite(value)
    assert not tiny_model.training


def test_evaluate_returns_the_full_metric_set(tiny_model, tiny_config, dataset_dir):
    loader = loader_for(dataset_dir / 'val',
                        dataset_dir / 'annotations' / 'val_annotations.json',
                        tiny_config)

    metrics = evaluate(tiny_model, loader, torch.device('cpu'), tiny_config)

    assert metrics['images'] == 1
    for key in ('precision', 'recall', 'f1', 'mean_iou', 'ap@0.5'):
        assert 0.0 <= metrics[key] <= 1.0


def test_masks_to_numpy_binarises_soft_masks():
    soft = torch.tensor([[[[0.1, 0.9], [0.6, 0.2]]]])

    binary = masks_to_numpy(soft, threshold=0.5)

    assert binary.shape == (1, 2, 2)
    assert binary.dtype == bool
    assert binary[0].tolist() == [[False, True], [True, False]]


def test_masks_to_numpy_handles_no_detections():
    assert masks_to_numpy(torch.zeros((0, 1, 8, 8))).shape == (0, 8, 8)


def test_checkpoint_round_trip_preserves_predictions(tiny_model, tiny_config, tmp_path):
    path = tmp_path / 'model.pt'
    save_checkpoint(tiny_model, path, tiny_config, epoch=3, metrics={'f1': 0.5})

    assert path.exists()

    restored, restored_config = load_model(path, device='cpu')

    assert restored_config.NUM_CLASSES == tiny_config.NUM_CLASSES
    assert restored_config.IMAGE_MAX_DIM == tiny_config.IMAGE_MAX_DIM
    assert not restored.training

    image = torch.rand(3, 48, 64)
    tiny_model.eval()
    with torch.no_grad():
        expected = tiny_model([image])[0]
        actual = restored([image])[0]

    torch.testing.assert_close(expected['scores'], actual['scores'])


def test_load_model_reports_a_missing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / 'absent.pt', device='cpu')


def test_predictor_output_is_json_serialisable(tiny_model, tiny_config):
    import json

    predictor = FirePredictor(tiny_model, tiny_config, 'cpu', score_threshold=0.0)
    result = predictor.predict(make_fire_image())

    assert result['image_size'] == {'height': 48, 'width': 64}
    assert result['num_detections'] == len(result['detections'])
    assert result['is_fire'] == (result['num_detections'] > 0)
    assert 0.0 <= result['fire_area_ratio'] <= 1.0

    payload = FirePredictor.serializable(result)
    assert 'masks' not in payload
    json.dumps(payload)  # must not raise

    for detection in result['detections']:
        assert set(detection['bounding_box']) == {'x1', 'y1', 'x2', 'y2'}
        assert detection['class_name'] in tiny_config.CLASS_NAMES


def test_predictor_score_threshold_filters_detections(tiny_model, tiny_config):
    image = make_fire_image()
    permissive = FirePredictor(tiny_model, tiny_config, 'cpu', score_threshold=0.0)
    strict = FirePredictor(tiny_model, tiny_config, 'cpu', score_threshold=0.999)

    assert strict.predict(image)['num_detections'] <= \
        permissive.predict(image)['num_detections']


def test_predictor_batch_matches_single(tiny_model, tiny_config):
    predictor = FirePredictor(tiny_model, tiny_config, 'cpu', score_threshold=0.0)
    images = [make_fire_image(), make_fire_image(box=(5, 5, 20, 20))]

    batch = predictor.predict_batch(images)

    assert len(batch) == 2
    assert batch[0]['num_detections'] == predictor.predict(images[0])['num_detections']
    assert predictor.predict_batch([]) == []


def test_overlay_render_keeps_image_shape(tiny_model, tiny_config, tmp_path):
    predictor = FirePredictor(tiny_model, tiny_config, 'cpu', score_threshold=0.0)
    image = make_fire_image()
    result = predictor.predict(image)

    overlay = predictor.render_overlay(image, result)

    assert overlay.shape == image.shape
    assert overlay.dtype == np.uint8

    written = predictor.save_overlay(image, result, tmp_path / 'out' / 'overlay.png')
    assert written.exists()


def test_mask_to_polygons_traces_a_rectangle():
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:30, 5:25] = True

    polygons = mask_to_polygons(mask)

    assert len(polygons) == 1
    assert len(polygons[0]) >= 3
    xs = [point[0] for point in polygons[0]]
    assert min(xs) <= 6 and max(xs) >= 23


def test_mask_to_polygons_on_empty_mask():
    assert mask_to_polygons(np.zeros((10, 10), dtype=bool)) == []


# --------------------------------------------------------------------------- #
# Checkpoint loading failure modes (Pass 2)
# --------------------------------------------------------------------------- #
def test_missing_checkpoint_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / 'absent.pt')


def test_empty_checkpoint_file_is_reported_as_empty(tmp_path):
    path = tmp_path / 'empty.pt'
    path.write_bytes(b'')
    with pytest.raises(ValueError, match='empty'):
        load_checkpoint(path)


def test_directory_instead_of_checkpoint_is_reported(tmp_path):
    directory = tmp_path / 'weights.pt'
    directory.mkdir()
    with pytest.raises(IsADirectoryError):
        load_checkpoint(directory)


def test_unsafe_checkpoint_is_refused_unless_explicitly_trusted(tmp_path, monkeypatch):
    """weights_only=True must not be silently downgraded to arbitrary unpickling."""
    path = tmp_path / 'pickled.pt'
    torch.save({'model_state_dict': {'w': torch.zeros(2)},
                'config': FireDetectionConfig(PRETRAINED_WEIGHTS='none')}, path)

    monkeypatch.delenv('FIRE_TRUST_CHECKPOINT', raising=False)
    with pytest.raises(ValueError, match='FIRE_TRUST_CHECKPOINT'):
        load_checkpoint(path)

    monkeypatch.setenv('FIRE_TRUST_CHECKPOINT', '1')
    assert 'model_state_dict' in load_checkpoint(path)


def test_corrupt_checkpoint_bytes_give_an_actionable_error(tmp_path):
    path = tmp_path / 'corrupt.pt'
    path.write_bytes(b'not a torch archive at all')
    with pytest.raises(ValueError, match='Could not safely load'):
        load_checkpoint(path)


def test_raw_state_dict_is_diagnosed_rather_than_key_erroring(tmp_path, tiny_config):
    """`torch.save(model.state_dict(), ...)` is the classic user mistake."""
    path = tmp_path / 'raw.pt'
    torch.save({'backbone.body.conv1.weight': torch.zeros(2, 2)}, path)

    with pytest.raises(KeyError, match='raw state_dict'):
        load_model(path, device='cpu')


def test_checkpoint_whose_weights_do_not_fit_its_config_is_reported(tmp_path, tiny_config):
    path = tmp_path / 'mismatch.pt'
    torch.save({'model_state_dict': {'nonsense.weight': torch.zeros(3)},
                'config': tiny_config.to_dict()}, path)

    with pytest.raises(RuntimeError, match='do not fit'):
        load_model(path, device='cpu')


# --------------------------------------------------------------------------- #
# Seeding (Pass 2)
# --------------------------------------------------------------------------- #
def test_set_seed_fixes_torch_numpy_and_random():
    import random as _random

    set_seed(1234)
    first = (torch.rand(3).tolist(), np.random.rand(3).tolist(),
             [_random.random() for _ in range(3)])

    set_seed(1234)
    second = (torch.rand(3).tolist(), np.random.rand(3).tolist(),
              [_random.random() for _ in range(3)])

    assert first == second


def test_set_seed_returns_an_independent_generator():
    generator = set_seed(7)
    drawn = torch.rand(4, generator=generator).tolist()

    generator = set_seed(7)
    assert torch.rand(4, generator=generator).tolist() == drawn


def test_set_seed_rejects_seeds_outside_32_bits():
    with pytest.raises(ValueError, match='32 bits'):
        set_seed(2 ** 40)


def test_seeded_flip_augmentation_is_reproducible(dataset_dir):
    def first_mask():
        set_seed(99)
        dataset = FireSegmentationDataset(
            images_dir=dataset_dir / 'train',
            annotations=dataset_dir / 'annotations' / 'train_annotations.json',
            horizontal_flip_prob=0.5)
        return np.stack([dataset[i][1]['masks'].numpy().sum(axis=(1, 2))
                         for i in range(len(dataset))])

    np.testing.assert_array_equal(first_mask(), first_mask())


def test_evaluate_restores_the_previous_module_mode(tiny_config, tiny_model,
                                                    dataset_dir):
    loader = loader_for(dataset_dir / 'val',
                        dataset_dir / 'annotations' / 'val_annotations.json',
                        tiny_config)
    tiny_model.train()
    try:
        evaluate(tiny_model, loader, torch.device('cpu'), tiny_config)
        assert tiny_model.training is True
    finally:
        tiny_model.eval()
