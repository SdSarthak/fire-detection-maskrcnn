"""Tests for the training CLI: argument validation, split hygiene, history.

These exercise the orchestration logic in ``train.py`` directly rather than
running a full fine-tune, so they stay fast and deterministic.
"""
from __future__ import annotations

import json

import pytest

import train
from src.config import FireDetectionConfig
from src.dataset import FireSegmentationDataset


# --------------------------------------------------------------------------- #
# CLI argument validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('argv', [
    ['--epochs', '0'],
    ['--epochs', '-3'],
    ['--batch-size', '0'],
    ['--batch-size', '-1'],
    ['--num-workers', '-1'],
    ['--lr', '0'],
    ['--lr', '-0.5'],
    ['--seed', '-1'],
])
def test_invalid_cli_values_exit_with_a_named_flag(argv, capsys):
    with pytest.raises(SystemExit) as excinfo:
        train.parse_args(argv)

    assert excinfo.value.code == 2
    assert argv[0] in capsys.readouterr().err


def test_valid_cli_values_are_accepted():
    args = train.parse_args(['--epochs', '2', '--batch-size', '1', '--lr', '0.01'])
    assert (args.epochs, args.batch_size, args.learning_rate) == (2, 1, 0.01)


def test_unknown_device_is_rejected_by_argparse():
    with pytest.raises(SystemExit):
        train.parse_args(['--device', 'tpu'])


def test_build_config_layers_cli_over_defaults():
    config = train.build_config(train.parse_args(['--epochs', '4', '--lr', '0.002']))
    assert config.TRAIN_EPOCHS == 4
    assert config.LEARNING_RATE == pytest.approx(0.002)
    # Untouched flags fall back to the dataclass defaults.
    assert config.BATCH_SIZE == FireDetectionConfig().BATCH_SIZE


# --------------------------------------------------------------------------- #
# Train/val leakage
# --------------------------------------------------------------------------- #
def test_overlapping_splits_are_detected(dataset_dir):
    annotations = dataset_dir / 'annotations' / 'train_annotations.json'
    train_dataset = FireSegmentationDataset(dataset_dir / 'train', annotations)
    same_dataset = FireSegmentationDataset(dataset_dir / 'train', annotations)

    assert train.check_split_leakage(train_dataset, same_dataset) == [
        'fire_train_1.jpg', 'fire_train_2.jpg']


def test_disjoint_splits_report_no_leakage(dataset_dir):
    train_dataset = FireSegmentationDataset(
        dataset_dir / 'train', dataset_dir / 'annotations' / 'train_annotations.json')
    val_dataset = FireSegmentationDataset(
        dataset_dir / 'val', dataset_dir / 'annotations' / 'val_annotations.json')

    assert train.check_split_leakage(train_dataset, val_dataset) == []


def test_build_dataloaders_refuses_a_leaking_val_split(dataset_dir, tmp_path):
    # Point the val split at the training images with the training labels.
    val_dir = dataset_dir / 'val'
    for image in (dataset_dir / 'train').iterdir():
        (val_dir / image.name).write_bytes(image.read_bytes())
    (dataset_dir / 'annotations' / 'val_annotations.json').write_text(
        (dataset_dir / 'annotations' / 'train_annotations.json').read_text(),
        encoding='utf-8')

    args = train.parse_args([
        '--data-dir', str(dataset_dir),
        '--annotations-dir', str(dataset_dir / 'annotations'),
    ])

    with pytest.raises(SystemExit, match='both the train and val splits'):
        train.build_dataloaders(args, FireDetectionConfig(NUM_WORKERS=0))


def test_missing_data_directory_exits_with_a_clear_message(tmp_path):
    args = train.parse_args(['--data-dir', str(tmp_path / 'nope'),
                             '--annotations-dir', str(tmp_path)])
    with pytest.raises(SystemExit, match='Data directory does not exist'):
        train.build_dataloaders(args, FireDetectionConfig())


def test_empty_training_split_exits_with_instructions(dataset_dir):
    for image in (dataset_dir / 'train').iterdir():
        image.unlink()
    args = train.parse_args(['--data-dir', str(dataset_dir),
                             '--annotations-dir', str(dataset_dir / 'annotations')])

    with pytest.raises(SystemExit, match='No annotated training images'):
        train.build_dataloaders(args, FireDetectionConfig())


# --------------------------------------------------------------------------- #
# History persistence
# --------------------------------------------------------------------------- #
def test_history_is_written_atomically(tmp_path):
    path = tmp_path / 'history.json'

    train.write_history(path, [{'epoch': 0, 'loss': 1.0}])
    train.write_history(path, [{'epoch': 0, 'loss': 1.0}, {'epoch': 1, 'loss': 0.5}])

    assert json.loads(path.read_text(encoding='utf-8')) == [
        {'epoch': 0, 'loss': 1.0}, {'epoch': 1, 'loss': 0.5}]
    assert not list(tmp_path.glob('*.tmp'))


# --------------------------------------------------------------------------- #
# Best-checkpoint selection
# --------------------------------------------------------------------------- #
def test_run_without_a_val_split_always_writes_a_fresh_checkpoint(dataset_dir,
                                                                  tmp_path):
    """Regression: the run used to finish having saved nothing.

    Without a validation split the tracked score is ``-training_loss``. The
    initial best was ``-1.0``, so an untrained Mask R-CNN (loss ~6) never beat
    it, and the "save the final weights" fallback was skipped whenever a
    checkpoint file already existed - leaving a stale model on disk while the
    log said training completed.
    """
    from src.model import load_checkpoint

    for annotation in (dataset_dir / 'annotations').glob('val_*.json'):
        annotation.unlink()

    checkpoint = tmp_path / 'model.pt'
    checkpoint.write_bytes(b'stale checkpoint from a previous run')

    exit_code = train.main([
        '--data-dir', str(dataset_dir),
        '--annotations-dir', str(dataset_dir / 'annotations'),
        '--pretrained', 'none',
        '--device', 'cpu',
        '--epochs', '1',
        '--batch-size', '2',
        '--output', str(checkpoint),
        '--history', str(tmp_path / 'history.json'),
    ])

    assert exit_code == 0
    payload = load_checkpoint(checkpoint)
    assert 'model_state_dict' in payload
    assert payload['epoch'] == 0

    history = json.loads((tmp_path / 'history.json').read_text(encoding='utf-8'))
    assert len(history) == 1 and history[0]['epoch'] == 0
