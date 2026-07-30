"""Tests for configuration defaults, environment overrides and validation."""
from __future__ import annotations

import pytest

from src.config import FireDetectionConfig, InferenceConfig


def test_defaults_describe_a_binary_fire_detector():
    config = FireDetectionConfig()
    assert config.NUM_CLASSES == 2
    assert config.CLASS_NAMES == ('__background__', 'fire')
    assert config.MODEL_PATH.endswith('.pt')


def test_environment_overrides_are_typed(monkeypatch):
    monkeypatch.setenv('FIRE_LEARNING_RATE', '0.0001')
    monkeypatch.setenv('FIRE_TRAIN_EPOCHS', '3')
    monkeypatch.setenv('FIRE_DEVICE', 'cpu')

    config = FireDetectionConfig.from_env()

    assert config.LEARNING_RATE == pytest.approx(0.0001)
    assert isinstance(config.TRAIN_EPOCHS, int) and config.TRAIN_EPOCHS == 3
    assert config.DEVICE == 'cpu'


def test_explicit_overrides_beat_the_environment(monkeypatch):
    monkeypatch.setenv('FIRE_TRAIN_EPOCHS', '3')
    assert FireDetectionConfig.from_env(TRAIN_EPOCHS=7).TRAIN_EPOCHS == 7


def test_none_overrides_are_ignored(monkeypatch):
    """CLI flags that were not supplied must not clobber env/defaults."""
    monkeypatch.setenv('FIRE_TRAIN_EPOCHS', '4')
    assert FireDetectionConfig.from_env(TRAIN_EPOCHS=None).TRAIN_EPOCHS == 4


def test_empty_environment_value_is_ignored(monkeypatch):
    monkeypatch.setenv('FIRE_DEVICE', '')
    assert FireDetectionConfig.from_env().DEVICE == 'auto'


def test_round_trips_through_a_dict():
    original = FireDetectionConfig(TRAIN_EPOCHS=5, LEARNING_RATE=0.002)
    restored = FireDetectionConfig.from_dict(original.to_dict())

    assert restored == original
    assert restored.RPN_ANCHOR_SCALES == original.RPN_ANCHOR_SCALES


def test_from_dict_ignores_unknown_keys():
    config = FireDetectionConfig.from_dict({'TRAIN_EPOCHS': 2, 'LEGACY_OPTION': 'x'})
    assert config.TRAIN_EPOCHS == 2


def test_to_dict_is_json_friendly():
    data = FireDetectionConfig().to_dict()
    assert isinstance(data['RPN_ANCHOR_SCALES'], list)
    assert isinstance(data['CLASS_NAMES'], list)


def test_class_count_must_match_class_names():
    with pytest.raises(ValueError, match='NUM_CLASSES'):
        FireDetectionConfig(NUM_CLASSES=3)


def test_image_dimensions_must_be_ordered():
    with pytest.raises(ValueError, match='IMAGE_MIN_DIM'):
        FireDetectionConfig(IMAGE_MIN_DIM=800, IMAGE_MAX_DIM=512)


def test_unknown_pretrained_source_is_rejected():
    with pytest.raises(ValueError, match='PRETRAINED_WEIGHTS'):
        FireDetectionConfig(PRETRAINED_WEIGHTS='magic')


def test_inference_config_lowers_the_threshold():
    assert InferenceConfig().DETECTION_MIN_CONFIDENCE < \
        FireDetectionConfig().DETECTION_MIN_CONFIDENCE
    assert InferenceConfig().BATCH_SIZE == 1


def test_display_prints_every_field(capsys):
    FireDetectionConfig().display()
    output = capsys.readouterr().out
    assert 'NUM_CLASSES: 2' in output
    assert 'LEARNING_RATE' in output
