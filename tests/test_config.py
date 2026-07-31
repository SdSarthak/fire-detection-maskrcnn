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


# --------------------------------------------------------------------------- #
# Boundary validation (Pass 2)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('field,value', [
    ('TRAIN_EPOCHS', 0),
    ('BATCH_SIZE', 0),
    ('BATCH_SIZE', -2),
    ('LEARNING_RATE', 0.0),
    ('LEARNING_RATE', -0.1),
    ('IMAGE_MIN_DIM', 0),
    ('LR_STEP_SIZE', 0),
    ('DETECTION_MAX_INSTANCES', 0),
])
def test_non_positive_hyperparameters_are_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        FireDetectionConfig(**{field: value})


@pytest.mark.parametrize('field,value', [
    ('NUM_WORKERS', -1),
    ('WEIGHT_DECAY', -0.001),
    ('GRAD_CLIP_NORM', -1.0),
])
def test_negative_non_negative_fields_are_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        FireDetectionConfig(**{field: value})


@pytest.mark.parametrize('field,value', [
    ('DETECTION_MIN_CONFIDENCE', 1.5),
    ('DETECTION_MIN_CONFIDENCE', -0.1),
    ('MASK_BINARY_THRESHOLD', 2.0),
    ('HORIZONTAL_FLIP_PROB', 1.2),
    ('RPN_NMS_THRESHOLD', -0.5),
    ('LEARNING_MOMENTUM', 1.01),
])
def test_probabilities_must_lie_in_the_unit_interval(field, value):
    with pytest.raises(ValueError, match=field):
        FireDetectionConfig(**{field: value})


def test_nan_learning_rate_is_rejected():
    with pytest.raises(ValueError, match='LEARNING_RATE'):
        FireDetectionConfig(LEARNING_RATE=float('nan'))


def test_infinite_learning_rate_is_rejected():
    with pytest.raises(ValueError, match='LEARNING_RATE'):
        FireDetectionConfig(LEARNING_RATE=float('inf'))


def test_eval_iou_threshold_of_zero_is_rejected():
    with pytest.raises(ValueError, match='EVAL_IOU_THRESHOLD'):
        FireDetectionConfig(EVAL_IOU_THRESHOLD=0.0)


def test_unknown_device_is_rejected():
    with pytest.raises(ValueError, match='DEVICE'):
        FireDetectionConfig(DEVICE='tpu')


def test_trainable_backbone_layers_are_bounded():
    with pytest.raises(ValueError, match='TRAINABLE_BACKBONE_LAYERS'):
        FireDetectionConfig(TRAINABLE_BACKBONE_LAYERS=9)


def test_empty_anchor_scales_are_rejected():
    with pytest.raises(ValueError, match='cannot be empty'):
        FireDetectionConfig(RPN_ANCHOR_SCALES=())


def test_non_positive_anchor_ratios_are_rejected():
    with pytest.raises(ValueError, match='RPN_ANCHOR_RATIOS'):
        FireDetectionConfig(RPN_ANCHOR_RATIOS=(1.0, 0.0))


# --------------------------------------------------------------------------- #
# Environment parsing errors name the variable (Pass 2)
# --------------------------------------------------------------------------- #
def test_unparseable_int_env_var_names_itself(monkeypatch):
    monkeypatch.setenv('FIRE_TRAIN_EPOCHS', 'many')
    with pytest.raises(ValueError, match='FIRE_TRAIN_EPOCHS'):
        FireDetectionConfig.from_env()


def test_unparseable_float_env_var_names_itself(monkeypatch):
    monkeypatch.setenv('FIRE_LEARNING_RATE', 'fast')
    with pytest.raises(ValueError, match='FIRE_LEARNING_RATE'):
        FireDetectionConfig.from_env()


def test_non_finite_float_env_var_is_rejected(monkeypatch):
    monkeypatch.setenv('FIRE_LEARNING_RATE', 'nan')
    with pytest.raises(ValueError, match='finite'):
        FireDetectionConfig.from_env()


def test_out_of_range_env_var_is_caught_by_post_init(monkeypatch):
    monkeypatch.setenv('FIRE_DETECTION_MIN_CONFIDENCE', '5')
    with pytest.raises(ValueError, match='DETECTION_MIN_CONFIDENCE'):
        FireDetectionConfig.from_env()


def test_misspelled_env_var_warns_instead_of_being_silently_ignored(monkeypatch):
    monkeypatch.setenv('FIRE_LEARNIGN_RATE', '0.01')
    with pytest.warns(RuntimeWarning, match='FIRE_LEARNIGN_RATE'):
        FireDetectionConfig.from_env()


def test_trust_checkpoint_env_var_is_not_reported_as_a_typo(monkeypatch):
    import warnings as _warnings
    monkeypatch.setenv('FIRE_TRUST_CHECKPOINT', '1')
    with _warnings.catch_warnings():
        _warnings.simplefilter('error')
        FireDetectionConfig.from_env()


def test_from_env_rejects_unknown_override_keys():
    with pytest.raises(TypeError, match='NOT_A_FIELD'):
        FireDetectionConfig.from_env(NOT_A_FIELD=1)


def test_inference_config_still_validates():
    with pytest.raises(ValueError, match='DETECTION_MIN_CONFIDENCE'):
        InferenceConfig(DETECTION_MIN_CONFIDENCE=-1.0)
