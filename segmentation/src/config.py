"""
Configuration for the Mask R-CNN fire detection model.

Every field can be overridden with an environment variable named
``FIRE_<FIELD>`` (for example ``FIRE_LEARNING_RATE=0.001``) via
:meth:`FireDetectionConfig.from_env`.
"""
from __future__ import annotations

import math
import os
import warnings
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Tuple

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
TRAIN_DIR = DATA_DIR / 'train'
VAL_DIR = DATA_DIR / 'val'
TEST_DIR = DATA_DIR / 'test'
ANNOTATIONS_DIR = DATA_DIR / 'annotations'
WEIGHTS_DIR = PROJECT_ROOT / 'weights'
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'

# Index 0 is always the implicit background class used by torchvision detectors.
CLASS_NAMES: Tuple[str, ...] = ('__background__', 'fire')

ENV_PREFIX = 'FIRE_'

# FIRE_* variables that are read elsewhere and are not configuration fields.
_ENV_ALLOWLIST = frozenset({'FIRE_TRUST_CHECKPOINT'})


def ensure_directories() -> None:
    """Create the project data/weights/output directories if they are missing."""
    for directory in (DATA_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR,
                      ANNOTATIONS_DIR, WEIGHTS_DIR, OUTPUTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


_TRUTHY = {'1', 'true', 'yes', 'on'}
_FALSY = {'0', 'false', 'no', 'off'}


def _coerce(value: str, target_type: Any, variable: str = '') -> Any:
    """Convert an environment string to the type of the matching config field.

    Raises a ``ValueError`` naming the offending variable: the bare
    ``int('fast')`` traceback gives no clue which ``FIRE_*`` setting is wrong.
    """
    where = f'{variable}=' if variable else ''
    text = value.strip()
    if target_type is bool:
        lowered = text.lower()
        if lowered in _TRUTHY:
            return True
        if lowered in _FALSY:
            return False
        raise ValueError(
            f'{where}{value!r} is not a boolean; use one of '
            f'{sorted(_TRUTHY | _FALSY)}')
    if target_type is int:
        try:
            return int(text)
        except ValueError:
            raise ValueError(f'{where}{value!r} is not an integer') from None
    if target_type is float:
        try:
            parsed = float(text)
        except ValueError:
            raise ValueError(f'{where}{value!r} is not a number') from None
        if not math.isfinite(parsed):
            raise ValueError(f'{where}{value!r} must be a finite number')
        return parsed
    return value


@dataclass
class FireDetectionConfig:
    """Hyper-parameters for building, training and running the model.

    Field names are upper case so that ``config.NUM_CLASSES`` style access
    reads the same way as the documented Mask R-CNN configuration.
    """

    # Identity
    NAME: str = 'fire_detection'

    # Input image handling (the detector rescales inputs into this range)
    IMAGE_MIN_DIM: int = 256
    IMAGE_MAX_DIM: int = 512

    # Backbone
    BACKBONE: str = 'resnet50'
    # 'coco' | 'imagenet' | 'none' -- which pre-trained weights to start from
    PRETRAINED_WEIGHTS: str = 'coco'
    # How many of the 5 ResNet stages stay trainable (0 = frozen backbone)
    TRAINABLE_BACKBONE_LAYERS: int = 3

    # Classes (background + fire)
    NUM_CLASSES: int = 2

    # Optimisation
    LEARNING_RATE: float = 0.005
    LEARNING_MOMENTUM: float = 0.9
    WEIGHT_DECAY: float = 0.0005
    LR_STEP_SIZE: int = 10
    LR_GAMMA: float = 0.1
    GRAD_CLIP_NORM: float = 10.0

    # Training schedule
    TRAIN_EPOCHS: int = 30
    BATCH_SIZE: int = 2
    NUM_WORKERS: int = 0
    SEED: int = 42

    # Augmentation
    HORIZONTAL_FLIP_PROB: float = 0.5

    # Region proposal network
    RPN_ANCHOR_SCALES: Tuple[int, ...] = (32, 64, 128, 256, 512)
    RPN_ANCHOR_RATIOS: Tuple[float, ...] = (0.5, 1.0, 2.0)
    RPN_NMS_THRESHOLD: float = 0.7
    RPN_TRAIN_ANCHORS_PER_IMAGE: int = 256

    # ROI Align
    POOL_SIZE: int = 7
    MASK_POOL_SIZE: int = 14
    MASK_HIDDEN_LAYER: int = 256

    # Detection / post-processing
    DETECTION_MAX_INSTANCES: int = 100
    DETECTION_MIN_CONFIDENCE: float = 0.7
    DETECTION_NMS_THRESHOLD: float = 0.3
    MASK_BINARY_THRESHOLD: float = 0.5

    # Evaluation
    EVAL_IOU_THRESHOLD: float = 0.5

    # Runtime
    DEVICE: str = 'auto'  # 'auto' | 'cpu' | 'cuda'

    # Artefacts
    MODEL_PATH: str = str(WEIGHTS_DIR / 'fire_detection_model.pt')
    CLASS_NAMES: Tuple[str, ...] = CLASS_NAMES

    # Fields that must be strictly positive, and fields that are probabilities.
    _POSITIVE_FIELDS = ('IMAGE_MIN_DIM', 'IMAGE_MAX_DIM', 'NUM_CLASSES',
                        'LEARNING_RATE', 'LR_STEP_SIZE', 'TRAIN_EPOCHS',
                        'BATCH_SIZE', 'POOL_SIZE', 'MASK_POOL_SIZE',
                        'MASK_HIDDEN_LAYER', 'DETECTION_MAX_INSTANCES',
                        'RPN_TRAIN_ANCHORS_PER_IMAGE')
    _UNIT_INTERVAL_FIELDS = ('LEARNING_MOMENTUM', 'HORIZONTAL_FLIP_PROB',
                             'RPN_NMS_THRESHOLD', 'DETECTION_MIN_CONFIDENCE',
                             'DETECTION_NMS_THRESHOLD', 'MASK_BINARY_THRESHOLD',
                             'LR_GAMMA')
    _NON_NEGATIVE_FIELDS = ('WEIGHT_DECAY', 'NUM_WORKERS', 'GRAD_CLIP_NORM', 'SEED')

    def __post_init__(self) -> None:
        # Tuples survive round-tripping through JSON/dict as lists; normalise.
        self.RPN_ANCHOR_SCALES = tuple(int(s) for s in self.RPN_ANCHOR_SCALES)
        self.RPN_ANCHOR_RATIOS = tuple(float(r) for r in self.RPN_ANCHOR_RATIOS)
        self.CLASS_NAMES = tuple(self.CLASS_NAMES)
        if self.NUM_CLASSES != len(self.CLASS_NAMES):
            raise ValueError(
                f'NUM_CLASSES ({self.NUM_CLASSES}) must match the number of '
                f'CLASS_NAMES ({len(self.CLASS_NAMES)}: {self.CLASS_NAMES})'
            )
        if self.IMAGE_MIN_DIM > self.IMAGE_MAX_DIM:
            raise ValueError('IMAGE_MIN_DIM cannot be larger than IMAGE_MAX_DIM')
        if self.PRETRAINED_WEIGHTS not in {'coco', 'imagenet', 'none'}:
            raise ValueError(
                "PRETRAINED_WEIGHTS must be one of 'coco', 'imagenet', 'none'; "
                f'got {self.PRETRAINED_WEIGHTS!r}'
            )
        if self.DEVICE not in {'auto', 'cpu', 'cuda'}:
            raise ValueError(
                f"DEVICE must be one of 'auto', 'cpu', 'cuda'; got {self.DEVICE!r}")
        if not 0 <= self.TRAINABLE_BACKBONE_LAYERS <= 5:
            raise ValueError(
                'TRAINABLE_BACKBONE_LAYERS must be between 0 and 5; got '
                f'{self.TRAINABLE_BACKBONE_LAYERS}')
        if not 0.0 < self.EVAL_IOU_THRESHOLD <= 1.0:
            raise ValueError(
                f'EVAL_IOU_THRESHOLD must be in (0, 1]; got {self.EVAL_IOU_THRESHOLD}')
        if not self.RPN_ANCHOR_SCALES or not self.RPN_ANCHOR_RATIOS:
            raise ValueError('RPN_ANCHOR_SCALES and RPN_ANCHOR_RATIOS cannot be empty')
        if any(s <= 0 for s in self.RPN_ANCHOR_SCALES):
            raise ValueError(
                f'RPN_ANCHOR_SCALES must all be positive; got {self.RPN_ANCHOR_SCALES}')
        if any(r <= 0 for r in self.RPN_ANCHOR_RATIOS):
            raise ValueError(
                f'RPN_ANCHOR_RATIOS must all be positive; got {self.RPN_ANCHOR_RATIOS}')

        for name in self._POSITIVE_FIELDS:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be a positive number; got {value!r}')
        for name in self._NON_NEGATIVE_FIELDS:
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be zero or greater; got {value!r}')
        for name in self._UNIT_INTERVAL_FIELDS:
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f'{name} must be in [0, 1]; got {value!r}')

    @classmethod
    def from_env(cls, **overrides: Any) -> 'FireDetectionConfig':
        """Build a config from defaults, ``FIRE_*`` env vars and explicit kwargs."""
        values: Dict[str, Any] = {}
        for f in fields(cls):
            raw = os.environ.get(ENV_PREFIX + f.name)
            if raw is None or raw == '':
                continue
            if f.type in ('Tuple[int, ...]', 'Tuple[float, ...]'):
                continue  # sequences are not configured through the environment
            values[f.name] = _coerce(
                raw,
                {'str': str, 'int': int, 'float': float, 'bool': bool}.get(f.type, str),
                variable=ENV_PREFIX + f.name)

        unknown = sorted(name for name in os.environ
                         if name.startswith(ENV_PREFIX)
                         and name[len(ENV_PREFIX):] not in {f.name for f in fields(cls)}
                         and name not in _ENV_ALLOWLIST)
        if unknown:
            warnings.warn(
                f'Ignoring unrecognised {ENV_PREFIX}* variable(s): '
                f'{", ".join(unknown)}. Check for typos - these have no effect.',
                RuntimeWarning, stacklevel=2)

        values.update({k: v for k, v in overrides.items() if v is not None})
        unexpected = sorted(set(values) - {f.name for f in fields(cls)})
        if unexpected:
            raise TypeError(
                f'Unknown configuration field(s): {", ".join(unexpected)}')
        return cls(**values)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FireDetectionConfig':
        """Rebuild a config from a serialised dict, ignoring unknown keys."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain, JSON-friendly dict."""
        data = asdict(self)
        for key in ('RPN_ANCHOR_SCALES', 'RPN_ANCHOR_RATIOS', 'CLASS_NAMES'):
            data[key] = list(data[key])
        return data

    def display(self) -> None:
        """Print the configuration."""
        print(f'{self.NAME} configuration:')
        for key, value in self.to_dict().items():
            print(f'  {key}: {value}')


@dataclass
class InferenceConfig(FireDetectionConfig):
    """Configuration tuned for inference: lower threshold, more instances."""

    BATCH_SIZE: int = 1
    DETECTION_MAX_INSTANCES: int = 500
    DETECTION_MIN_CONFIDENCE: float = 0.5
