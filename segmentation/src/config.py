"""
Configuration for the Mask R-CNN fire detection model.

Every field can be overridden with an environment variable named
``FIRE_<FIELD>`` (for example ``FIRE_LEARNING_RATE=0.001``) via
:meth:`FireDetectionConfig.from_env`.
"""
from __future__ import annotations

import os
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


def ensure_directories() -> None:
    """Create the project data/weights/output directories if they are missing."""
    for directory in (DATA_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR,
                      ANNOTATIONS_DIR, WEIGHTS_DIR, OUTPUTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def _coerce(value: str, target_type: Any) -> Any:
    """Convert an environment string to the type of the matching config field."""
    if target_type is bool:
        return value.strip().lower() in {'1', 'true', 'yes', 'on'}
    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)
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
            values[f.name] = _coerce(raw, {'str': str, 'int': int,
                                           'float': float, 'bool': bool}.get(f.type, str))
        values.update({k: v for k, v in overrides.items() if v is not None})
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
