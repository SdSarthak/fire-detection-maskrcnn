"""
Configuration for Mask R-CNN Fire Detection Model
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
TRAIN_DIR = DATA_DIR / 'train'
VAL_DIR = DATA_DIR / 'val'
TEST_DIR = DATA_DIR / 'test'
ANNOTATIONS_DIR = DATA_DIR / 'annotations'
WEIGHTS_DIR = PROJECT_ROOT / 'weights'
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'

# Create directories if they don't exist
for directory in [DATA_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR, ANNOTATIONS_DIR, WEIGHTS_DIR, OUTPUTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model Configuration
class FireDetectionConfig:
    """Configuration for Mask R-CNN model training"""
    
    # Model configuration
    NAME = "fire_detection"
    
    # Image settings
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_SCALE = 2.0
    
    # Backbone architecture
    BACKBONE = "resnet50"
    
    # Training parameters
    NUM_CLASSES = 2  # Background + Fire
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0001
    
    # Training schedule
    TRAIN_EPOCHS = 50
    VALIDATION_STEPS = 10
    STEPS_PER_EPOCH = 100
    
    # RPN anchor scales
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]
    
    # Region proposal network
    RPN_NMS_THRESHOLD = 0.7
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256
    
    # ROI Align
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14
    
    # Detection settings
    DETECTION_MAX_INSTANCES = 100
    DETECTION_MIN_CONFIDENCE = 0.7
    DETECTION_NMS_THRESHOLD = 0.3
    
    # Backbone weights
    PRETRAINED_WEIGHTS = str(WEIGHTS_DIR / 'mask_rcnn_coco.h5')
    BACKBONE_WEIGHTS = str(WEIGHTS_DIR / 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
    
    # Model save path
    MODEL_PATH = str(WEIGHTS_DIR / 'fire_detection_model.h5')
    
    def __init__(self):
        """Initialize configuration"""
        pass
    
    def display(self):
        """Print configuration"""
        print("Fire Detection Config:")
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                print(f"  {key}: {value}")


class InferenceConfig(FireDetectionConfig):
    """Configuration for inference"""
    BATCH_SIZE = 1
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MAX_INSTANCES = 500
    DETECTION_MIN_CONFIDENCE = 0.5
