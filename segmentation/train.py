"""
Training pipeline for Mask R-CNN Fire Detection
Orchestrates data loading, model building, and training
"""
import numpy as np
import tensorflow as tf
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import FireDetectionConfig, PROJECT_ROOT, WEIGHTS_DIR, OUTPUTS_DIR
from src.dataset import LoadDataset, LoadBackbone
from src.model import MaskRCNNBuilder, ModelTrainer


def download_pretrained_weights():
    """Download pre-trained weights if not available"""
    import os
    
    weights_file = WEIGHTS_DIR / 'mask_rcnn_coco.h5'
    
    if not weights_file.exists():
        print("Downloading pre-trained Mask R-CNN weights...")
        print("Note: Weights should be downloaded from:")
        print("https://github.com/matterport/Mask_RCNN/releases/download/v2.1/mask_rcnn_coco.h5")
        print(f"And placed in: {WEIGHTS_DIR}")


def create_data_generators(config):
    """
    Create training and validation data generators
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (train_generator, val_generator)
    """
    # Simple data augmentation
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        fill_mode='nearest'
    )
    
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0/255.0
    )
    
    return train_datagen, val_datagen


def main():
    """Main training pipeline"""
    
    print("=" * 60)
    print("Mask R-CNN Fire Detection - Training Pipeline")
    print("=" * 60)
    
    # Configuration
    config = FireDetectionConfig()
    config.display()
    
    print("\n" + "=" * 60)
    print("Step 1: Download Pre-trained Weights")
    print("=" * 60)
    download_pretrained_weights()
    
    print("\n" + "=" * 60)
    print("Step 2: Load Dataset")
    print("=" * 60)
    
    dataset_loader = LoadDataset(
        data_dir=PROJECT_ROOT / 'data',
        annotations_dir=PROJECT_ROOT / 'data' / 'annotations'
    )
    
    print("Loading training dataset...")
    train_dataset = dataset_loader.load_images_from_dir(
        'train',
        annotation_file='train_annotations.json'
    )
    print(f"  ✓ Training samples: {len(train_dataset)}")
    
    print("Loading validation dataset...")
    val_dataset = dataset_loader.load_images_from_dir(
        'val',
        annotation_file='val_annotations.json'
    )
    print(f"  ✓ Validation samples: {len(val_dataset)}")
    
    print("\n" + "=" * 60)
    print("Step 3: Load Backbone Weights")
    print("=" * 60)
    
    backbone_loader = LoadBackbone(weights_path=str(WEIGHTS_DIR / 'mask_rcnn_coco.h5'))
    print("Backbone loader initialized")
    
    print("\n" + "=" * 60)
    print("Step 4: Build Model")
    print("=" * 60)
    
    builder = MaskRCNNBuilder(config)
    model = builder.build_complete_model()
    print("  ✓ Model architecture created")
    
    # Compile model
    model = builder.compile_model(model)
    print("  ✓ Model compiled")
    
    # Print model summary
    print("\nModel Summary:")
    model.summary()
    
    print("\n" + "=" * 60)
    print("Step 5: Create Data Generators")
    print("=" * 60)
    
    train_gen, val_gen = create_data_generators(config)
    print("  ✓ Data generators created")
    
    print("\n" + "=" * 60)
    print("Step 6: Train Model")
    print("=" * 60)
    
    trainer = ModelTrainer(model, config)
    
    print("Starting training...")
    print(f"  Epochs: {config.TRAIN_EPOCHS}")
    print(f"  Steps per epoch: {config.STEPS_PER_EPOCH}")
    
    # For demonstration, we'll create dummy data
    # In production, this would use actual data generators
    print("\n  NOTE: Using dummy data for demonstration")
    print("  To train with real data:")
    print("  1. Place training images in data/train/")
    print("  2. Place validation images in data/val/")
    print("  3. Create annotation JSON files using VGG Annotator")
    
    # Create dummy training data
    dummy_images = np.random.rand(5, 256, 256, 3)
    dummy_classes = np.random.randint(0, config.NUM_CLASSES, (5, config.NUM_CLASSES))
    dummy_bboxes = np.random.rand(5, config.NUM_CLASSES * 4)
    dummy_masks = np.random.rand(5, 112, 112, config.NUM_CLASSES)
    
    print("\nTraining model on dummy data (for demonstration)...")
    history = trainer.train(
        train_generator=(dummy_images, (dummy_classes, dummy_bboxes, dummy_masks)),
        val_generator=None,
        epochs=2  # Use 2 epochs for demonstration
    )
    
    print("\n" + "=" * 60)
    print("Step 7: Save Model")
    print("=" * 60)
    
    trainer.save_model(config.MODEL_PATH)
    print(f"  ✓ Model saved to {config.MODEL_PATH}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nModel Location: {config.MODEL_PATH}")
    print(f"Outputs Directory: {OUTPUTS_DIR}")
    
    return model


if __name__ == "__main__":
    model = main()
