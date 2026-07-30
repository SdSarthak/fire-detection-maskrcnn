"""
Inference pipeline for Mask R-CNN Fire Detection
Makes predictions and visualizes results
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import FireDetectionConfig, PROJECT_ROOT, OUTPUTS_DIR
from src.model import ModelInference
from src.dataset import LoadDataset


def plot_results(image, predictions, output_path=None):
    """
    Plot image with predictions
    
    Args:
        image: Input image
        predictions: Model predictions (class, bbox, mask)
        output_path: Path to save visualization
    """
    class_logits, bbox_output, mask_output = predictions
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Class predictions
    class_pred = class_logits[0]
    axes[1].imshow(image)
    axes[1].set_title(f"Predicted Class: {'Fire' if np.argmax(class_pred) == 1 else 'Background'}")
    axes[1].axis('off')
    
    # Mask prediction
    mask_pred = mask_output[0]
    axes[2].imshow(image)
    if mask_pred.shape[2] > 1:
        # Show fire class mask
        axes[2].imshow(mask_pred[:, :, 1], alpha=0.5, cmap='hot')
    axes[2].set_title("Predicted Mask")
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved visualization to {output_path}")
    
    plt.show()


def run_inference_on_test_images(model_path, config):
    """
    Run inference on test images
    
    Args:
        model_path: Path to trained model
        config: Configuration object
    """
    print("=" * 60)
    print("Mask R-CNN Fire Detection - Inference Pipeline")
    print("=" * 60)
    
    print("\nStep 1: Load Model")
    print("=" * 60)
    
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using train.py")
        return
    
    try:
        inference = ModelInference(model_path, config)
        print(f"  ✓ Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    print("\nStep 2: Load Test Data")
    print("=" * 60)
    
    dataset_loader = LoadDataset(
        data_dir=PROJECT_ROOT / 'data',
        annotations_dir=PROJECT_ROOT / 'data' / 'annotations'
    )
    
    test_dataset = dataset_loader.load_images_from_dir('test')
    
    if not test_dataset:
        print("No test images found in data/test/")
        print("To test the model, place test images in data/test/")
        print("\nCreating a dummy test image for demonstration...")
        
        # Create dummy test image
        dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        test_dataset = [{
            'path': None,
            'filename': 'dummy_test.jpg',
            'image_id': 0
        }]
        
        # Save dummy image for visualization
        cv2.imwrite(str(PROJECT_ROOT / 'data' / 'test' / 'dummy_test.jpg'), dummy_image)
    
    print(f"  ✓ Found {len(test_dataset)} test images")
    
    print("\nStep 3: Run Inference")
    print("=" * 60)
    
    for idx, test_info in enumerate(test_dataset):
        image_path = test_info['path']
        filename = test_info['filename']
        
        print(f"\n  Processing {idx + 1}/{len(test_dataset)}: {filename}")
        
        # Load image
        if image_path is None:
            # Use dummy image from step 2
            image_path = str(PROJECT_ROOT / 'data' / 'test' / filename)
        
        try:
            image = dataset_loader.load_image(image_path)
        except Exception as e:
            print(f"    ✗ Error loading image: {e}")
            continue
        
        # Make prediction
        try:
            predictions = inference.predict(image)
            class_pred = predictions[0][0]
            confidence = np.max(class_pred)
            predicted_class = np.argmax(class_pred)
            
            print(f"    ✓ Confidence: {confidence:.4f}")
            print(f"    ✓ Predicted: {'Fire' if predicted_class == 1 else 'Background'}")
            
            # Save visualization
            output_path = OUTPUTS_DIR / f"prediction_{filename}"
            plot_results(image, predictions, output_path=output_path)
            
        except Exception as e:
            print(f"    ✗ Error during inference: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Inference Complete!")
    print("=" * 60)
    print(f"Results saved to: {OUTPUTS_DIR}")


def main():
    """Main inference pipeline"""
    
    config = FireDetectionConfig()
    
    # Run inference
    run_inference_on_test_images(config.MODEL_PATH, config)


if __name__ == "__main__":
    main()
