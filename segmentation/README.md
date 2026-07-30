# Segmentation Module Documentation

This module handles model development, training, and inference for the fire detection system using Mask R-CNN.

## Module Overview

The segmentation module is responsible for:
1. Loading and preprocessing training data
2. Building the Mask R-CNN model architecture
3. Training the model on annotated fire images
4. Evaluating model performance
5. Running inference on new images
6. Visualizing predictions

## Directory Structure

```
segmentation/
├── src/
│   ├── config.py       # Configuration management
│   ├── dataset.py      # Data loading and preprocessing
│   ├── model.py        # Model architecture and training
│   └── __init__.py
├── data/
│   ├── train/          # 20 training images
│   ├── val/            # 10 validation images
│   ├── test/           # Test images
│   └── annotations/    # JSON annotation files
├── weights/            # Pre-trained and trained model weights
├── outputs/            # Prediction results and visualizations
├── train.py            # Training script
├── infer.py            # Inference script
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager
- Virtual environment (recommended)

### Setup

```bash
# Navigate to segmentation directory
cd segmentation

# Create virtual environment (optional but recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

All model parameters are defined in `src/config.py`:

```python
class FireDetectionConfig:
    # Image processing
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 512
    
    # Model
    NUM_CLASSES = 2  # Background + Fire
    BACKBONE = "resnet50"
    
    # Training
    TRAIN_EPOCHS = 50
    STEPS_PER_EPOCH = 100
    LEARNING_RATE = 0.001
    
    # Detection
    DETECTION_MIN_CONFIDENCE = 0.7
    DETECTION_NMS_THRESHOLD = 0.3
```

### Key Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| IMAGE_MIN_DIM | 256 | Minimum image dimension |
| IMAGE_MAX_DIM | 512 | Maximum image dimension |
| NUM_CLASSES | 2 | Number of classes (background + fire) |
| LEARNING_RATE | 0.001 | Optimizer learning rate |
| TRAIN_EPOCHS | 50 | Number of training epochs |
| DETECTION_MIN_CONFIDENCE | 0.7 | Confidence threshold for detections |

## Data Preparation

### Dataset Requirements

The project expects:
- **Training Set**: 20 labeled fire images
- **Validation Set**: 10 labeled fire images
- **Test Set**: Arbitrary number of test images
- **Annotations**: JSON files with region masks (from VGG Annotator)

### Creating Annotations with VGG Annotator

1. Go to [VGG Annotator](https://www.robots.ox.ac.uk/~vgg/software/via/via_demo.html)
2. Load your images
3. Draw polygons around fire regions
4. Mark regions with attribute `object: fire`
5. Export as JSON: Project → Export Annotations → as JSON
6. Save JSON files to `data/annotations/`

### Annotation File Format

```json
{
  "image_id_1": {
    "filename": "fire_1.jpg",
    "regions": [
      {
        "shape_attributes": {
          "name": "polygon",
          "all_points_x": [100, 150, 200, 100],
          "all_points_y": [100, 80, 120, 150]
        },
        "region_attributes": {
          "object": "fire"
        }
      }
    ]
  }
}
```

## Model Components

### Dataset Module (`dataset.py`)

Handles loading and preprocessing data:

```python
from src.dataset import LoadDataset, LoadBackbone

# Initialize dataset loader
loader = LoadDataset(data_dir='data', annotations_dir='data/annotations')

# Load training dataset
train_data = loader.load_images_from_dir('train', 'train_annotations.json')

# Load validation dataset
val_data = loader.load_images_from_dir('val', 'val_annotations.json')
```

### Key Classes

#### LoadDataset
- Loads images from directories
- Parses VGG Annotator JSON annotations
- Creates binary masks from polygon annotations
- Returns structured dataset

```python
class LoadDataset:
    def __init__(self, data_dir: str, annotations_dir: str)
    def load_via_annotations(self, json_file: str) -> Dict
    def load_images_from_dir(self, directory: str, annotation_file: str = None) -> List[Dict]
    def load_image(self, image_path: str) -> np.ndarray
    def create_mask_from_polygon(self, image_shape, polygon_points) -> np.ndarray
    def get_dataset(self, subset: str = 'train', annotation_file: str = None) -> List[Dict]
```

#### LoadBackbone
- Manages pre-trained weights
- Loads COCO pre-trained weights
- Extracts features using backbone

```python
class LoadBackbone:
    def __init__(self, weights_path: str = None)
    def load_pretrained_weights(self, model)
    def load_coco_weights(self, model)
    def extract_features(self, model, image: np.ndarray) -> np.ndarray
```

### Model Module (`model.py`)

Contains model architecture and training:

#### MaskRCNNBuilder
Builds Mask R-CNN architecture:

```python
from src.model import MaskRCNNBuilder

builder = MaskRCNNBuilder(config)

# Build complete model
model = builder.build_complete_model()

# Compile model
model = builder.compile_model(model)

# Get model summary
model.summary()
```

**Model Components:**
- ResNet50 backbone
- Region Proposal Network (RPN)
- ROI Align layer
- Detection head (classification + bounding box)
- Mask head (segmentation)

#### ModelTrainer
Handles training process:

```python
from src.model import ModelTrainer

trainer = ModelTrainer(model, config)

# Train model
history = trainer.train(
    train_generator=train_data,
    val_generator=val_data,
    epochs=50
)

# Save trained model
trainer.save_model('weights/fire_detection_model.h5')
```

**Features:**
- Early stopping
- Learning rate reduction
- Model checkpointing
- Training history tracking

#### ModelInference
Performs inference:

```python
from src.model import ModelInference

inference = ModelInference('weights/fire_detection_model.h5', config)

# Make prediction
predictions = inference.predict(image)
class_logits, bbox_output, mask_output = predictions
```

## Training

### Starting Training

```bash
python train.py
```

This script:
1. Loads configuration
2. Downloads pre-trained weights
3. Loads dataset
4. Builds model
5. Compiles model
6. Trains for specified epochs
7. Saves trained model

### Training Process

```
Step 1: Download Pre-trained Weights
  ✓ Weights configuration initialized

Step 2: Load Dataset
  ✓ Training samples: 20
  ✓ Validation samples: 10

Step 3: Load Backbone Weights
  ✓ Backbone loader initialized

Step 4: Build Model
  ✓ Model architecture created
  ✓ Model compiled

Step 5: Create Data Generators
  ✓ Data generators created

Step 6: Train Model
  Starting training...
  Epochs: 50
  Steps per epoch: 100

Step 7: Save Model
  ✓ Model saved to weights/fire_detection_model.h5
```

### Monitoring Training

During training, the script will output:
- Epoch progress
- Loss values
- Accuracy metrics
- Validation results

```
Epoch 1/50
100/100 [==============================] - 120s 1s/step
loss: 0.5432 - accuracy: 0.8234
val_loss: 0.4321 - val_accuracy: 0.8567
```

### Training Parameters

Adjust in `src/config.py`:

```python
TRAIN_EPOCHS = 50              # Increase for better accuracy
LEARNING_RATE = 0.001         # Decrease for stability, increase for speed
STEPS_PER_EPOCH = 100         # Based on dataset size
VALIDATION_STEPS = 10         # Validation frequency
DETECTION_MIN_CONFIDENCE = 0.7 # Confidence threshold
```

## Inference

### Running Inference

```bash
python infer.py
```

This script:
1. Loads trained model
2. Loads test images
3. Makes predictions
4. Visualizes results
5. Saves output images

### Inference Output

For each test image, the script produces:
- Predicted class (Fire/Background)
- Confidence score
- Bounding box coordinates (if fire detected)
- Segmentation mask visualization

Example output:
```json
{
  "predicted_class": "Fire",
  "confidence": 0.95,
  "is_fire": true,
  "bounding_box": {
    "x1": 100,
    "y1": 150,
    "x2": 300,
    "y2": 400
  },
  "filename": "fire_image.jpg"
}
```

### Custom Inference

```python
from src.model import ModelInference
from src.dataset import LoadDataset
import cv2

# Initialize
config = FireDetectionConfig()
inference = ModelInference('weights/fire_detection_model.h5', config)
loader = LoadDataset('data', 'data/annotations')

# Load image
image = loader.load_image('path/to/image.jpg')

# Make prediction
predictions = inference.predict(image)
class_logits, bbox, mask = predictions

# Extract results
confidence = class_logits[0]
print(f"Confidence: {confidence}")
print(f"Fire detected: {confidence[1] > 0.5}")
```

## Model Evaluation

### Evaluation Metrics

The model is evaluated on:

1. **Classification Metrics**
   - Accuracy: Overall correctness
   - Precision: True positive rate
   - Recall: Detection rate
   - F1-Score: Harmonic mean

2. **Detection Metrics**
   - mAP (mean Average Precision): Detection quality at various IoU thresholds
   - Precision-Recall curve

3. **Segmentation Metrics**
   - IoU (Intersection over Union): Mask overlap quality
   - Dice Coefficient: Segmentation similarity

### Evaluating on Test Set

```python
from src.model import ModelInference
from src.dataset import LoadDataset
import numpy as np

config = FireDetectionConfig()
inference = ModelInference('weights/fire_detection_model.h5', config)
loader = LoadDataset('data', 'data/annotations')

# Load test dataset
test_data = loader.get_dataset('test', 'test_annotations.json')

predictions = []
ground_truth = []

for sample in test_data:
    image = loader.load_image(sample['path'])
    pred = inference.predict(image)
    predictions.append(pred)
    ground_truth.append(sample['annotations'])

# Calculate metrics
# (implementation depends on evaluation framework)
```

## Output Visualization

The inference script generates visualizations showing:
1. Original image
2. Predicted fire class
3. Predicted segmentation mask
4. Bounding box overlay

Visualizations are saved to `outputs/` directory.

## Performance Optimization

### Speed Optimization

1. **Reduce image size**: Decrease IMAGE_MAX_DIM
2. **Use batch processing**: Process multiple images simultaneously
3. **Enable mixed precision**: Use float16 for faster computation
4. **Use TensorRT**: Optimize model for inference

### Accuracy Optimization

1. **Increase training epochs**: Train longer for better fit
2. **Use data augmentation**: Reduce overfitting
3. **Fine-tune backbone**: Train more layers
4. **Collect more data**: Use larger training set
5. **Adjust learning rate**: Use learning rate schedules

### Memory Optimization

1. **Reduce batch size**: Lower memory usage
2. **Use gradient accumulation**: Simulate larger batches
3. **Enable gradient checkpointing**: Trade compute for memory
4. **Use mixed precision training**: Reduce memory footprint

## Common Issues and Solutions

### Issue: Out of Memory (OOM)

```bash
# Reduce batch size in config
BATCH_SIZE = 1

# Reduce image size
IMAGE_MAX_DIM = 256

# Use gradient checkpointing
# (Add to model training)
```

### Issue: Model Not Converging

```bash
# Check learning rate
LEARNING_RATE = 0.0001  # Decrease if loss oscillates

# Add more training data
# Check data preprocessing

# Verify annotations are correct
```

### Issue: Low Accuracy

```bash
# Increase training epochs
TRAIN_EPOCHS = 100

# Collect more diverse data
# Verify annotation quality

# Adjust confidence threshold
DETECTION_MIN_CONFIDENCE = 0.5
```

## Advanced Topics

### Custom Loss Functions

```python
def custom_loss(y_true, y_pred):
    # Weighted loss for class imbalance
    class_weights = [1.0, 2.0]  # Higher weight for fire class
    return weighted_loss(y_true, y_pred, class_weights)
```

### Data Augmentation

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)
```

### Transfer Learning

```python
# Load pre-trained backbone
backbone = keras.applications.ResNet50(weights='imagenet')

# Freeze early layers
for layer in backbone.layers[:-10]:
    layer.trainable = False

# Fine-tune last layers during training
```

## Testing

### Unit Tests

```bash
# Create test file: test_model.py
python -m pytest test_model.py -v
```

### Integration Tests

```bash
# Test complete pipeline
python -m pytest test_integration.py -v
```

## Next Steps

1. **Prepare Data**: Gather and annotate fire images using VGG Annotator
2. **Place Data**: Organize images and annotations in `data/` directory
3. **Train Model**: Run `python train.py`
4. **Evaluate**: Check accuracy and adjust hyperparameters
5. **Deploy**: Move trained model to `mlops/` for deployment
6. **Monitor**: Track model performance in production

## References

- [Mask R-CNN Paper](https://arxiv.org/abs/1703.06870)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Keras Documentation](https://keras.io/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [VGG Annotator](https://www.robots.ox.ac.uk/~vgg/software/via/)

## Support

For issues specific to the segmentation module:
- Check the troubleshooting section
- Review configuration parameters
- Verify data format
- Check Python version compatibility

---

**Module Version**: 1.0.0
**Last Updated**: January 29, 2024
