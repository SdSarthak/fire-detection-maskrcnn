"""
Dataset loader for Mask R-CNN Fire Detection
Handles loading images and annotations from VGG Annotator JSON files
"""
import json
import os
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
from typing import Dict, Tuple, List


class LoadDataset:
    """Load and manage dataset for fire detection"""
    
    def __init__(self, data_dir: str, annotations_dir: str):
        """
        Initialize dataset loader
        
        Args:
            data_dir: Directory containing image files
            annotations_dir: Directory containing VIA annotation JSON files
        """
        self.data_dir = Path(data_dir)
        self.annotations_dir = Path(annotations_dir)
        self.images = []
        self.annotations = {}
        self.image_info = []
        
    def load_via_annotations(self, json_file: str) -> Dict:
        """
        Load annotations from VGG Image Annotator (VIA) JSON file
        
        Args:
            json_file: Path to VIA annotation JSON file
            
        Returns:
            Dictionary with annotations
        """
        json_path = self.annotations_dir / json_file
        
        if not json_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {json_path}")
        
        with open(json_path, 'r') as f:
            annotations_data = json.load(f)
        
        return annotations_data
    
    def parse_via_annotations(self, via_data: Dict) -> Dict:
        """
        Parse VIA annotations into structured format
        
        Args:
            via_data: Raw VIA annotation data
            
        Returns:
            Parsed annotations with image names as keys
        """
        parsed_annotations = {}
        
        for image_id, image_data in via_data.items():
            filename = image_data.get('filename')
            regions = image_data.get('regions', [])
            
            masks = []
            bboxes = []
            
            for region in regions:
                region_attributes = region.get('region_attributes', {})
                shape_attributes = region.get('shape_attributes', {})
                
                # Extract polygon points
                if 'all_points_x' in shape_attributes and 'all_points_y' in shape_attributes:
                    x_points = shape_attributes['all_points_x']
                    y_points = shape_attributes['all_points_y']
                    
                    masks.append({
                        'x_points': x_points,
                        'y_points': y_points
                    })
                
                # Extract bounding box if available
                if 'x' in shape_attributes and 'y' in shape_attributes:
                    width = shape_attributes.get('width', 0)
                    height = shape_attributes.get('height', 0)
                    
                    bboxes.append({
                        'x': shape_attributes['x'],
                        'y': shape_attributes['y'],
                        'width': width,
                        'height': height
                    })
            
            parsed_annotations[filename] = {
                'masks': masks,
                'bboxes': bboxes,
                'image_id': image_id
            }
        
        return parsed_annotations
    
    def load_images_from_dir(self, directory: str, annotation_file: str = None) -> List[Dict]:
        """
        Load images from directory
        
        Args:
            directory: Directory containing images
            annotation_file: Optional annotation file name
            
        Returns:
            List of image information dictionaries
        """
        img_dir = self.data_dir / directory
        
        if not img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
        
        image_list = []
        annotations = {}
        
        # Load annotations if provided
        if annotation_file:
            via_data = self.load_via_annotations(annotation_file)
            annotations = self.parse_via_annotations(via_data)
        
        # Supported image formats
        image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
        
        for image_path in sorted(img_dir.glob('**/*')):
            if image_path.suffix.lower() in image_extensions:
                image_info = {
                    'path': str(image_path),
                    'filename': image_path.name,
                    'image_id': len(image_list),
                    'annotations': annotations.get(image_path.name, {})
                }
                image_list.append(image_info)
        
        return image_list
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load image as numpy array
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image as numpy array
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def create_mask_from_polygon(self, image_shape: Tuple[int, int, int], 
                                polygon_points: Dict) -> np.ndarray:
        """
        Create binary mask from polygon points
        
        Args:
            image_shape: Shape of image (height, width, channels)
            polygon_points: Dict with 'x_points' and 'y_points'
            
        Returns:
            Binary mask as numpy array
        """
        mask = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)
        
        x_points = polygon_points.get('x_points', [])
        y_points = polygon_points.get('y_points', [])
        
        if len(x_points) > 2 and len(y_points) > 2:
            points = np.array(list(zip(x_points, y_points)), dtype=np.int32)
            cv2.fillPoly(mask, [points], 1)
        
        return mask
    
    def get_dataset(self, subset: str = 'train', 
                   annotation_file: str = None) -> List[Dict]:
        """
        Get complete dataset
        
        Args:
            subset: 'train', 'val', or 'test'
            annotation_file: Name of annotation JSON file
            
        Returns:
            List of dataset samples with images and annotations
        """
        dataset = self.load_images_from_dir(subset, annotation_file)
        return dataset


class LoadBackbone:
    """Load pre-trained backbone weights for feature extraction"""
    
    def __init__(self, weights_path: str = None):
        """
        Initialize backbone loader
        
        Args:
            weights_path: Path to pre-trained weights
        """
        self.weights_path = weights_path
    
    def load_pretrained_weights(self, model):
        """
        Load pre-trained weights into model
        
        Args:
            model: Model to load weights into
            
        Returns:
            Model with loaded weights
        """
        if self.weights_path and os.path.exists(self.weights_path):
            try:
                model.load_weights(self.weights_path, by_name=True)
                print(f"Loaded pre-trained weights from: {self.weights_path}")
            except Exception as e:
                print(f"Could not load weights: {e}")
        
        return model
    
    def load_coco_weights(self, model):
        """
        Load COCO pre-trained weights for Mask R-CNN
        
        Args:
            model: Mask R-CNN model instance
            
        Returns:
            Model with COCO weights
        """
        print("Loading COCO pre-trained weights...")
        try:
            model.load_weights(self.weights_path, by_name=True)
        except Exception as e:
            print(f"Error loading COCO weights: {e}")
        
        return model
    
    def extract_features(self, model, image: np.ndarray) -> np.ndarray:
        """
        Extract features using backbone
        
        Args:
            model: Backbone model
            image: Input image array
            
        Returns:
            Feature maps
        """
        features = model.predict(np.expand_dims(image, axis=0))
        return features
