"""
Flask application for Mask R-CNN Fire Detection inference
Provides REST API for image segmentation
"""
import os
import numpy as np
import cv2
import json
from pathlib import Path
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
import traceback
import sys
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = Path(__file__).parent / 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'tiff', 'tif'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder
UPLOAD_FOLDER.mkdir(exist_ok=True)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)

# Model path - should be mounted or provided via environment variable
MODEL_PATH = os.environ.get('MODEL_PATH', '/models/fire_detection_model.h5')

# Global model variable
model = None


def load_model():
    """Load the pre-trained Mask R-CNN model"""
    global model
    try:
        print(f"Loading model from {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✓ Model loaded successfully")
        return True
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        traceback.print_exc()
        return False


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_path, target_size=(256, 256)):
    """
    Preprocess image for model inference
    
    Args:
        image_path: Path to image file
        target_size: Target size for resizing
        
    Returns:
        Preprocessed image array
    """
    # Read image
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError("Failed to read image")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Store original dimensions
    original_height, original_width = image.shape[:2]
    
    # Resize
    image_resized = cv2.resize(image, target_size)
    
    # Normalize
    image_normalized = image_resized.astype('float32') / 255.0
    
    return image_normalized, image_resized, (original_height, original_width)


def postprocess_predictions(predictions, confidence_threshold=0.5):
    """
    Postprocess model predictions
    
    Args:
        predictions: Model output
        confidence_threshold: Confidence threshold for detections
        
    Returns:
        Processed predictions dictionary
    """
    class_logits, bbox_output, mask_output = predictions
    
    class_pred = class_logits[0]
    bbox_pred = bbox_output[0]
    mask_pred = mask_output[0]
    
    predicted_class = int(np.argmax(class_pred))
    confidence = float(np.max(class_pred))
    
    # Prepare result
    result = {
        'predicted_class': 'Fire' if predicted_class == 1 else 'Background',
        'confidence': float(confidence),
        'is_fire': confidence > confidence_threshold and predicted_class == 1
    }
    
    # Add bounding box information
    if predicted_class == 1:  # Fire detected
        bbox_values = bbox_pred[:4]  # x1, y1, x2, y2
        result['bounding_box'] = {
            'x1': float(bbox_values[0]),
            'y1': float(bbox_values[1]),
            'x2': float(bbox_values[2]),
            'y2': float(bbox_values[3])
        }
    
    return result


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Make prediction on uploaded image
    
    Expected request:
    - POST /predict with image file as 'file' form parameter
    
    Returns:
    - JSON with prediction results
    """
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                'error': 'Model not loaded',
                'status': 'error'
            }), 503
        
        # Check if file is provided
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file part in request',
                'status': 'error'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'status': 'error'
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'error': f'File type not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}',
                'status': 'error'
            }), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = Path(app.config['UPLOAD_FOLDER']) / filename
        file.save(str(filepath))
        
        # Preprocess image
        image, image_resized, original_dims = preprocess_image(str(filepath))
        
        # Make prediction
        image_input = np.expand_dims(image, axis=0)
        predictions = model.predict(image_input)
        
        # Postprocess predictions
        result = postprocess_predictions(predictions)
        
        # Add metadata
        result['filename'] = filename
        result['original_dimensions'] = {
            'height': int(original_dims[0]),
            'width': int(original_dims[1])
        }
        result['status'] = 'success'
        
        # Clean up uploaded file
        filepath.unlink()
        
        return jsonify(result), 200
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Make predictions on multiple images
    
    Expected request:
    - POST /batch_predict with multiple image files
    
    Returns:
    - JSON with batch prediction results
    """
    try:
        if model is None:
            return jsonify({
                'error': 'Model not loaded',
                'status': 'error'
            }), 503
        
        if 'files' not in request.files:
            return jsonify({
                'error': 'No files in request',
                'status': 'error'
            }), 400
        
        files = request.files.getlist('files')
        
        if not files:
            return jsonify({
                'error': 'No files selected',
                'status': 'error'
            }), 400
        
        results = []
        
        for file in files:
            if not allowed_file(file.filename):
                results.append({
                    'filename': file.filename,
                    'error': 'File type not allowed',
                    'status': 'error'
                })
                continue
            
            try:
                # Save and process
                filename = secure_filename(file.filename)
                filepath = Path(app.config['UPLOAD_FOLDER']) / filename
                file.save(str(filepath))
                
                # Preprocess and predict
                image, _, original_dims = preprocess_image(str(filepath))
                image_input = np.expand_dims(image, axis=0)
                predictions = model.predict(image_input, verbose=0)
                
                # Postprocess
                result = postprocess_predictions(predictions)
                result['filename'] = filename
                result['status'] = 'success'
                
                results.append(result)
                
                # Clean up
                filepath.unlink()
                
            except Exception as e:
                results.append({
                    'filename': file.filename,
                    'error': str(e),
                    'status': 'error'
                })
        
        return jsonify({
            'results': results,
            'total': len(results),
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        print(f"Error during batch prediction: {e}")
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    if model is None:
        return jsonify({
            'error': 'Model not loaded',
            'status': 'error'
        }), 503
    
    return jsonify({
        'model_name': 'Mask R-CNN Fire Detection',
        'version': '1.0.0',
        'input_shape': 'Variable (images)',
        'output_shapes': {
            'classification': 'N x 2',
            'bounding_box': 'N x 8',
            'segmentation': 'N x H x W x 2'
        },
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'max_file_size': f"{MAX_CONTENT_LENGTH / (1024 * 1024):.0f}MB",
        'status': 'loaded'
    }), 200


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({
        'error': f'File too large. Maximum size: {MAX_CONTENT_LENGTH / (1024 * 1024):.0f}MB',
        'status': 'error'
    }), 413


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'status': 'error'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'error': 'Internal server error',
        'status': 'error'
    }), 500


if __name__ == '__main__':
    print("=" * 60)
    print("Mask R-CNN Fire Detection Flask API")
    print("=" * 60)
    
    # Load model
    if load_model():
        print("\nStarting Flask server...")
        print("Available endpoints:")
        print("  GET  /health - Health check")
        print("  GET  /model_info - Model information")
        print("  POST /predict - Single image prediction")
        print("  POST /batch_predict - Batch prediction")
        print("\n" + "=" * 60)
        
        # Run with uWSGI in production
        # For development:
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print("✗ Failed to load model. Please check the model path.")
        print(f"Expected path: {MODEL_PATH}")
        sys.exit(1)
