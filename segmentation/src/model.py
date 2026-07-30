"""
Mask R-CNN Model for Fire Detection
Builds the complete Mask R-CNN architecture for image segmentation
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
from pathlib import Path


class MaskRCNNBuilder:
    """Build Mask R-CNN model for fire detection"""
    
    def __init__(self, config):
        """
        Initialize model builder
        
        Args:
            config: Configuration object with model parameters
        """
        self.config = config
    
    def build_backbone(self, input_shape=(None, None, 3)):
        """
        Build ResNet50 backbone
        
        Args:
            input_shape: Input image shape
            
        Returns:
            ResNet50 backbone model
        """
        # Load pre-trained ResNet50
        backbone = keras.applications.ResNet50(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze early layers
        for layer in backbone.layers[:-10]:
            layer.trainable = False
        
        return backbone
    
    def build_rpn(self, features):
        """
        Build Region Proposal Network (RPN)
        
        Args:
            features: Feature maps from backbone
            
        Returns:
            RPN outputs (proposals, scores)
        """
        # Simplified RPN implementation
        rpn = keras.Sequential([
            layers.Conv2D(512, 3, padding='same', activation='relu'),
            layers.Conv2D(3 * len(self.config.RPN_ANCHOR_SCALES), 1, padding='same')
        ])(features)
        
        return rpn
    
    def build_roi_pool(self, features, proposals, pool_size=7):
        """
        Build ROI Align layer
        
        Args:
            features: Feature maps
            proposals: Region proposals from RPN
            pool_size: Size of pooled features
            
        Returns:
            Pooled ROI features
        """
        # Simplified ROI pooling
        pooled = layers.GlobalAveragePooling2D()(features)
        return pooled
    
    def build_detection_head(self, roi_features):
        """
        Build detection classification and bounding box regression head
        
        Args:
            roi_features: Pooled ROI features
            
        Returns:
            Classification and regression outputs
        """
        x = layers.Dense(1024, activation='relu')(roi_features)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(1024, activation='relu')(x)
        
        # Classification branch
        class_logits = layers.Dense(
            self.config.NUM_CLASSES,
            activation='softmax',
            name='class_output'
        )(x)
        
        # Bounding box regression branch
        bbox_output = layers.Dense(
            self.config.NUM_CLASSES * 4,
            name='bbox_output'
        )(x)
        
        return class_logits, bbox_output
    
    def build_mask_head(self, roi_features, num_classes):
        """
        Build mask prediction head
        
        Args:
            roi_features: ROI features
            num_classes: Number of classes
            
        Returns:
            Mask predictions
        """
        # Deconvolution layers for mask generation
        x = layers.Dense(256, activation='relu')(roi_features)
        x = layers.Reshape((16, 16, 1))(x)
        
        x = layers.Conv2DTranspose(256, 2, strides=2, activation='relu')(x)
        x = layers.Conv2DTranspose(256, 2, strides=2, activation='relu')(x)
        x = layers.Conv2DTranspose(256, 2, strides=2, activation='relu')(x)
        
        # Output mask
        mask_output = layers.Conv2D(
            num_classes,
            1,
            activation='sigmoid',
            name='mask_output'
        )(x)
        
        return mask_output
    
    def build_complete_model(self):
        """
        Build complete Mask R-CNN model
        
        Returns:
            Compiled Mask R-CNN model
        """
        # Input
        image_input = keras.Input(shape=(None, None, 3), name='image_input')
        
        # Backbone
        backbone = self.build_backbone()
        features = backbone(image_input)
        
        # Simplified model for demonstration
        # In production, this would include full RPN, ROI Align, etc.
        
        # Flatten features
        flat_features = layers.GlobalAveragePooling2D()(features)
        
        # Detection head
        class_logits = layers.Dense(
            self.config.NUM_CLASSES,
            activation='softmax',
            name='class_output'
        )(flat_features)
        
        bbox_output = layers.Dense(
            self.config.NUM_CLASSES * 4,
            name='bbox_output'
        )(flat_features)
        
        # Mask head
        mask_features = layers.Dense(256, activation='relu')(flat_features)
        mask_features = layers.Reshape((16, 16, 1))(mask_features)
        
        mask_features = layers.Conv2DTranspose(256, 2, strides=2, activation='relu')(mask_features)
        mask_features = layers.Conv2DTranspose(256, 2, strides=2, activation='relu')(mask_features)
        mask_features = layers.Conv2DTranspose(256, 2, strides=2, activation='relu')(mask_features)
        
        mask_output = layers.Conv2D(
            self.config.NUM_CLASSES,
            1,
            activation='sigmoid',
            name='mask_output'
        )(mask_features)
        
        # Model
        model = keras.Model(
            inputs=image_input,
            outputs=[class_logits, bbox_output, mask_output],
            name='MaskRCNN_FireDetection'
        )
        
        return model
    
    def compile_model(self, model):
        """
        Compile model
        
        Args:
            model: Model to compile
            
        Returns:
            Compiled model
        """
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.LEARNING_RATE),
            loss={
                'class_output': 'categorical_crossentropy',
                'bbox_output': 'mse',
                'mask_output': 'binary_crossentropy'
            },
            loss_weights={
                'class_output': 1.0,
                'bbox_output': 1.0,
                'mask_output': 1.0
            },
            metrics=['accuracy']
        )
        
        return model


class ModelTrainer:
    """Train Mask R-CNN model"""
    
    def __init__(self, model, config):
        """
        Initialize trainer
        
        Args:
            model: Compiled model
            config: Configuration object
        """
        self.model = model
        self.config = config
        self.history = None
    
    def train(self, train_generator, val_generator, epochs=None):
        """
        Train model
        
        Args:
            train_generator: Training data generator
            val_generator: Validation data generator
            epochs: Number of epochs (uses config if not provided)
            
        Returns:
            Training history
        """
        if epochs is None:
            epochs = self.config.TRAIN_EPOCHS
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                str(self.config.MODEL_PATH),
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        self.history = self.model.fit(
            train_generator,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=self.config.VALIDATION_STEPS,
            callbacks=callbacks
        )
        
        return self.history
    
    def save_model(self, path):
        """
        Save trained model
        
        Args:
            path: Path to save model
        """
        self.model.save(path)
        print(f"Model saved to {path}")


class ModelInference:
    """Perform inference with trained Mask R-CNN model"""
    
    def __init__(self, model_path, config):
        """
        Initialize inference
        
        Args:
            model_path: Path to trained model
            config: Configuration object
        """
        self.model = keras.models.load_model(model_path)
        self.config = config
    
    def predict(self, image):
        """
        Make predictions on image
        
        Args:
            image: Input image array
            
        Returns:
            Predictions (class, bbox, mask)
        """
        # Preprocess image
        processed_image = self._preprocess(image)
        
        # Make prediction
        predictions = self.model.predict(np.expand_dims(processed_image, axis=0))
        
        return predictions
    
    def _preprocess(self, image):
        """
        Preprocess image for inference
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        # Normalize
        image = image.astype('float32') / 255.0
        
        # Resize if needed
        if image.shape[0] > self.config.IMAGE_MAX_DIM or image.shape[1] > self.config.IMAGE_MAX_DIM:
            scale = self.config.IMAGE_MAX_DIM / max(image.shape[0], image.shape[1])
            new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
            image = tf.image.resize(image, new_size)
        
        return image
