"""
VGG16 Defect Detector Wrapper
Extracted from executed_Defect-Detection-of-Aircraft.ipynb
"""

import cv2
import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.applications import VGG16
from keras.optimizers import Adam
from pathlib import Path
from typing import Dict, List


class VGG16DefectDetector:
    """
    VGG16-based defect detector for aircraft damage classification.
    Classifies damage as 'dent' or 'crack'.
    """
    
    def __init__(self, checkpoint_path: Path = None, device: str = None):
        """
        Initialize VGG16 defect detector.
        
        Args:
            checkpoint_path: Path to trained model weights (optional)
            device: 'cpu' or 'cuda' (not used for TensorFlow, kept for compatibility)
        """
        self.device = device or "cpu"
        self.img_rows, self.img_cols = 224, 224
        self.input_shape = (self.img_rows, self.img_cols, 3)
        self.class_names = ["normal", "defect"]
        
        # Build model
        self.model = self._build_model()
        
        # Load checkpoint if provided
        if checkpoint_path and Path(checkpoint_path).exists():
            try:
                self.model.load_weights(checkpoint_path)
                print(f"✅ VGG16 model loaded from {checkpoint_path}")
            except Exception as e:
                print(f"⚠️ Could not load weights: {e}")
                print("📌 Using randomly initialized model")
        else:
            print("📌 Using randomly initialized VGG16 model")
    
    def _build_model(self):
        """Build VGG16 model architecture."""
        # Load pre-trained VGG16
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=self.input_shape)
        
        # Flatten and add custom layers
        output = base_model.layers[-1].output
        output = Flatten()(output)
        base_model = Model(base_model.input, output)
        
        # Freeze base model layers
        for layer in base_model.layers:
            layer.trainable = False
        
        # Build custom model
        model = Sequential()
        model.add(base_model)
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def detect(self, frame: np.ndarray, detections: List[Dict]) -> Dict:
        """
        Run defect detection on detected objects.
        
        Args:
            frame: BGR image
            detections: List of detection dictionaries with 'bbox' key
        
        Returns:
            Dictionary with defect results per object
        """
        results = {"objects": {}}
        
        for det in detections:
            obj_id = det.get("id", 0)
            bbox = det.get("bbox", [0, 0, frame.shape[1], frame.shape[0]])
            x1, y1, x2, y2 = map(int, bbox)
            
            # Ensure valid ROI
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            roi = frame[y1:y2, x1:x2]
            
            if roi.size == 0:
                continue
            
            # Preprocess and predict
            tensor = self._preprocess(roi)
            
            with tf.no_grad():
                logits = self.model(tensor, training=False)
                confidence = float(logits[0].numpy()[0])
            
            # Determine class (binary classification)
            class_id = 1 if confidence > 0.5 else 0
            confidence = confidence if class_id == 1 else 1 - confidence
            
            results["objects"][obj_id] = {
                "predicted_class": self.class_names[class_id],
                "class_id": class_id,
                "confidence": float(confidence),
                "has_defect": class_id == 1
            }
        
        return results
    
    def _preprocess(self, roi: np.ndarray) -> tf.Tensor:
        """
        Preprocess ROI for model input.
        
        Args:
            roi: Region of interest image
        
        Returns:
            Preprocessed tensor
        """
        # Resize to model input size
        roi = cv2.resize(roi, (self.img_rows, self.img_cols))
        
        # Convert BGR to RGB
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        
        # Normalize
        roi = roi.astype(np.float32) / 255.0
        
        # Convert to tensor
        tensor = tf.convert_to_tensor(roi[np.newaxis, ...], dtype=tf.float32)
        
        return tensor
