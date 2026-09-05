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
        # The training notebook used flow_from_directory over crack/ and dent/,
        # which maps crack=0 and dent=1. BOTH are damage; the dataset has no
        # healthy class. The old ["normal", "defect"] labelling meant a crack
        # was reported as "normal", i.e. as not a defect.
        self.class_names = ["crack", "dent"]

        # Build model. VGG16(weights='imagenet') downloads ~58 MB on first use,
        # so this can fail on an offline machine; let that surface here rather
        # than halfway through a frame.
        self.model = self._build_model()

        self.weights_loaded = False

        # Load checkpoint if provided
        if checkpoint_path and Path(checkpoint_path).exists():
            try:
                self.model.load_weights(checkpoint_path)
                self.weights_loaded = True
                print(f"[ok] VGG16 model loaded from {checkpoint_path}")
            except Exception as e:
                print(f"[warn] Could not load weights: {e}")
                print("[warn] Predictions from this model are MEANINGLESS "
                      "(randomly initialized head).")
        else:
            print("[warn] No VGG16 checkpoint found. Predictions from this "
                  "model are MEANINGLESS (randomly initialized head).")

        if not self.weights_loaded:
            # best_model.pth is a PyTorch Faster R-CNN; Keras cannot read it and
            # no .keras/.weights.h5 exists anywhere in the project, so this path
            # is the norm rather than the exception. Use DefectDetector instead.
            print("[warn] Prefer models.defect_detector.DefectDetector, which "
                  "has real trained weights.")
    
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
            
            # `tf.no_grad()` does not exist — that is the PyTorch API
            # (torch.no_grad) and it raised AttributeError on every call,
            # taking the whole pipeline down. Keras builds no gradient tape
            # under training=False, so no context manager is needed here.
            logits = self.model(tensor, training=False)
            confidence = float(logits[0].numpy()[0])

            # Sigmoid output: p is P(class 1) = P(dent), so 1-p is P(crack).
            class_id = 1 if confidence > 0.5 else 0
            confidence = confidence if class_id == 1 else 1 - confidence

            results["objects"][obj_id] = {
                "predicted_class": self.class_names[class_id],
                "class_id": class_id,
                "confidence": float(confidence),
                # Both trained classes are damage: crack and dent. There is no
                # healthy class, so a prediction either way means damage.
                "has_defect": True,
                "weights_loaded": self.weights_loaded,
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
