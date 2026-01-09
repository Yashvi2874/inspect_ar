"""
YOLO Detector Module for ZENITH INSPECT-AR
Handles object detection using YOLOv8 with contour-based fallback
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any


class YOLODetector:
    def __init__(self, models_folder: Path = None, use_yolo=True):
        """
        Initialize YOLO detector
        Args:
            models_folder: Path to models directory
            use_yolo: If True, attempt to load YOLOv8 model, otherwise use contour detection
        """
        self.models_folder = models_folder
        self.model = None
        self.use_yolo = use_yolo
        
        if use_yolo:
            try:
                from ultralytics import YOLO
                # Try to load YOLOv8 model (will download if not present)
                self.model = YOLO('yolov8n.pt')  # Nano model (fastest)
                print("✅ YOLOv8 model loaded successfully")
            except Exception as e:
                print(f"⚠️ Could not load YOLOv8: {e}")
                print("📌 Falling back to enhanced contour detection")
                self.model = None
        else:
            print("📌 Using enhanced contour detection")
        
    def detect(self, image: np.ndarray, conf_threshold=0.25) -> Dict:
        """
        Detect objects in the image using YOLOv8 or enhanced contour detection
        Args:
            image: Input image (BGR format)
            conf_threshold: Confidence threshold for detections
        Returns:
            Dictionary with 'detections' and 'count'
        """
        if self.model is not None:
            return self._detect_with_yolo(image, conf_threshold)
        else:
            return self._detect_with_contours(image)
    
    def _detect_with_yolo(self, image: np.ndarray, conf_threshold: float) -> Dict:
        """
        Detect objects using YOLOv8
        """
        try:
            # Run inference
            results = self.model(image, conf=conf_threshold, verbose=False)
            
            detections = []
            
            for i, result in enumerate(results):
                boxes = result.boxes
                
                for j, box in enumerate(boxes):
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Ignore humans
                    if self.model.names[class_id] == "person":
                        continue
                    
                    # Calculate area
                    area = (x2 - x1) * (y2 - y1)
                    
                    detection = {
                        'id': j,
                        'class': self.model.names[class_id],
                        'class_id': class_id,
                        'confidence': confidence,
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'area': float(area)
                    }
                    
                    detections.append(detection)
            
            return {
                'detections': detections,
                'count': len(detections)
            }
            
        except Exception as e:
            print(f"Error in YOLO detection: {str(e)}")
            return {
                'detections': [],
                'count': 0
            }
    
    def _detect_with_contours(self, image: np.ndarray) -> Dict:
        """
        Enhanced contour-based detection with adaptive thresholding
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply bilateral filter to preserve edges while reducing noise
            blurred = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
            
            # Use adaptive thresholding for better detection in varying lighting
            thresh = cv2.adaptiveThreshold(
                blurred,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                blockSize=21,
                C=5
            )
            
            # Also try Otsu's thresholding
            _, thresh_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Combine both thresholds
            thresh_combined = cv2.bitwise_or(thresh, thresh_otsu)
            
            # Morphological operations to clean up
            kernel = np.ones((5, 5), np.uint8)
            thresh_combined = cv2.morphologyEx(thresh_combined, cv2.MORPH_CLOSE, kernel, iterations=2)
            thresh_combined = cv2.morphologyEx(thresh_combined, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Find contours
            contours, _ = cv2.findContours(thresh_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detections = []
            
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                
                # Filter small contours (noise) - lower threshold for better detection
                if area > 300:  # Reduced from 500
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Calculate confidence based on contour properties
                    aspect_ratio = float(w) / h if h > 0 else 0
                    extent = area / (w * h) if (w * h) > 0 else 0
                    
                    # Get hull and calculate solidity
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = area / hull_area if hull_area > 0 else 0
                    
                    # Estimate confidence based on geometric properties
                    # Higher confidence for solid, well-formed objects
                    confidence = min(0.95, 
                                   0.4 +  # Base confidence
                                   (extent * 0.2) +  # Fill ratio
                                   (solidity * 0.2) +  # Solidity
                                   (min(area / 1000, 1.0) * 0.15))  # Size factor
                    
                    detection = {
                        'id': i,
                        'class': 'object',  # Generic class for contour detection
                        'confidence': confidence,
                        'bbox': [float(x), float(y), float(x + w), float(y + h)],
                        'area': float(area),
                        'solidity': float(solidity),
                        'extent': float(extent)
                    }
                    
                    detections.append(detection)
            
            return {
                'detections': detections,
                'count': len(detections)
            }
            
        except Exception as e:
            print(f"Error in contour detection: {str(e)}")
            return {
                'detections': [],
                'count': 0
            }