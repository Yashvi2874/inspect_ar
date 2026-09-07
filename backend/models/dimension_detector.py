"""
Dimension Detector Module for ZENITH INSPECT-AR
Handles dimension measurement of detected objects using calibrated ArUco markers
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any


class DimensionDetector:
    def __init__(self, models_folder: Path = None):
        """
        Initialize dimension detector
        """
        self.models_folder = models_folder
        print("DimensionDetector initialized")

    @staticmethod
    def _binarise(gray: np.ndarray) -> np.ndarray:
        """
        Threshold a ROI so the OBJECT becomes white, whichever way round it is.

        Otsu splits the histogram into two groups but says nothing about which
        group is the object. Plain THRESH_BINARY always keeps the brighter
        group, so a dark part on a light panel — the common case on aircraft
        skin — makes the BACKGROUND the largest contour, and the measurement
        describes the panel instead of the part.

        Polarity is decided by comparing the ROI's border with its centre. The
        detection box is drawn around the object, so the centre is the object
        and the border ring is mostly background. If the centre is darker than
        the border, the object is dark and the threshold is inverted.
        """
        h, w = gray.shape[:2]
        if h < 8 or w < 8:
            # Too small to sample a border ring meaningfully; keep it simple.
            _, binary = cv2.threshold(gray, 0, 255,
                                      cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            return binary

        # Border ring: the outer eighth on each side.
        by, bx = max(1, h // 8), max(1, w // 8)
        border = np.concatenate([
            gray[:by, :].ravel(), gray[-by:, :].ravel(),
            gray[:, :bx].ravel(), gray[:, -bx:].ravel(),
        ])
        centre = gray[h // 4:3 * h // 4, w // 4:3 * w // 4]

        object_is_dark = float(centre.mean()) < float(border.mean())
        mode = cv2.THRESH_BINARY_INV if object_is_dark else cv2.THRESH_BINARY

        _, binary = cv2.threshold(gray, 0, 255, mode | cv2.THRESH_OTSU)

        # Close pinholes so one object yields one contour rather than several.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    def measure(self, image: np.ndarray, detections: List[Dict], pixel_to_mm_ratio: float) -> Dict:
        """
        Measure dimensions of detected objects
        """
        try:
            measurements = {'objects': {}, 'overall_confidence': 95.0}
            
            for detection in detections:
                obj_id = detection['id']
                
                # Extract bounding box coordinates
                x1, y1, x2, y2 = detection['bbox']
                
                # Get the region of interest
                roi = image[int(y1):int(y2), int(x1):int(x2)]
                
                # Convert to grayscale for contour detection. The ROI is already
                # 2-D when ImageProcessor ran with enhance_contrast=False, and
                # cvtColor raises on that rather than passing it through.
                gray = roi if roi.ndim == 2 else cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                
                # Threshold so the object is white regardless of polarity.
                binary = self._binarise(gray)
                
                # Find contours
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Get the largest contour (assuming it's the main object)
                    largest_contour = max(contours, key=cv2.contourArea)
                    
                    # Get rotated rectangle for accurate dimensions regardless of orientation
                    rect = cv2.minAreaRect(largest_contour)
                    width_px, height_px = rect[1]
                    
                    # Ensure width is the longer dimension
                    if width_px < height_px:
                        width_px, height_px = height_px, width_px
                    
                    # Convert to mm using the calibration ratio
                    width_mm = width_px * pixel_to_mm_ratio
                    height_mm = height_px * pixel_to_mm_ratio
                    
                    # Calculate area and perimeter
                    area_px = cv2.contourArea(largest_contour)
                    area_mm2 = area_px * (pixel_to_mm_ratio ** 2)
                    perimeter_px = cv2.arcLength(largest_contour, True)
                    perimeter_mm = perimeter_px * pixel_to_mm_ratio
                    
                    # Calculate bounding box dimensions in mm
                    bbox_width_mm = (x2 - x1) * pixel_to_mm_ratio
                    bbox_height_mm = (y2 - y1) * pixel_to_mm_ratio
                    
                    # Calculate aspect ratio
                    aspect_ratio = width_mm / height_mm if height_mm > 0 else 0
                    
                    # Store measurements
                    measurements['objects'][obj_id] = {
                        'width_mm': round(float(width_mm), 2),
                        'height_mm': round(float(height_mm), 2),
                        'area_mm2': round(float(area_mm2), 2),
                        'perimeter_mm': round(float(perimeter_mm), 2),
                        'bbox_width_mm': round(float(bbox_width_mm), 2),
                        'bbox_height_mm': round(float(bbox_height_mm), 2),
                        'aspect_ratio': round(float(aspect_ratio), 2),
                        'confidence': detection.get('confidence', 0.0)
                    }
                else:
                    # Default values if no contours found
                    measurements['objects'][obj_id] = {
                        'width_mm': 0.0,
                        'height_mm': 0.0,
                        'area_mm2': 0.0,
                        'perimeter_mm': 0.0,
                        'bbox_width_mm': 0.0,
                        'bbox_height_mm': 0.0,
                        'aspect_ratio': 0.0,
                        'confidence': detection.get('confidence', 0.0)
                    }
            
            return measurements
            
        except Exception as e:
            print(f"Error in dimension measurement: {str(e)}")
            # Return default measurements in case of error
            measurements = {'objects': {}, 'overall_confidence': 50.0}
            for detection in detections:
                obj_id = detection['id']
                measurements['objects'][obj_id] = {
                    'width_mm': 0.0,
                    'height_mm': 0.0,
                    'area_mm2': 0.0,
                    'perimeter_mm': 0.0,
                    'bbox_width_mm': 0.0,
                    'bbox_height_mm': 0.0,
                    'aspect_ratio': 0.0,
                    'confidence': detection.get('confidence', 0.0)
                }
            return measurements