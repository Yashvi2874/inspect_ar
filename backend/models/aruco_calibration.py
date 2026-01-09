"""
Aruco Calibration Module for ZENITH INSPECT-AR
Handles ArUco marker detection and calibration for dimension measurement
"""

import cv2
import numpy as np
from typing import Dict, Tuple, List, Optional


class ArucoCalibrator:
    def __init__(self):
        """Initialize ArUco detector with common dictionary and parameters"""
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.parameters = cv2.aruco.DetectorParameters()
        
        # Set parameters for robust detection
        self.parameters.adaptiveThreshWinSizeMin = 3
        self.parameters.adaptiveThreshWinSizeMax = 23
        self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.parameters.cornerRefinementWinSize = 5
        self.parameters.minMarkerPerimeterRate = 0.03
        
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
        
    def calibrate(self, image: np.ndarray, marker_size_mm: float = 50.0) -> Dict:
        """
        Calibrate using ArUco marker to establish pixel-to-mm ratio
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Detect markers
            corners, ids, rejected_img_points = self.detector.detectMarkers(gray)
            
            if ids is None or len(ids) == 0:
                return {
                    'success': False,
                    'error': 'No ArUco markers detected',
                    'marker_count': 0,
                    'reference_size_mm': 0.0
                }
            
            # Use the first detected marker for calibration
            marker_corners = corners[0]
            
            # Calculate pixel to mm ratio
            side_lengths = []
            for i in range(4):
                p1 = marker_corners[0][i]
                p2 = marker_corners[0][(i + 1) % 4]
                side_len = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                side_lengths.append(side_len)
            
            avg_side_length = np.mean(side_lengths)
            pixel_to_mm_ratio = marker_size_mm / avg_side_length
            
            return {
                'success': True,
                'calibration': {
                    'pixel_to_mm_ratio': pixel_to_mm_ratio,
                    'marker_corners': marker_corners.tolist(),
                    'marker_ids': ids.flatten().tolist()
                },
                'marker_count': len(ids),
                'reference_size_mm': marker_size_mm
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'marker_count': 0,
                'reference_size_mm': 0.0
            }
    
    def detect_and_calibrate(self, image: np.ndarray, marker_size_mm: float = 50.0) -> Dict:
        """
        Detect markers and return calibration data
        """
        result = self.calibrate(image, marker_size_mm)
        if result['success']:
            return {
                'marker_count': result['marker_count'],
                'pixel_to_mm_ratio': result['calibration']['pixel_to_mm_ratio'],
                'reference_size_mm': result['reference_size_mm']
            }
        else:
            # Return default values when calibration fails
            return {
                'marker_count': 0,
                'pixel_to_mm_ratio': 1.0,  # Default fallback
                'reference_size_mm': marker_size_mm
            }