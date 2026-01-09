"""
Image Processor Module for ZENITH INSPECT-AR
Handles image preprocessing and enhancement operations
"""

import cv2
import numpy as np
from typing import Dict, Any


class ImageProcessor:
    def __init__(self):
        """
        Initialize image processor
        """
        print("ImageProcessor initialized")
        
    def preprocess(self, image: np.ndarray, enhance_contrast: bool = True) -> np.ndarray:
        """
        Preprocess image for better detection and measurement
        """
        try:
            # Make a copy to avoid modifying the original
            processed = image.copy()
            
            # Convert to grayscale if needed
            if len(processed.shape) == 3:
                gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            else:
                gray = processed
            
            # Enhance contrast if requested
            if enhance_contrast:
                # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray = clahe.apply(gray)
                
                # Convert back to BGR if original was color
                if len(image.shape) == 3:
                    processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                else:
                    processed = gray
            else:
                processed = gray if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            return processed
            
        except Exception as e:
            print(f"Error in image preprocessing: {str(e)}")
            return image  # Return original image if preprocessing fails
    
    def enhance_quality(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image quality for better measurements
        """
        try:
            # Apply bilateral filter to reduce noise while preserving edges
            enhanced = cv2.bilateralFilter(image, 9, 75, 75)
            
            # Apply slight sharpening
            kernel = np.array([[-1,-1,-1],
                              [-1, 9,-1],
                              [-1,-1,-1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
            
            return enhanced
            
        except Exception as e:
            print(f"Error in image enhancement: {str(e)}")
            return image  # Return original image if enhancement fails