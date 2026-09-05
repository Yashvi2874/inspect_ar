"""
Integrated Inspection Pipeline
Combines VGG16 defect detection, BLIP captioning, and dimension measurement
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import time

from models.yolo_detector import YOLODetector
from models.vgg16_wrapper import VGG16DefectDetector
from models.blip_wrapper import BLIPCaptioner
from models.dimension_detector import DimensionDetector
from models.aruco_calibration import ArucoCalibrator
from utils.image_processor import ImageProcessor
import industrial_measurement as im


class IntegratedInspectionPipeline:
    """
    Complete inspection pipeline combining:
    - Object detection (YOLO)
    - Defect classification (VGG16)
    - Image captioning (BLIP)
    - Dimension measurement (ArUco + contours)
    """
    
    def __init__(
        self,
        models_folder: Path = None,
        use_vgg16: bool = True,
        use_blip: bool = True,
        use_dimensions: bool = True
    ):
        """
        Initialize the integrated pipeline.
        
        Args:
            models_folder: Path to models directory
            use_vgg16: Enable VGG16 defect detection
            use_blip: Enable BLIP captioning
            use_dimensions: Enable dimension measurement
        """
        self.models_folder = models_folder or Path("backend/models")
        
        print("🚀 Initializing Integrated Inspection Pipeline...")
        
        # Initialize object detector
        print("📌 Loading YOLO detector...")
        self.yolo_detector = YOLODetector(models_folder=self.models_folder)
        
        # Initialize defect detector
        if use_vgg16:
            print("📌 Loading VGG16 defect detector...")
            checkpoint = self.models_folder / "best_model.pth"
            self.vgg16_detector = VGG16DefectDetector(checkpoint_path=checkpoint)
        else:
            self.vgg16_detector = None
        
        # Initialize captioner
        if use_blip:
            print("📌 Loading BLIP captioner...")
            self.blip_captioner = BLIPCaptioner()
        else:
            self.blip_captioner = None
        
        # Initialize dimension detector
        if use_dimensions:
            print("📌 Loading dimension detector...")
            self.dimension_detector = DimensionDetector(models_folder=self.models_folder)
            self.aruco_calibrator = ArucoCalibrator()
        else:
            self.dimension_detector = None
            self.aruco_calibrator = None
        
        # Image processor
        self.image_processor = ImageProcessor()
        
        # Calibration data
        self.calibration_data = None
        self.pixel_to_mm_ratio = None
        
        print("✅ Pipeline initialized successfully")
    
    def calibrate(self, frame: np.ndarray, marker_size_mm: float = 50.0) -> bool:
        """
        Calibrate camera using ArUco marker.
        
        Args:
            frame: Input frame with ArUco marker
            marker_size_mm: Size of marker in mm
        
        Returns:
            True if calibration successful
        """
        if self.aruco_calibrator is None:
            print("⚠️ Dimension detector not enabled")
            return False
        
        try:
            print("📌 Calibrating camera...")
            self.calibration_data = self.aruco_calibrator.detect_and_calibrate(
                frame,
                marker_size_mm=marker_size_mm
            )
            
            if self.calibration_data and "pixel_to_mm_ratio" in self.calibration_data:
                self.pixel_to_mm_ratio = self.calibration_data["pixel_to_mm_ratio"]
                print(f"✅ Calibration successful. Ratio: {self.pixel_to_mm_ratio:.4f} mm/px")
                return True
            else:
                print("⚠️ Calibration failed - marker not detected")
                return False
        except Exception as e:
            print(f"⚠️ Calibration error: {e}")
            return False
    
    def process_frame(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.25,
        enhance_contrast: bool = True
    ) -> Dict:
        """
        Process a single frame through the complete pipeline.
        
        Args:
            frame: Input BGR image
            conf_threshold: Confidence threshold for YOLO
            enhance_contrast: Apply contrast enhancement
        
        Returns:
            Dictionary with all detection results
        """
        start_time = time.time()
        
        # Preprocess image
        processed_frame = self.image_processor.preprocess(
            frame,
            enhance_contrast=enhance_contrast
        )
        
        results = {
            "timestamp": time.time(),
            "frame_shape": frame.shape,
            "detections": [],
            "defects": {},
            "captions": {},
            "dimensions": {},
            "processing_time": 0
        }
        
        # Step 1: Object Detection (YOLO)
        print("🔍 Step 1: Object Detection...")
        yolo_results = self.yolo_detector.detect(processed_frame, conf_threshold=conf_threshold)
        detections = yolo_results.get("detections", [])
        results["detections"] = detections
        print(f"   Found {len(detections)} objects")
        
        if len(detections) == 0:
            results["processing_time"] = time.time() - start_time
            return results
        
        # Step 2: Defect Classification (VGG16)
        if self.vgg16_detector is not None:
            print("🔍 Step 2: Defect Classification (VGG16)...")
            defect_results = self.vgg16_detector.detect(processed_frame, detections)
            results["defects"] = defect_results.get("objects", {})
            print(f"   Classified {len(results['defects'])} objects")
        
        # Step 3: Image Captioning (BLIP)
        if self.blip_captioner is not None:
            print("🔍 Step 3: Image Captioning (BLIP)...")
            caption_results = self.blip_captioner.process_detections(processed_frame, detections)
            results["captions"] = caption_results.get("captions", {})
            print(f"   Generated captions for {len(results['captions'])} objects")
        
        # Step 4: Dimension Measurement
        if self.dimension_detector is not None and self.pixel_to_mm_ratio is not None:
            print("🔍 Step 4: Dimension Measurement...")
            dimension_results = self.dimension_detector.measure(
                processed_frame,
                detections,
                self.pixel_to_mm_ratio
            )
            results["dimensions"] = dimension_results
            print(f"   Measured dimensions for {len(detections)} objects")
        
        results["processing_time"] = time.time() - start_time
        return results
    
    def draw_results(
        self,
        frame: np.ndarray,
        results: Dict,
        show_defects: bool = True,
        show_captions: bool = True,
        show_dimensions: bool = True
    ) -> np.ndarray:
        """
        Draw all results on frame.
        
        Args:
            frame: Input frame
            results: Results dictionary from process_frame
            show_defects: Draw defect classifications
            show_captions: Draw captions
            show_dimensions: Draw dimensions
        
        Returns:
            Annotated frame
        """
        output_frame = frame.copy()
        
        # Draw detections
        for det in results.get("detections", []):
            obj_id = det.get("id", 0)
            bbox = det.get("bbox", [0, 0, frame.shape[1], frame.shape[0]])
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw bounding box
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw object ID
            cv2.putText(
                output_frame,
                f"ID: {obj_id}",
                (x1, y1 - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            
            # Draw defect info
            if show_defects and obj_id in results.get("defects", {}):
                defect_info = results["defects"][obj_id]
                defect_class = defect_info.get("predicted_class", "unknown")
                confidence = defect_info.get("confidence", 0)
                
                color = (0, 0, 255) if defect_info.get("has_defect") else (0, 255, 0)
                cv2.putText(
                    output_frame,
                    f"{defect_class}: {confidence:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )
            
            # Draw caption
            if show_captions and obj_id in results.get("captions", {}):
                caption = results["captions"][obj_id].get("caption", "")
                if caption:
                    # Truncate long captions
                    caption = caption[:40] + "..." if len(caption) > 40 else caption
                    cv2.putText(
                        output_frame,
                        caption,
                        (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 0, 0),
                        1
                    )
            
            # Draw dimensions
            if show_dimensions and obj_id in results.get("dimensions", {}):
                dim_info = results["dimensions"][obj_id]
                width = dim_info.get("width", 0)
                height = dim_info.get("height", 0)
                
                cv2.putText(
                    output_frame,
                    f"W:{width:.1f}mm H:{height:.1f}mm",
                    (x1, y2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 0),
                    1
                )
        
        # Draw processing time
        fps = 1.0 / max(results.get("processing_time", 0.001), 0.001)
        cv2.putText(
            output_frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        return output_frame
    
    def print_results(self, results: Dict):
        """Print results in human-readable format."""
        print("\n" + "="*60)
        print("INSPECTION RESULTS")
        print("="*60)
        print(f"Processing time: {results['processing_time']:.3f}s")
        print(f"Objects detected: {len(results['detections'])}")
        
        for det in results.get("detections", []):
            obj_id = det.get("id", 0)
            print(f"\n--- Object {obj_id} ---")
            print(f"  Confidence: {det.get('confidence', 0):.2f}")
            
            if obj_id in results.get("defects", {}):
                defect = results["defects"][obj_id]
                print(f"  Defect: {defect.get('predicted_class')} ({defect.get('confidence', 0):.2f})")
            
            if obj_id in results.get("captions", {}):
                caption = results["captions"][obj_id].get("caption", "")
                print(f"  Caption: {caption}")
            
            if obj_id in results.get("dimensions", {}):
                dims = results["dimensions"][obj_id]
                print(f"  Dimensions: {dims.get('width', 0):.1f}mm x {dims.get('height', 0):.1f}mm")
        
        print("\n" + "="*60 + "\n")
