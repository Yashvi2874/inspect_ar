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
from models.defect_detector import DefectDetector
from models.blip_wrapper import BLIPCaptioner
from models.dimension_detector import DimensionDetector
from models.aruco_calibration import ArucoCalibrator
from utils.image_processor import ImageProcessor

# models.vgg16_wrapper is imported lazily, inside __init__, because importing it
# pulls in the whole TensorFlow runtime. Loading TensorFlow and PyTorch into one
# process segfaults on some setups (reproduced here), and the VGG16 stage is off
# by default, so paying that cost at module scope would be both risky and
# pointless.

# NOTE: `import industrial_measurement` used to sit here. That module has no
# __main__ guard: importing it calls input() twice and opens a webcam at module
# scope, so this file could never be imported at all. The alias was referenced
# zero times. Do not reinstate it without wrapping that module in a main().

# Everything below resolves relative to backend/, not to the caller's working
# directory, so the pipeline behaves the same from any cwd.
BACKEND_DIR = Path(__file__).resolve().parent


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
        use_defect_detector: bool = True,
        use_yolo: bool = False,
        use_vgg16: bool = False,
        use_blip: bool = True,
        use_dimensions: bool = True
    ):
        """
        Initialize the integrated pipeline.

        Args:
            models_folder: Path to models directory. Defaults to backend/models.
            use_defect_detector: Use the trained Faster R-CNN damage detector.
                This is the only stage in the project trained on aircraft
                damage, so it is on by default and it both localises and labels
                damage in one pass.
            use_yolo: Also run YOLO. Off by default: the bundled weights are
                stock COCO, which has no damage classes and reports things like
                "clock" on aircraft skin.
            use_vgg16: Run the Keras VGG16 classifier. Off by default: no
                trained weights for it exist anywhere in the project, so it
                would classify using a randomly initialised head.
            use_blip: Enable BLIP captioning.
            use_dimensions: Enable dimension measurement.
        """
        self.models_folder = Path(models_folder) if models_folder else BACKEND_DIR / "models"

        print("Initializing Integrated Inspection Pipeline...")

        # Trained damage detector — the primary detection stage
        if use_defect_detector:
            print("  Loading Faster R-CNN defect detector...")
            self.defect_detector = DefectDetector(
                checkpoint_path=self.models_folder / "best_model.pth"
            )
        else:
            self.defect_detector = None

        # Generic object detector (optional, stock COCO classes)
        if use_yolo:
            print("  Loading YOLO detector...")
            self.yolo_detector = YOLODetector(models_folder=self.models_folder)
        else:
            self.yolo_detector = None

        # Legacy Keras classifier (optional, untrained)
        if use_vgg16:
            print("  Loading VGG16 defect detector...")
            from models.vgg16_wrapper import VGG16DefectDetector  # pulls in TensorFlow
            checkpoint = self.models_folder / "best_model.pth"
            self.vgg16_detector = VGG16DefectDetector(checkpoint_path=checkpoint)
        else:
            self.vgg16_detector = None

        if self.defect_detector is None and self.yolo_detector is None:
            raise ValueError(
                "No detection stage enabled: set use_defect_detector or use_yolo."
            )

        # Initialize captioner
        if use_blip:
            print("  Loading BLIP captioner...")
            # BLIP fetches ~1 GB from Hugging Face on first use. That fails on
            # an offline machine, and captioning is the least essential stage,
            # so a failure here degrades the pipeline instead of stopping it.
            # (A native crash from an out-of-memory load cannot be caught here;
            # if the process dies outright, run with use_blip=False.)
            try:
                self.blip_captioner = BLIPCaptioner()
            except Exception as e:
                print(f"  [warn] BLIP unavailable, captions disabled: {e}")
                self.blip_captioner = None
        else:
            self.blip_captioner = None

        # Initialize dimension detector
        if use_dimensions:
            print("  Loading dimension detector...")
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

        print("[ok] Pipeline initialized successfully")
    
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
            print("[warn] Dimension detector not enabled")
            return False

        try:
            print("Calibrating camera...")
            self.calibration_data = self.aruco_calibrator.detect_and_calibrate(
                frame,
                marker_size_mm=marker_size_mm
            )

            # Test the success flag, not merely the presence of the ratio key.
            # The key is always present; on failure its value is None.
            if self.calibration_data.get("success"):
                self.pixel_to_mm_ratio = self.calibration_data["pixel_to_mm_ratio"]
                print(f"[ok] Calibration successful. Ratio: {self.pixel_to_mm_ratio:.4f} mm/px")
                return True

            self.pixel_to_mm_ratio = None
            reason = self.calibration_data.get("error", "marker not detected")
            print(f"[warn] Calibration failed - {reason}")
            return False
        except Exception as e:
            self.pixel_to_mm_ratio = None
            print(f"[warn] Calibration error: {e}")
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
        
        # Step 1: Damage detection (trained Faster R-CNN)
        detections = []
        if self.defect_detector is not None:
            print("Step 1: Damage detection (Faster R-CNN)...")
            # The detector runs on the ORIGINAL frame, not the preprocessed one.
            # ImageProcessor.preprocess collapses the image to grayscale and back
            # to 3 identical channels; the detector was trained on colour.
            detections = self.defect_detector.detect_frame(frame)
            results["detections"] = detections
            # This model localises and labels in one pass, so the defect map is
            # a direct read of the same detections rather than a second stage.
            results["defects"] = {
                d["id"]: {
                    "predicted_class": d["class"],
                    "class_id": d["class_id"],
                    "confidence": d["confidence"],
                    "has_defect": True,
                    "untrained_class": d["untrained_class"],
                }
                for d in detections
            }
            print(f"   Found {len(detections)} damage regions")

        # Step 1b: Generic object detection (optional, stock COCO classes)
        if self.yolo_detector is not None:
            print("Step 1b: Object detection (YOLO)...")
            yolo_results = self.yolo_detector.detect(processed_frame, conf_threshold=conf_threshold)
            yolo_detections = yolo_results.get("detections", [])
            # Keep ids unique across both detectors.
            offset = len(detections)
            for i, d in enumerate(yolo_detections):
                d["id"] = offset + i
            detections = detections + yolo_detections
            results["detections"] = detections
            print(f"   Found {len(yolo_detections)} generic objects")

        if len(detections) == 0:
            results["processing_time"] = time.time() - start_time
            return results

        # Step 2: Defect Classification (VGG16, optional legacy path)
        if self.vgg16_detector is not None:
            print("Step 2: Defect Classification (VGG16)...")
            defect_results = self.vgg16_detector.detect(processed_frame, detections)
            results["defects"] = defect_results.get("objects", {})
            print(f"   Classified {len(results['defects'])} objects")

        # Step 3: Image Captioning (BLIP)
        if self.blip_captioner is not None:
            print("Step 3: Image Captioning (BLIP)...")
            caption_results = self.blip_captioner.process_detections(processed_frame, detections)
            results["captions"] = caption_results.get("captions", {})
            print(f"   Generated captions for {len(results['captions'])} objects")

        # Step 4: Dimension Measurement
        if self.dimension_detector is not None and self.pixel_to_mm_ratio is not None:
            print("Step 4: Dimension Measurement...")
            dimension_results = self.dimension_detector.measure(
                processed_frame,
                detections,
                self.pixel_to_mm_ratio
            )
            # measure() returns {'objects': {...}, 'overall_confidence': ...}.
            # Assigning that envelope whole meant every `obj_id in
            # results["dimensions"]` test compared an int against the keys
            # "objects"/"overall_confidence" and was always False, so measured
            # dimensions could never reach the display.
            results["dimensions"] = dimension_results.get("objects", {})
            results["dimension_confidence"] = dimension_results.get("overall_confidence")
            print(f"   Measured dimensions for {len(results['dimensions'])} objects")

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
                # DimensionDetector.measure emits width_mm/height_mm/area_mm2.
                width = dim_info.get("width_mm", 0)
                height = dim_info.get("height_mm", 0)

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
                print(f"  Dimensions: {dims.get('width_mm', 0):.1f}mm x {dims.get('height_mm', 0):.1f}mm")
        
        print("\n" + "="*60 + "\n")
