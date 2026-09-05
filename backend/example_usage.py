"""
Simple example usage of the integrated inspection pipeline
"""

import cv2
import os
import numpy as np
from pathlib import Path

# Suppress warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Paths are anchored to this file, not to the caller's working directory. The
# docs say to run this from backend/, but these paths used to be written
# repo-root-relative ("backend/models"), so from backend/ they resolved to
# backend/backend/models: the model folder was missing, the dataset was not
# found, and the output directory raised FileNotFoundError.
BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parent
DATASET_DIR = REPO_ROOT / "aircraft_damage_dataset_v1"

from integrated_inspection_pipeline import IntegratedInspectionPipeline


def example_1_basic_usage():
    """Example 1: Basic usage with default settings"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Usage")
    print("="*70)
    
    # Initialize pipeline
    pipeline = IntegratedInspectionPipeline()
    
    # Load image
    image_path = str(DATASET_DIR / "test" / "dent" / "144_10_JPG_jpg.rf.4d008cc33e217c1606b76585469d626b.jpg")
    if not Path(image_path).exists():
        print(f"⚠️ Image not found: {image_path}")
        return
    
    frame = cv2.imread(image_path)
    print(f"✅ Loaded image: {image_path}")
    
    # Process
    results = pipeline.process_frame(frame)
    
    # Print results
    pipeline.print_results(results)


def example_2_with_visualization():
    """Example 2: Process and visualize results"""
    print("\n" + "="*70)
    print("EXAMPLE 2: With Visualization")
    print("="*70)
    
    # Initialize
    pipeline = IntegratedInspectionPipeline()
    
    # Load image
    image_path = str(DATASET_DIR / "test" / "dent" / "144_10_JPG_jpg.rf.4d008cc33e217c1606b76585469d626b.jpg")
    if not Path(image_path).exists():
        print(f"⚠️ Image not found: {image_path}")
        return
    
    frame = cv2.imread(image_path)
    
    # Process
    results = pipeline.process_frame(frame)
    
    # Draw results
    output = pipeline.draw_results(frame, results)
    
    # Save
    output_path = "example_output.jpg"
    cv2.imwrite(output_path, output)
    print(f"✅ Saved visualization to {output_path}")


def example_3_fast_mode():
    """Example 3: Fast mode (disable BLIP for speed)"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Fast Mode (No BLIP)")
    print("="*70)
    
    # Initialize without BLIP
    pipeline = IntegratedInspectionPipeline(use_blip=False)
    
    # Load image
    image_path = str(DATASET_DIR / "test" / "dent" / "144_10_JPG_jpg.rf.4d008cc33e217c1606b76585469d626b.jpg")
    if not Path(image_path).exists():
        print(f"⚠️ Image not found: {image_path}")
        return
    
    frame = cv2.imread(image_path)
    
    # Process
    results = pipeline.process_frame(frame)
    
    # Print results
    print(f"Processing time: {results['processing_time']:.3f}s")
    print(f"Objects detected: {len(results['detections'])}")
    print(f"Defects found: {len(results['defects'])}")


def example_4_with_calibration():
    """Example 4: With dimension measurement (requires calibration)"""
    print("\n" + "="*70)
    print("EXAMPLE 4: With Dimension Measurement")
    print("="*70)
    
    # Initialize
    pipeline = IntegratedInspectionPipeline(use_blip=False)
    
    # Load image
    image_path = str(DATASET_DIR / "test" / "dent" / "144_10_JPG_jpg.rf.4d008cc33e217c1606b76585469d626b.jpg")
    if not Path(image_path).exists():
        print(f"⚠️ Image not found: {image_path}")
        return
    
    frame = cv2.imread(image_path)
    
    # Try to calibrate
    print("📌 Attempting calibration...")
    if pipeline.calibrate(frame, marker_size_mm=50.0):
        print("✅ Calibration successful")
        
        # Process
        results = pipeline.process_frame(frame)
        
        # Print dimensions
        if results['dimensions']:
            print("\n📏 Dimensions:")
            for obj_id, dims in results['dimensions'].items():
                print(f"  Object {obj_id}:")
                print(f"    Width: {dims.get('width_mm', 0):.1f}mm")
                print(f"    Height: {dims.get('height_mm', 0):.1f}mm")
                print(f"    Area: {dims.get('area_mm2', 0):.1f}mm²")
        else:
            print("⚠️ No dimensions measured")
    else:
        print("⚠️ Calibration failed (ArUco marker not detected)")


def example_5_batch_processing():
    """Example 5: Batch process multiple images"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Batch Processing")
    print("="*70)
    
    # Initialize
    pipeline = IntegratedInspectionPipeline(use_blip=False)
    
    # Find images
    dataset_path = DATASET_DIR / "test"
    if not dataset_path.exists():
        print(f"⚠️ Dataset not found: {dataset_path}")
        return
    
    images = list(dataset_path.glob("**/*.jpg"))[:5]  # First 5 images
    print(f"📷 Found {len(images)} images")
    
    # Process each
    total_time = 0
    for idx, image_path in enumerate(images, 1):
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        
        results = pipeline.process_frame(frame)
        total_time += results['processing_time']
        
        print(f"  {idx}. {image_path.name}: {len(results['detections'])} objects, "
              f"{results['processing_time']:.3f}s")
    
    avg_time = total_time / len(images) if images else 0
    avg_fps = 1.0 / avg_time if avg_time > 0 else 0
    print(f"\n📊 Average: {avg_time:.3f}s per image ({avg_fps:.1f} FPS)")


def example_6_custom_configuration():
    """Example 6: Custom configuration"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Custom Configuration")
    print("="*70)
    
    # Initialize with custom settings
    pipeline = IntegratedInspectionPipeline(
        models_folder=BACKEND_DIR / "models",
        use_vgg16=True,      # Enable defect detection
        use_blip=False,      # Disable for speed
        use_dimensions=True  # Enable measurements
    )
    
    print("✅ Pipeline configured:")
    print("   - VGG16 defect detection: ENABLED")
    print("   - BLIP captioning: DISABLED")
    print("   - Dimension measurement: ENABLED")
    
    # Load image
    image_path = str(DATASET_DIR / "test" / "dent" / "144_10_JPG_jpg.rf.4d008cc33e217c1606b76585469d626b.jpg")
    if not Path(image_path).exists():
        print(f"⚠️ Image not found: {image_path}")
        return
    
    frame = cv2.imread(image_path)
    
    # Process
    results = pipeline.process_frame(frame)
    
    # Print summary
    print(f"\n📊 Results:")
    print(f"   Objects detected: {len(results['detections'])}")
    print(f"   Defects classified: {len(results['defects'])}")
    print(f"   Dimensions measured: {len(results['dimensions'])}")
    print(f"   Processing time: {results['processing_time']:.3f}s")


def example_7_error_handling():
    """Example 7: Error handling"""
    print("\n" + "="*70)
    print("EXAMPLE 7: Error Handling")
    print("="*70)
    
    # Initialize
    pipeline = IntegratedInspectionPipeline()
    
    # Try with non-existent image
    print("📌 Testing with non-existent image...")
    try:
        frame = cv2.imread("non_existent_image.jpg")
        if frame is None:
            print("⚠️ Image not found, creating dummy frame")
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        results = pipeline.process_frame(frame)
        print(f"✅ Processed successfully: {len(results['detections'])} objects")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Try with invalid frame
    print("\n📌 Testing with invalid frame...")
    try:
        invalid_frame = None
        if invalid_frame is not None:
            results = pipeline.process_frame(invalid_frame)
        else:
            print("⚠️ Invalid frame, skipping")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("INTEGRATED INSPECTION PIPELINE - EXAMPLES")
    print("="*70)
    
    print("\nAvailable examples:")
    print("  1. Basic usage")
    print("  2. With visualization")
    print("  3. Fast mode (no BLIP)")
    print("  4. With dimension measurement")
    print("  5. Batch processing")
    print("  6. Custom configuration")
    print("  7. Error handling")
    print("  0. Run all examples")
    
    choice = input("\nSelect example (0-7): ").strip()
    
    examples = {
        "1": example_1_basic_usage,
        "2": example_2_with_visualization,
        "3": example_3_fast_mode,
        "4": example_4_with_calibration,
        "5": example_5_batch_processing,
        "6": example_6_custom_configuration,
        "7": example_7_error_handling,
    }
    
    if choice == "0":
        for example_func in examples.values():
            try:
                example_func()
            except Exception as e:
                print(f"❌ Error: {e}")
    elif choice in examples:
        try:
            examples[choice]()
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print("❌ Invalid choice")
    
    print("\n" + "="*70)
    print("Examples completed!")
    print("="*70 + "\n")
