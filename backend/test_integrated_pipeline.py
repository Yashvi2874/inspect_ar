"""
Test script for integrated inspection pipeline
Tests all components together with sample images
"""

import cv2
import numpy as np
from pathlib import Path
import time
import os

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from integrated_inspection_pipeline import IntegratedInspectionPipeline


def find_sample_images(dataset_path: Path, max_images: int = 3) -> list:
    """Find sample images from dataset."""
    images = []
    
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                images.append(Path(root) / file)
                if len(images) >= max_images:
                    return images
    
    return images


def test_pipeline():
    """Test the integrated pipeline."""
    
    print("\n" + "="*70)
    print("INTEGRATED INSPECTION PIPELINE TEST")
    print("="*70 + "\n")
    
    # Initialize pipeline
    print("🚀 Initializing pipeline...")
    pipeline = IntegratedInspectionPipeline(
        models_folder=Path("backend/models"),
        use_vgg16=True,
        use_blip=True,
        use_dimensions=True
    )
    
    # Find sample images
    dataset_path = Path("aircraft_damage_dataset_v1/test")
    if not dataset_path.exists():
        print(f"⚠️ Dataset not found at {dataset_path}")
        print("📌 Using a test image instead...")
        
        # Create a test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        sample_images = [("generated_test.jpg", test_image)]
    else:
        sample_images = find_sample_images(dataset_path, max_images=3)
        print(f"✅ Found {len(sample_images)} sample images")
    
    if not sample_images:
        print("❌ No sample images found!")
        return
    
    # Test each image
    total_time = 0
    total_frames = 0
    
    for idx, image_path in enumerate(sample_images, 1):
        print(f"\n{'='*70}")
        print(f"TEST {idx}: Processing image")
        print(f"{'='*70}")
        
        # Load image
        if isinstance(image_path, tuple):
            image_name, frame = image_path
        else:
            image_name = image_path.name
            frame = cv2.imread(str(image_path))
        
        if frame is None:
            print(f"⚠️ Could not load image: {image_name}")
            continue
        
        print(f"📷 Image: {image_name}")
        print(f"📏 Size: {frame.shape[1]}x{frame.shape[0]}")
        
        # Process frame
        print("\n🔄 Processing frame...")
        start = time.time()
        results = pipeline.process_frame(frame, conf_threshold=0.25)
        elapsed = time.time() - start
        
        total_time += elapsed
        total_frames += 1
        
        # Print results
        pipeline.print_results(results)
        
        # Draw results
        print("🎨 Drawing results...")
        output_frame = pipeline.draw_results(
            frame,
            results,
            show_defects=True,
            show_captions=True,
            show_dimensions=True
        )
        
        # Save output
        output_path = Path("backend/test_output") / f"result_{idx}.jpg"
        output_path.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(output_path), output_frame)
        print(f"✅ Result saved to {output_path}")
        
        # Performance metrics
        fps = 1.0 / elapsed if elapsed > 0 else 0
        print(f"\n⏱️  Processing time: {elapsed:.3f}s")
        print(f"📊 FPS: {fps:.1f}")
    
    # Summary
    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}")
    print(f"Total frames processed: {total_frames}")
    print(f"Total time: {total_time:.3f}s")
    if total_frames > 0:
        avg_fps = total_frames / total_time
        print(f"Average FPS: {avg_fps:.1f}")
        print(f"Average time per frame: {total_time/total_frames:.3f}s")
    
    # Performance check
    if total_frames > 0:
        avg_fps = total_frames / total_time
        if avg_fps >= 15:
            print(f"\n✅ PERFORMANCE OK: {avg_fps:.1f} FPS (target: 15+ FPS)")
        else:
            print(f"\n⚠️ PERFORMANCE WARNING: {avg_fps:.1f} FPS (target: 15+ FPS)")
    
    print(f"\n{'='*70}\n")


def test_real_time_video():
    """Test with webcam or video file."""
    print("\n" + "="*70)
    print("REAL-TIME VIDEO TEST")
    print("="*70 + "\n")
    
    # Initialize pipeline
    print("🚀 Initializing pipeline...")
    pipeline = IntegratedInspectionPipeline(
        models_folder=Path("backend/models"),
        use_vgg16=True,
        use_blip=False,  # Disable BLIP for real-time (slower)
        use_dimensions=True
    )
    
    # Open video source (0 = webcam)
    print("📹 Opening video source (webcam)...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("⚠️ Could not open webcam. Skipping real-time test.")
        return
    
    print("✅ Webcam opened. Press 'q' to quit, 'c' to calibrate")
    
    calibrated = False
    frame_count = 0
    fps_times = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Resize for faster processing
            frame = cv2.resize(frame, (640, 480))
            
            # Process frame
            start = time.time()
            results = pipeline.process_frame(frame, conf_threshold=0.25)
            elapsed = time.time() - start
            fps_times.append(elapsed)
            
            # Draw results
            output_frame = pipeline.draw_results(
                frame,
                results,
                show_defects=True,
                show_captions=False,  # Disabled for speed
                show_dimensions=calibrated
            )
            
            # Display
            cv2.imshow("Integrated Inspection Pipeline", output_frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                print("\n📌 Calibrating...")
                if pipeline.calibrate(frame, marker_size_mm=50.0):
                    calibrated = True
                    print("✅ Calibration successful")
                else:
                    print("⚠️ Calibration failed")
            
            # Print stats every 30 frames
            if frame_count % 30 == 0:
                avg_fps = 30 / sum(fps_times[-30:]) if fps_times else 0
                print(f"Frame {frame_count}: {avg_fps:.1f} FPS")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        if fps_times:
            avg_fps = len(fps_times) / sum(fps_times)
            print(f"\n✅ Real-time test completed")
            print(f"   Frames processed: {frame_count}")
            print(f"   Average FPS: {avg_fps:.1f}")


if __name__ == "__main__":
    import sys
    
    print("\n🔬 INTEGRATED INSPECTION PIPELINE TEST SUITE\n")
    print("Available tests:")
    print("  1. Static image test (default)")
    print("  2. Real-time video test")
    print("\nUsage: python test_integrated_pipeline.py [1|2]")
    
    test_type = sys.argv[1] if len(sys.argv) > 1 else "1"
    
    if test_type == "2":
        test_real_time_video()
    else:
        test_pipeline()
