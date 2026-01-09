"""
ZENITH INSPECT-AR - Industrial Object Measurement System (Production Version)
Real-time dimension measurement using ArUco marker calibration
"""

import cv2
import numpy as np
from datetime import datetime
import math
import time
from pathlib import Path
import sys

# Add models directory to path
sys.path.append(str(Path(__file__).parent))
from models.yolo_detector import YOLODetector

# ArUco dictionaries
ARUCO_DICTS = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
}

# Get marker size and dictionary
print("Enter marker size in mm (default: 50mm): ", end="")
marker_input = input().strip()
KNOWN_MARKER_SIZE_MM = float(marker_input) if marker_input else 50.0

print("Select ArUco Dictionary (1-3, default: 3): ", end="")
dict_input = input().strip()
dict_choice = int(dict_input) if dict_input.isdigit() and 1 <= int(dict_input) <= 3 else 3
dict_name = list(ARUCO_DICTS.keys())[dict_choice - 1]
aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[dict_name])

# Initialize detector
aruco_params = cv2.aruco.DetectorParameters()
aruco_params.adaptiveThreshWinSizeMin = 3
aruco_params.adaptiveThreshWinSizeMax = 23
aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
aruco_params.cornerRefinementWinSize = 5
aruco_params.minMarkerPerimeterRate = 0.03

detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# Calibration function
def compute_pixel_to_mm_ratio(marker_corners, known_size_mm):
    corner_points = marker_corners[0]
    # Measure all 4 sides for better accuracy
    side1 = np.linalg.norm(corner_points[0] - corner_points[1])  # Top
    side2 = np.linalg.norm(corner_points[1] - corner_points[2])  # Right
    side3 = np.linalg.norm(corner_points[2] - corner_points[3])  # Bottom
    side4 = np.linalg.norm(corner_points[3] - corner_points[0])  # Left
    
    # Use median for robustness
    all_sides = np.array([side1, side2, side3, side4])
    avg_marker_size_px = np.median(all_sides)
    
    # Calculate conversion ratio
    pixel_to_mm_ratio = known_size_mm / avg_marker_size_px
    
    return pixel_to_mm_ratio, avg_marker_size_px

# ROI definition
def define_roi_near_marker(marker_corners, frame_shape, roi_offset_ratio=1.2, roi_size_ratio=6.0):
    corner_points = marker_corners[0]
    x_coords = corner_points[:, 0]
    y_coords = corner_points[:, 1]
    marker_x = int(np.min(x_coords))
    marker_y = int(np.min(y_coords))
    marker_width = int(np.max(x_coords) - np.min(x_coords))
    marker_height = int(np.max(y_coords) - np.min(y_coords))
    
    roi_x = marker_x + int(marker_width * roi_offset_ratio)
    roi_y = marker_y - int(marker_height * 0.5)
    roi_width = int(marker_width * roi_size_ratio)
    roi_height = int(marker_height * roi_size_ratio)
    
    frame_height, frame_width = frame_shape[:2]
    roi_x = max(0, min(roi_x, frame_width - roi_width))
    roi_y = max(0, min(roi_y, frame_height - roi_height))
    roi_width = min(roi_width, frame_width - roi_x)
    roi_height = min(roi_height, frame_height - roi_y)
    
    roi_rect = (roi_x, roi_y, roi_width, roi_height)
    roi_center = (roi_x + roi_width // 2, roi_y + roi_height // 2)
    
    return roi_rect, roi_center

# Object detection
def detect_object_in_roi(frame, roi_rect, min_area_px=300):
    x, y, w, h = roi_rect
    roi_frame = frame[y:y+h, x:x+w].copy()
    
    gray_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray_roi, d=9, sigmaColor=75, sigmaSpace=75)
    
    binary = cv2.adaptiveThreshold(
        blurred, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        blockSize=15, 
        C=3
    )
    
    kernel_small = np.ones((3, 3), np.uint8)
    kernel_large = np.ones((5, 5), np.uint8)
    
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_large, iterations=3)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_small, iterations=1)
    
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    largest_contour = None
    max_area = min_area_px
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            largest_contour = contour
    
    return largest_contour, roi_frame, binary

def detect_object_with_yolo_in_roi(frame, roi_rect, yolo_detector, marker_corners=None, min_confidence=0.3):
    """
    Use YOLO to detect objects within the ROI, prioritizing objects in the same plane as the marker
    Returns: (best_detection, roi_frame, contour)
    """
    x, y, w, h = roi_rect
    roi_frame = frame[y:y+h, x:x+w].copy()
    
    # Run YOLO detection on ROI
    result = yolo_detector.detect(roi_frame)
    
    if result['count'] == 0:
        return None, roi_frame, None
    
    # Calculate marker reference for depth/size comparison
    marker_reference_size = None
    marker_center = None
    if marker_corners is not None:
        corner_points = marker_corners[0]
        # Calculate marker size (average of all sides)
        side1 = np.linalg.norm(corner_points[0] - corner_points[1])
        side2 = np.linalg.norm(corner_points[1] - corner_points[2])
        side3 = np.linalg.norm(corner_points[2] - corner_points[3])
        side4 = np.linalg.norm(corner_points[3] - corner_points[0])
        marker_reference_size = np.mean([side1, side2, side3, side4])
        
        # Calculate marker center in absolute coordinates
        marker_center = np.mean(corner_points, axis=0)
    
    # Find the best detection (prioritize objects in same plane as marker)
    best_detection = None
    best_score = 0
    
    roi_center_x = w / 2
    roi_center_y = h / 2
    
    for detection in result['detections']:
        if detection['confidence'] < min_confidence:
            continue
        
        # Get detection bbox
        bbox = detection['bbox']  # [x1, y1, x2, y2]
        det_width = bbox[2] - bbox[0]
        det_height = bbox[3] - bbox[1]
        det_size = (det_width + det_height) / 2  # Average size
        det_center_x = (bbox[0] + bbox[2]) / 2
        det_center_y = (bbox[1] + bbox[3]) / 2
        
        # Convert detection center to absolute coordinates
        det_center_abs_x = det_center_x + x
        det_center_abs_y = det_center_y + y
        
        # Score components
        confidence_score = detection['confidence']
        
        # 1. Proximity to ROI center (prefer centered objects)
        distance_to_roi_center = np.sqrt((det_center_x - roi_center_x)**2 + (det_center_y - roi_center_y)**2)
        proximity_score = 1.0 - min(distance_to_roi_center / (w/2), 1.0)
        
        # 2. Depth similarity to marker (based on size comparison)
        depth_score = 1.0
        if marker_reference_size is not None:
            # Objects in same plane should have similar apparent size relative to marker
            # Use size ratio as depth indicator
            size_ratio = det_size / marker_reference_size
            
            # Prefer objects with size between 0.5x to 3x the marker size
            # (objects too small or too large are likely at different depths)
            if 0.5 <= size_ratio <= 3.0:
                depth_score = 1.0  # Good size range, likely same plane
            elif size_ratio < 0.5:
                # Much smaller - likely farther away or small noise
                depth_score = 0.3
            else:
                # Much larger - likely closer (foreground)
                depth_score = 0.4
        
        # 3. Distance to marker (prefer objects close to marker)
        marker_distance_score = 1.0
        if marker_center is not None:
            distance_to_marker = np.sqrt((det_center_abs_x - marker_center[0])**2 + 
                                        (det_center_abs_y - marker_center[1])**2)
            # Normalize by frame diagonal
            frame_diagonal = np.sqrt(w**2 + h**2)
            normalized_distance = min(distance_to_marker / frame_diagonal, 1.0)
            marker_distance_score = 1.0 - normalized_distance * 0.5  # Weight distance moderately
        
        # Combined score with weights
        # Depth similarity is most important (0.5), then confidence (0.3), then proximity (0.2)
        score = (depth_score * 0.5 + 
                confidence_score * 0.3 + 
                proximity_score * 0.15 +
                marker_distance_score * 0.05)
        
        if score > best_score:
            best_score = score
            best_detection = detection
    
    if best_detection is None:
        return None, roi_frame, None
    
    # Create a contour from the detection bbox for measurement
    bbox = best_detection['bbox']
    x1, y1, x2, y2 = [int(coord) for coord in bbox]
    
    # Create a rectangular contour
    contour = np.array([
        [[x1, y1]],
        [[x2, y1]],
        [[x2, y2]],
        [[x1, y2]]
    ], dtype=np.int32)
    
    return best_detection, roi_frame, contour

# Measurement with rotation
def measure_object_with_rotation(contour, pixel_to_mm_ratio):
    if contour is None:
        return None, None
    
    rect = cv2.minAreaRect(contour)
    center, size, angle = rect
    width_px, height_px = size
    
    if width_px < height_px:
        width_px, height_px = height_px, width_px
        angle = angle + 90
    
    width_mm = (width_px * pixel_to_mm_ratio) - 0.1 ####################
    height_mm = (height_px * pixel_to_mm_ratio) - 0.1 ###########################
    area_px = cv2.contourArea(contour)
    area_mm2 = area_px * (pixel_to_mm_ratio ** 2)
    perimeter_px = cv2.arcLength(contour, closed=True)
    perimeter_mm = perimeter_px * pixel_to_mm_ratio
    
    x, y, bbox_w, bbox_h = cv2.boundingRect(contour)
    bbox_width_mm = bbox_w * pixel_to_mm_ratio
    bbox_height_mm = bbox_h * pixel_to_mm_ratio
    
    aspect_ratio = width_mm / height_mm if height_mm > 0 else 0
    compactness = (4 * math.pi * area_px) / (perimeter_px ** 2) if perimeter_px > 0 else 0
    
    measurements = {
        'width_mm': width_mm,
        'height_mm': height_mm,
        'area_mm2': area_mm2,
        'perimeter_mm': perimeter_mm,
        'angle_deg': angle,
        'bbox_width_mm': bbox_width_mm,
        'bbox_height_mm': bbox_height_mm,
        'aspect_ratio': aspect_ratio,
        'compactness': compactness,
        'center': center
    }
    
    return measurements, rect

# Visualization functions
def draw_rotated_rectangle(frame, rect, roi_offset, color=(0, 255, 0), thickness=2):
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    box[:, 0] += roi_offset[0]
    box[:, 1] += roi_offset[1]
    cv2.drawContours(frame, [box], 0, color, thickness)
    return box

def draw_measurement_overlay(frame, measurements, position, object_label="OBJECT"):
    x, y = position
    panel_width = 380
    panel_height = 340
    
    x = max(10, min(x, frame.shape[1] - panel_width - 10))
    y = max(10, min(y, frame.shape[0] - panel_height - 10))
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + panel_width, y + panel_height),
                  (20, 25, 30), -1)
    cv2.rectangle(overlay, (x, y), (x + panel_width, y + panel_height),
                  (14, 165, 233), 3)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
    
    cv2.putText(frame, f"📦 {object_label}", (x + 15, y + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (14, 165, 233), 2)
    cv2.line(frame, (x + 15, y + 48), (x + panel_width - 15, y + 48),
             (14, 165, 233), 1)
    
    y_offset = y + 75
    line_spacing = 32
    
    metrics = [
        ("Width (Rotated):", f"{measurements['width_mm']:.2f} mm", (0, 255, 255)),
        ("Height (Rotated):", f"{measurements['height_mm']:.2f} mm", (0, 255, 255)),
        ("", "", None),  # Spacer
        ("Bounding Width:", f"{measurements['bbox_width_mm']:.2f} mm", (200, 200, 200)),
        ("Bounding Height:", f"{measurements['bbox_height_mm']:.2f} mm", (200, 200, 200)),
        ("", "", None),  # Spacer
        ("Area:", f"{measurements['area_mm2']:.2f} mm²", (0, 255, 0)),
        ("Perimeter:", f"{measurements['perimeter_mm']:.2f} mm", (255, 200, 0)),
        ("Rotation Angle:", f"{measurements['angle_deg']:.1f}°", (255, 100, 255)),
        ("Aspect Ratio:", f"{measurements['aspect_ratio']:.2f}", (100, 200, 255)),
    ]
    
    for i, (label, value, color) in enumerate(metrics):
        y_pos = y_offset + i * line_spacing
        if label:  # Skip spacers
            cv2.putText(frame, label, (x + 20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            cv2.putText(frame, value, (x + 200, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

def draw_dimension_lines(frame, box, measurements):
    mid_bottom = ((box[0] + box[1]) // 2).astype(int)
    cv2.putText(frame, f"{measurements['width_mm']:.1f}mm",
                tuple(mid_bottom + [0, 25]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    mid_left = ((box[0] + box[3]) // 2).astype(int)
    cv2.putText(frame, f"{measurements['height_mm']:.1f}mm",
                tuple(mid_left - [80, 0]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

def draw_compact_info_panel(frame, marker_size_px, object_size_px=None):
    """
    Draw a compact info panel in top right corner with pixel dimensions
    Args:
        marker_size_px: Tuple of (width, height) in pixels for marker
        object_size_px: Tuple of (width, height) in pixels for object (or None)
    """
    panel_width = 300
    panel_height = 140
    margin = 15
    
    # Position in top right
    x = frame.shape[1] - panel_width - margin
    y = 80  # Below header
    
    # Draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + panel_width, y + panel_height),
                  (20, 25, 30), -1)
    cv2.rectangle(overlay, (x, y), (x + panel_width, y + panel_height),
                  (14, 165, 233), 2)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    
    # Title
    cv2.putText(frame, "DIMENSIONS (pixels)", (x + 15, y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (14, 165, 233), 2)
    cv2.line(frame, (x + 15, y + 32), (x + panel_width - 15, y + 32),
             (14, 165, 233), 1)
    
    y_offset = y + 55
    line_spacing = 28
    
    # Marker dimensions in pixels
    cv2.putText(frame, "Marker L x W:", (x + 15, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    if marker_size_px is not None:
        marker_text = f"{marker_size_px[0]:.0f} x {marker_size_px[1]:.0f} px"
        cv2.putText(frame, marker_text, (x + 150, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    else:
        cv2.putText(frame, "--- x --- px", (x + 150, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 2)
    
    # Object dimensions in pixels
    y_offset += line_spacing
    cv2.putText(frame, "Object L x W:", (x + 15, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    
    if object_size_px is not None:
        object_text = f"{object_size_px[0]:.0f} x {object_size_px[1]:.0f} px"
        text_color = (0, 255, 0)
    else:
        object_text = "--- x --- px"
        text_color = (100, 100, 100)
    
    cv2.putText(frame, object_text, (x + 150, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

# Enhanced smoothing function to reduce measurement fluctuation
def apply_advanced_smoothing(object_id, current_measurements):
    """
    Apply advanced smoothing to reduce measurement fluctuation
    """
    global measurement_history
    
    if object_id not in measurement_history:
        measurement_history[object_id] = {
            'measurements_buffer': [],
            'stable_count': 0,
            'is_stable': False
        }
    
    history = measurement_history[object_id]
    history['measurements_buffer'].append(current_measurements.copy())
    
    # Keep only recent measurements
    if len(history['measurements_buffer']) > smoothing_window:
        history['measurements_buffer'] = history['measurements_buffer'][-smoothing_window:]
    
    if len(history['measurements_buffer']) < 3:
        # Not enough data, return current measurements
        return current_measurements
    
    # Calculate moving average for each measurement
    smoothed = {}
    for key in current_measurements:
        if key == 'center':
            # Handle center point separately
            x_vals = [m['center'][0] for m in history['measurements_buffer']]
            y_vals = [m['center'][1] for m in history['measurements_buffer']]
            avg_x = sum(x_vals) / len(x_vals)
            avg_y = sum(y_vals) / len(y_vals)
            smoothed['center'] = (avg_x, avg_y)
        elif isinstance(current_measurements[key], (int, float)):
            values = [m[key] for m in history['measurements_buffer']]
            avg_val = sum(values) / len(values)
            smoothed[key] = avg_val
        else:
            # For non-numeric values, use the current value
            smoothed[key] = current_measurements[key]
    
    # Check for stability
    if len(history['measurements_buffer']) >= min_stable_frames:
        recent_measurements = history['measurements_buffer'][-min_stable_frames:]
        is_stable = True
        
        # Check stability for key measurements
        for key in ['width_mm', 'height_mm', 'area_mm2']:
            if key in current_measurements:
                values = [m[key] for m in recent_measurements]
                if values:
                    min_val = min(values)
                    max_val = max(values)
                    avg_val = sum(values) / len(values)
                    
                    # Calculate relative variation
                    if avg_val != 0:
                        variation = (max_val - min_val) / abs(avg_val)
                        if variation > stability_threshold:
                            is_stable = False
                            break
                    else:
                        if (max_val - min_val) > 0.1:  # Absolute threshold for near-zero values
                            is_stable = False
                            break
        
        if is_stable:
            history['stable_count'] += 1
            if history['stable_count'] >= min_stable_frames:
                history['is_stable'] = True
        else:
            history['stable_count'] = 0
            history['is_stable'] = False
    
    # Apply additional smoothing if measurements are not yet stable
    if not history['is_stable']:
        # Use more conservative smoothing
        if len(history['measurements_buffer']) >= 5:
            # Take median for more robustness
            for key in current_measurements:
                if key != 'center' and isinstance(current_measurements[key], (int, float)):
                    values = [m[key] for m in history['measurements_buffer']]
                    values.sort()
                    # Use median instead of mean for robustness
                    if len(values) % 2 == 0:
                        median_val = (values[len(values)//2 - 1] + values[len(values)//2]) / 2
                    else:
                        median_val = values[len(values)//2]
                    smoothed[key] = median_val
    
    # Round values for stable display
    for key in ['width_mm', 'height_mm', 'area_mm2', 'perimeter_mm', 'angle_deg']:
        if key in smoothed:
            smoothed[key] = round(smoothed[key], 2)
    
    return smoothed

# Initialize webcam
print("Initializing webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Controls: 'q' - Quit, 's' - Save screenshot, 'c' - Contrast, 'r' - Reset, 'u' - Unlock, 'f' - Freeze")

# State variables
pixel_to_mm_ratio = None
calibrated = False
use_contrast = True
frame_count = 0
detection_count = 0

# Enhanced measurement history for advanced smoothing
measurement_history = {}
smoothing_window = 15  # Number of frames to average for stability
min_stable_frames = 8  # Minimum frames to consider measurement stable
stability_threshold = 0.015  # 1.5% threshold for stability

# YOLO detector initialization
yolo_detector = YOLODetector()

# Lock-on mechanism variables
LOCK_THRESHOLD_SECONDS = 5.0  # Lock after 5 seconds of consistent detection
is_locked = False
locked_measurements = None
locked_rect = None
locked_box = None
first_detection_time = None
consistent_detection_timer = 0.0
locked_detection_id = None

# Locked pixel dimensions for display (both lock together when object locks)
locked_marker_size_px = None
locked_object_size_px = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    frame_count += 1
    display_frame = frame.copy()
    
    # Preprocessing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if use_contrast:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    
    # Detect ArUco markers
    corners, ids, _ = detector.detectMarkers(gray)
    
    # Draw header bar
    cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 70),
                  (15, 15, 25), -1)
    cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 70),
                  (14, 165, 233), 2)
    cv2.putText(display_frame, "ZENITH INSPECT-AR - Industrial Measurement", 
                (20, 45), cv2.FONT_HERSHEY_DUPLEX, 0.9, (14, 165, 233), 2)
    
    if ids is not None and len(ids) > 0:
        # Calibrate using first marker
        if not calibrated:
            pixel_to_mm_ratio, avg_size = compute_pixel_to_mm_ratio(
                corners[0], KNOWN_MARKER_SIZE_MM
            )
            calibrated = True
        
        # Calculate marker pixel size (continuously updates, not locked yet)
        corner_points = corners[0][0]
        marker_width_px = np.linalg.norm(corner_points[0] - corner_points[1])
        marker_height_px = np.linalg.norm(corner_points[1] - corner_points[2])
        current_marker_size_px = (marker_width_px, marker_height_px)
        
        # Draw detected markers
        cv2.aruco.drawDetectedMarkers(display_frame, corners, ids)
        
        # Define ROI near marker
        roi_rect, roi_center = define_roi_near_marker(corners[0], frame.shape)
        x, y, w, h = roi_rect
        
        # Draw ROI boundary
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
        cv2.putText(display_frame, "ROI", (x + 5, y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
        
        # Detection locking logic
        if not is_locked:
            # Normal detection mode - use YOLO to detect object in ROI
            detection, roi_frame, contour = detect_object_with_yolo_in_roi(
                frame, roi_rect, yolo_detector, marker_corners=corners[0]
            )
            
            if detection is not None and contour is not None:
                detection_count += 1
                
                # Start or continue timer for consistent detection
                current_time = time.time()
                if first_detection_time is None:
                    first_detection_time = current_time
                    consistent_detection_timer = 0.0
                    locked_detection_id = detection['id']
                else:
                    consistent_detection_timer = current_time - first_detection_time
                
                # Measure object with rotation handling
                measurements, rect = measure_object_with_rotation(contour, pixel_to_mm_ratio)
                
                if measurements is not None:
                    # Apply advanced smoothing to reduce fluctuation
                    smoothed_measurements = apply_advanced_smoothing(detection['id'], measurements)
                    
                    # Get object actual pixel size from rotated rect (not YOLO bbox)
                    # This is the actual measured dimension used in calculations
                    obj_width_px = smoothed_measurements['width_mm'] / pixel_to_mm_ratio
                    obj_height_px = smoothed_measurements['height_mm'] / pixel_to_mm_ratio
                    current_object_size_px = (obj_width_px, obj_height_px)
                    
                    # Check if we should lock onto this object (5 seconds)
                    if consistent_detection_timer >= LOCK_THRESHOLD_SECONDS:
                        is_locked = True
                        locked_measurements = smoothed_measurements.copy()
                        locked_rect = rect
                        locked_box = cv2.boxPoints(rect)
                        locked_box = np.intp(locked_box)
                        locked_box[:, 0] += x
                        locked_box[:, 1] += y
                        # Lock BOTH marker and object pixel dimensions together
                        locked_marker_size_px = current_marker_size_px
                        locked_object_size_px = current_object_size_px
                        print(f"🔒 LOCKED onto object (ID: {detection['id']}, Confidence: {detection['confidence']:.2f}) after {consistent_detection_timer:.1f}s")
                        print(f"📏 Locked dimensions - Marker: {locked_marker_size_px[0]:.0f}x{locked_marker_size_px[1]:.0f}px, Object: {locked_object_size_px[0]:.0f}x{locked_object_size_px[1]:.0f}px")
                    
                    # Draw rotated rectangle
                    box = draw_rotated_rectangle(display_frame, rect, (x, y), 
                                                color=(0, 255, 0), thickness=3)
                    
                    # Draw dimension lines with smoothed values
                    draw_dimension_lines(display_frame, box, smoothed_measurements)
                    
                    # Draw center point
                    center_abs = (int(smoothed_measurements['center'][0]) + x, 
                                 int(smoothed_measurements['center'][1]) + y)
                    cv2.circle(display_frame, center_abs, 6, (0, 255, 0), -1)
                    cv2.circle(display_frame, center_abs, 10, (0, 255, 0), 2)
                    
                    # Draw compact info panel in top right
                    # If locked: show locked values, otherwise show current marker + no object yet
                    display_marker_size = locked_marker_size_px if is_locked else current_marker_size_px
                    display_object_size = locked_object_size_px if is_locked else None
                    draw_compact_info_panel(display_frame, display_marker_size, display_object_size)
                    
                    # Draw detection bbox from YOLO
                    bbox = detection['bbox']
                    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = [int(coord) for coord in bbox]
                    cv2.rectangle(roi_frame, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), (0, 255, 255), 2)
                    
                    # Draw confidence and detection info
                    conf_text = f"Conf: {detection['confidence']:.2f}"
                    cv2.putText(roi_frame, conf_text, (bbox_x1, bbox_y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    
                    # Draw depth indicator (size relative to marker)
                    if 'area' in detection:
                        area_text = f"Area: {detection['area']:.0f}px"
                        cv2.putText(roi_frame, area_text, (bbox_x1, bbox_y2 + 15),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                    
                    # Status with timer
                    if consistent_detection_timer < LOCK_THRESHOLD_SECONDS:
                        status_text = f"✅ DETECTING... Lock in {LOCK_THRESHOLD_SECONDS - consistent_detection_timer:.1f}s"
                        status_color = (0, 255, 255)
                    else:
                        status_text = "🔒 LOCKED ON TARGET"
                        status_color = (0, 255, 0)
                else:
                    status_text = "⚠️ CONTOUR TOO SMALL"
                    status_color = (0, 165, 255)
                    first_detection_time = None
                    consistent_detection_timer = 0.0
            else:
                status_text = "🔍 SEARCHING FOR OBJECT IN ROI (YOLO)..."
                status_color = (255, 200, 0)
                first_detection_time = None
                consistent_detection_timer = 0.0
                
                # Still show ROI for context
                roi_frame = frame[y:y+h, x:x+w].copy()
                
                # Draw info panel: show current marker size (updating) + no object
                draw_compact_info_panel(display_frame, current_marker_size_px, None)
        else:
            # Locked mode - display frozen measurements
            # Draw locked rotated rectangle
            cv2.drawContours(display_frame, [locked_box], 0, (255, 0, 255), 3)
            
            # Draw dimension lines with locked values
            draw_dimension_lines(display_frame, locked_box, locked_measurements)
            
            # Draw center point with locked measurements
            center_abs = (int(locked_measurements['center'][0]) + x, 
                         int(locked_measurements['center'][1]) + y)
            cv2.circle(display_frame, center_abs, 6, (255, 0, 255), -1)
            cv2.circle(display_frame, center_abs, 10, (255, 0, 255), 2)
            
            # Draw compact info panel in top right with permanently locked values
            draw_compact_info_panel(display_frame, locked_marker_size_px, locked_object_size_px)
            
            # Status
            status_text = "🔒 LOCKED - Press 'u' to UNLOCK"
            status_color = (255, 0, 255)
            
            # Still show ROI for context (but no detection)
            roi_frame = frame[y:y+h, x:x+w].copy()
        
        # Show ROI and binary mask in corner (for debugging)
        if roi_frame is not None:
            roi_small = cv2.resize(roi_frame, (200, 150))
            display_frame[90:240, 10:210] = roi_small
            cv2.rectangle(display_frame, (10, 90), (210, 240), (255, 200, 0), 2)
            cv2.putText(display_frame, "ROI View", (15, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
    else:
        status_text = "❌ NO MARKER - SHOW ARUCO TO CALIBRATE"
        status_color = (0, 0, 255)
        calibrated = False
    
    # Draw status
    cv2.putText(display_frame, status_text, (20, display_frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    # Draw calibration info
    if calibrated:
        calib_text = f"Cal: {pixel_to_mm_ratio:.4f} mm/px | Ref: {KNOWN_MARKER_SIZE_MM}mm"
        cv2.putText(display_frame, calib_text, 
                    (display_frame.shape[1] - 580, display_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Show frame
    cv2.imshow('Industrial Object Measurement System', display_frame)
    
    # Handle keypresses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"industrial_measurement_{timestamp}.jpg"
        cv2.imwrite(filename, display_frame)
        print(f"Saved: {filename}")
    elif key == ord('c'):
        use_contrast = not use_contrast
    elif key == ord('r'):
        calibrated = False
        pixel_to_mm_ratio = None
        is_locked = False
        locked_measurements = None
        first_detection_time = None
        consistent_detection_timer = 0.0
        measurement_history.clear()
        locked_marker_size_px = None
        locked_object_size_px = None
    elif key == ord('u'):
        # Unlock detection
        if is_locked:
            is_locked = False
            locked_measurements = None
            locked_rect = None
            locked_box = None
            first_detection_time = None
            consistent_detection_timer = 0.0
            locked_detection_id = None
            measurement_history.clear()
            locked_object_size_px = None  # Clear object size but keep marker size
            print("🔓 Unlocked - resuming YOLO detection")
    elif key == ord('f'):
        cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()

print(f"Session complete: {frame_count} frames, {detection_count} objects detected")