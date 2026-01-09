"""
ZENITH INSPECT-AR - Industrial Object Measurement System (Production Version)
Real-time dimension measurement using ArUco marker calibration
"""

import cv2
import numpy as np
from datetime import datetime
import math

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
    
    width_mm = width_px * pixel_to_mm_ratio
    height_mm = height_px * pixel_to_mm_ratio
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

print("Controls: 'q' - Quit, 's' - Save screenshot, 'c' - Contrast, 'r' - Reset, 'f' - Freeze")

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
        
        # Draw detected markers
        cv2.aruco.drawDetectedMarkers(display_frame, corners, ids)
        
        # Define ROI near marker
        roi_rect, roi_center = define_roi_near_marker(corners[0], frame.shape)
        x, y, w, h = roi_rect
        
        # Draw ROI boundary
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
        cv2.putText(display_frame, "ROI", (x + 5, y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
        
        # Detect object in ROI
        contour, roi_frame, binary_mask = detect_object_in_roi(frame, roi_rect)
        
        if contour is not None:
            detection_count += 1
            
            # Measure object with rotation handling
            measurements, rect = measure_object_with_rotation(contour, pixel_to_mm_ratio)
            
            if measurements is not None:
                # Apply advanced smoothing to reduce fluctuation
                smoothed_measurements = apply_advanced_smoothing(0, measurements)  # Use 0 as object_id for single object
                
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
                
                # Draw measurement panel with smoothed values
                panel_x = display_frame.shape[1] - 420
                panel_y = 90
                draw_measurement_overlay(display_frame, smoothed_measurements, 
                                       (panel_x, panel_y), "TARGET OBJECT")
                
                # Draw contour in ROI view
                cv2.drawContours(roi_frame, [contour], -1, (0, 255, 0), 2)
                
                # Status
                status_text = "✅ OBJECT DETECTED & MEASURED"
                status_color = (0, 255, 0)
            else:
                status_text = "⚠️ CONTOUR TOO SMALL"
                status_color = (0, 165, 255)
        else:
            status_text = "🔍 SEARCHING FOR OBJECT IN ROI..."
            status_color = (255, 200, 0)
        
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
    elif key == ord('f'):
        cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()

print(f"Session complete: {frame_count} frames, {detection_count} objects detected")