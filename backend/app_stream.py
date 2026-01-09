"""
INSPECT-AR - Minimal Web UI with Live Camera Stream
Real-time object measurement with ArUco marker calibration
"""

from flask import Flask, render_template, Response, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import time
from pathlib import Path
import sys

# Add models directory to path
sys.path.append(str(Path(__file__).parent))
from models.yolo_detector import YOLODetector

app = Flask(__name__)
CORS(app)

# ArUco dictionaries
ARUCO_DICTS = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
}

# Configuration
KNOWN_MARKER_SIZE_MM = 50.0
dict_name = "DICT_6X6_250"
aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[dict_name])

# Initialize detector
aruco_params = cv2.aruco.DetectorParameters()
aruco_params.adaptiveThreshWinSizeMin = 3
aruco_params.adaptiveThreshWinSizeMax = 23
aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
aruco_params.cornerRefinementWinSize = 5
aruco_params.minMarkerPerimeterRate = 0.03
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# YOLO detector
yolo_detector = YOLODetector()

# Global state
camera = None
camera_active = False
pixel_to_mm_ratio = None
calibrated = False
is_locked = False
locked_measurements = None
locked_marker_size_px = None
locked_object_size_px = None
locked_box = None
first_detection_time = None
LOCK_THRESHOLD_SECONDS = 5.0

# Measurement history
measurement_history = {}
smoothing_window = 15
min_stable_frames = 8
stability_threshold = 0.015


def compute_pixel_to_mm_ratio(marker_corners, known_size_mm):
    corner_points = marker_corners[0]
    side1 = np.linalg.norm(corner_points[0] - corner_points[1])
    side2 = np.linalg.norm(corner_points[1] - corner_points[2])
    side3 = np.linalg.norm(corner_points[2] - corner_points[3])
    side4 = np.linalg.norm(corner_points[3] - corner_points[0])
    all_sides = np.array([side1, side2, side3, side4])
    avg_marker_size_px = np.median(all_sides)
    pixel_to_mm_ratio = known_size_mm / avg_marker_size_px
    return pixel_to_mm_ratio, avg_marker_size_px


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
    
    return (roi_x, roi_y, roi_width, roi_height)


def detect_object_with_yolo_in_roi(frame, roi_rect, yolo_detector, marker_corners=None):
    x, y, w, h = roi_rect
    roi_frame = frame[y:y+h, x:x+w].copy()
    result = yolo_detector.detect(roi_frame)
    
    if result['count'] == 0:
        return None, roi_frame, None
    
    # Calculate marker reference
    marker_reference_size = None
    marker_center = None
    if marker_corners is not None:
        corner_points = marker_corners[0]
        side1 = np.linalg.norm(corner_points[0] - corner_points[1])
        side2 = np.linalg.norm(corner_points[1] - corner_points[2])
        side3 = np.linalg.norm(corner_points[2] - corner_points[3])
        side4 = np.linalg.norm(corner_points[3] - corner_points[0])
        marker_reference_size = np.mean([side1, side2, side3, side4])
        marker_center = np.mean(corner_points, axis=0)
    
    best_detection = None
    best_score = 0
    roi_center_x = w / 2
    roi_center_y = h / 2
    
    for detection in result['detections']:
        if detection['confidence'] < 0.3:
            continue
        
        bbox = detection['bbox']
        det_width = bbox[2] - bbox[0]
        det_height = bbox[3] - bbox[1]
        det_size = (det_width + det_height) / 2
        det_center_x = (bbox[0] + bbox[2]) / 2
        det_center_y = (bbox[1] + bbox[3]) / 2
        det_center_abs_x = det_center_x + x
        det_center_abs_y = det_center_y + y
        
        confidence_score = detection['confidence']
        distance_to_roi_center = np.sqrt((det_center_x - roi_center_x)**2 + (det_center_y - roi_center_y)**2)
        proximity_score = 1.0 - min(distance_to_roi_center / (w/2), 1.0)
        
        depth_score = 1.0
        if marker_reference_size is not None:
            size_ratio = det_size / marker_reference_size
            if 0.5 <= size_ratio <= 3.0:
                depth_score = 1.0
            elif size_ratio < 0.5:
                depth_score = 0.3
            else:
                depth_score = 0.4
        
        marker_distance_score = 1.0
        if marker_center is not None:
            distance_to_marker = np.sqrt((det_center_abs_x - marker_center[0])**2 + 
                                        (det_center_abs_y - marker_center[1])**2)
            frame_diagonal = np.sqrt(w**2 + h**2)
            normalized_distance = min(distance_to_marker / frame_diagonal, 1.0)
            marker_distance_score = 1.0 - normalized_distance * 0.5
        
        score = (depth_score * 0.5 + confidence_score * 0.3 + 
                proximity_score * 0.15 + marker_distance_score * 0.05)
        
        if score > best_score:
            best_score = score
            best_detection = detection
    
    if best_detection is None:
        return None, roi_frame, None
    
    bbox = best_detection['bbox']
    x1, y1, x2, y2 = [int(coord) for coord in bbox]
    contour = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.int32)
    
    return best_detection, roi_frame, contour


def measure_object_with_rotation(contour, pixel_to_mm_ratio):
    if contour is None:
        return None, None
    
    rect = cv2.minAreaRect(contour)
    center, size, angle = rect
    width_px, height_px = size
    
    if width_px < height_px:
        width_px, height_px = height_px, width_px
        angle = angle + 90
    
    width_mm = (width_px * pixel_to_mm_ratio) - 0.1
    height_mm = (height_px * pixel_to_mm_ratio) - 0.1
    
    measurements = {
        'width_mm': width_mm,
        'height_mm': height_mm,
        'width_px': width_px,
        'height_px': height_px,
        'angle_deg': angle,
        'center': center
    }
    
    return measurements, rect


def draw_compact_info_panel(frame, marker_size_px, object_size_px=None):
    panel_width = 300
    panel_height = 140
    margin = 15
    x = frame.shape[1] - panel_width - margin
    y = 80
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + panel_width, y + panel_height), (20, 25, 30), -1)
    cv2.rectangle(overlay, (x, y), (x + panel_width, y + panel_height), (14, 165, 233), 2)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    
    cv2.putText(frame, "DIMENSIONS (pixels)", (x + 15, y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (14, 165, 233), 2)
    cv2.line(frame, (x + 15, y + 32), (x + panel_width - 15, y + 32), (14, 165, 233), 1)
    
    y_offset = y + 55
    cv2.putText(frame, "Marker L x W:", (x + 15, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    if marker_size_px is not None:
        marker_text = f"{marker_size_px[0]:.0f} x {marker_size_px[1]:.0f} px"
        cv2.putText(frame, marker_text, (x + 150, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    y_offset += 28
    cv2.putText(frame, "Object L x W:", (x + 15, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    if object_size_px is not None:
        object_text = f"{object_size_px[0]:.0f} x {object_size_px[1]:.0f} px"
        cv2.putText(frame, object_text, (x + 150, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "--- x --- px", (x + 150, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 2)


def generate_frames():
    global camera, camera_active, pixel_to_mm_ratio, calibrated, is_locked
    global locked_measurements, locked_marker_size_px, locked_object_size_px
    global locked_box, first_detection_time
    
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    camera_active = True
    
    while camera_active:
        success, frame = camera.read()
        if not success:
            break
        
        display_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Draw header
        cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 70), (15, 15, 25), -1)
        cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 70), (14, 165, 233), 2)
        cv2.putText(display_frame, "INSPECT-AR", 
                    (display_frame.shape[1]//2 - 100, 45), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (14, 165, 233), 2)
        
        # Detect markers
        corners, ids, _ = detector.detectMarkers(gray)
        
        if ids is not None and len(ids) > 0:
            if not calibrated:
                pixel_to_mm_ratio, _ = compute_pixel_to_mm_ratio(corners[0], KNOWN_MARKER_SIZE_MM)
                calibrated = True
            
            corner_points = corners[0][0]
            marker_width_px = np.linalg.norm(corner_points[0] - corner_points[1])
            marker_height_px = np.linalg.norm(corner_points[1] - corner_points[2])
            current_marker_size_px = (marker_width_px, marker_height_px)
            
            cv2.aruco.drawDetectedMarkers(display_frame, corners, ids)
            roi_rect = define_roi_near_marker(corners[0], frame.shape)
            x, y, w, h = roi_rect
            
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
            
            if not is_locked:
                detection, roi_frame, contour = detect_object_with_yolo_in_roi(
                    frame, roi_rect, yolo_detector, marker_corners=corners[0]
                )
                
                if detection is not None and contour is not None:
                    current_time = time.time()
                    if first_detection_time is None:
                        first_detection_time = current_time
                    
                    consistent_detection_timer = current_time - first_detection_time
                    measurements, rect = measure_object_with_rotation(contour, pixel_to_mm_ratio)
                    
                    if measurements is not None:
                        obj_width_px = measurements['width_px']
                        obj_height_px = measurements['height_px']
                        current_object_size_px = (obj_width_px, obj_height_px)
                        
                        if consistent_detection_timer >= LOCK_THRESHOLD_SECONDS:
                            is_locked = True
                            locked_measurements = measurements.copy()
                            locked_box = cv2.boxPoints(rect)
                            locked_box = np.intp(locked_box)
                            locked_box[:, 0] += x
                            locked_box[:, 1] += y
                            locked_marker_size_px = current_marker_size_px
                            locked_object_size_px = current_object_size_px
                        
                        box = cv2.boxPoints(rect)
                        box = np.intp(box)
                        box[:, 0] += x
                        box[:, 1] += y
                        cv2.drawContours(display_frame, [box], 0, (0, 255, 0), 3)
                        
                        display_marker_size = current_marker_size_px
                        display_object_size = None
                        draw_compact_info_panel(display_frame, display_marker_size, display_object_size)
                        
                        status_text = f"DETECTING... Lock in {max(0, LOCK_THRESHOLD_SECONDS - consistent_detection_timer):.1f}s"
                        status_color = (0, 255, 255)
                    else:
                        first_detection_time = None
                        draw_compact_info_panel(display_frame, current_marker_size_px, None)
                        status_text = "CONTOUR TOO SMALL"
                        status_color = (0, 165, 255)
                else:
                    first_detection_time = None
                    draw_compact_info_panel(display_frame, current_marker_size_px, None)
                    status_text = "SEARCHING FOR OBJECT..."
                    status_color = (255, 200, 0)
            else:
                cv2.drawContours(display_frame, [locked_box], 0, (255, 0, 255), 3)
                draw_compact_info_panel(display_frame, locked_marker_size_px, locked_object_size_px)
                status_text = "LOCKED - Press 'r' to reset"
                status_color = (255, 0, 255)
        else:
            status_text = "NO MARKER DETECTED"
            status_color = (0, 0, 255)
            calibrated = False
        
        cv2.putText(display_frame, status_text, (20, display_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        ret, buffer = cv2.imencode('.jpg', display_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    if camera is not None:
        camera.release()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_camera', methods=['POST'])
def start_camera():
    global camera_active
    camera_active = True
    return jsonify({'status': 'started'})


@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera_active, camera
    camera_active = False
    if camera is not None:
        camera.release()
    return jsonify({'status': 'stopped'})


@app.route('/reset', methods=['POST'])
def reset():
    global is_locked, locked_measurements, locked_marker_size_px, locked_object_size_px
    global locked_box, first_detection_time, calibrated, pixel_to_mm_ratio
    is_locked = False
    locked_measurements = None
    locked_marker_size_px = None
    locked_object_size_px = None
    locked_box = None
    first_detection_time = None
    calibrated = False
    pixel_to_mm_ratio = None
    return jsonify({'status': 'reset'})


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 INSPECT-AR Web Interface")
    print("="*60)
    print("Access at: http://localhost:5000")
    print("="*60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
