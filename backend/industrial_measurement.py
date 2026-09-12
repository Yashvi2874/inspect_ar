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
from utils.video_source import open_capture

# ArUco dictionaries
ARUCO_DICTS = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
}

# Available marker sizes
MARKER_SIZES = [47, 50, 40, 30]  # mm

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
# Defaults, so that importing this module needs no stdin and opens no camera.
# configure() overwrites them from the interactive prompts when run as a script.
KNOWN_MARKER_SIZE_MM = MARKER_SIZES[0]
dict_name = list(ARUCO_DICTS.keys())[2]
aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[dict_name])

# Smoothing state. These live at module scope because apply_advanced_smoothing()
# reads them as globals; making them locals of main() would break it.
measurement_history = {}
smoothing_window = 20          # frames averaged for stability
min_stable_frames = 8          # frames before a measurement counts as stable
stability_threshold = 0.010    # 1% relative threshold
stability_std_threshold = 0.5  # standard-deviation threshold


def configure():
    """Ask for marker size and ArUco dictionary, and rebuild the detector.

    Called only from main(). Sets the module-level configuration so the
    functions below, which read these as globals, see the chosen values.
    """
    global KNOWN_MARKER_SIZE_MM, dict_name, aruco_dict, detector, marker_length

    print("Available ArUco marker sizes:")
    for i, size in enumerate(MARKER_SIZES, 1):
        print(f"  {i}. {size}x{size}mm")

    print("\nEnter marker size (1-4, default: 1 for 47x47mm): ", end="")
    size_input = input().strip()
    size_choice = (int(size_input)
                   if size_input.isdigit() and 1 <= int(size_input) <= len(MARKER_SIZES)
                   else 1)
    KNOWN_MARKER_SIZE_MM = MARKER_SIZES[size_choice - 1]

    print("\nAvailable ArUco dictionaries:")
    for i, (name, _) in enumerate(ARUCO_DICTS.items(), 1):
        print(f"  {i}. {name}")

    print("Select ArUco Dictionary (1-3, default: 3): ", end="")
    dict_input = input().strip()
    dict_choice = (int(dict_input)
                   if dict_input.isdigit() and 1 <= int(dict_input) <= 3
                   else 3)
    dict_name = list(ARUCO_DICTS.keys())[dict_choice - 1]
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[dict_name])

    # Rebuild the two objects derived from the choices above.
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    marker_length = KNOWN_MARKER_SIZE_MM / 1000.0


# Initialize detector
aruco_params = cv2.aruco.DetectorParameters()
aruco_params.adaptiveThreshWinSizeMin = 3
aruco_params.adaptiveThreshWinSizeMax = 23
aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
aruco_params.cornerRefinementWinSize = 5
aruco_params.minMarkerPerimeterRate = 0.03

# Camera intrinsic parameters (you may need to calibrate these for your specific camera)
# These are placeholder values - you should replace them with actual calibrated values
CAMERA_MATRIX = np.array([
    [800, 0, 640],  # fx, 0, cx
    [0, 800, 360],  # 0, fy, cy
    [0, 0, 1]       # 0, 0, 1
], dtype=np.float32)
DIST_COEFFS = np.zeros((4, 1))  # Assuming no lens distortion for simplicity


def calibrate_camera_intrinsics():
    """
    Function to guide user through camera calibration process
    This is a simplified version - in practice, you'd use multiple calibration images
    """
    print("\n=== CAMERA CALIBRATION GUIDE ===")
    print("To improve measurement accuracy:")
    print("1. Print a chessboard pattern (8x6 squares, known square size)")
    print("2. Capture 15-25 images from different angles")
    print("3. Use OpenCV's calibration functions")
    print("4. Replace CAMERA_MATRIX and DIST_COEFFS with calibrated values")
    print("\nFor best results, use a checkerboard with known square size in mm\n")


def estimate_marker_pose(corners, ids, marker_length):
    """
    Estimate pose of ArUco markers to get 3D information
    """
    if ids is None or len(ids) == 0:
        return None, None, None, None
    
    # Estimate pose of each marker
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners, marker_length, CAMERA_MATRIX, DIST_COEFFS
    )
    
    # Calculate distance to each marker and extract tilt information
    distances = []
    tilt_angles = []
    
    for i in range(len(tvecs)):
        # Extract translation vector (x, y, z)
        tvec = tvecs[i][0]
        # Calculate distance from camera to marker in meters, then convert to mm
        distance = np.sqrt(tvec[0]**2 + tvec[1]**2 + tvec[2]**2) * 1000  # mm
        distances.append(distance)
        
        # Convert rotation vector to rotation matrix
        rmat, _ = cv2.Rodrigues(rvecs[i][0])
        
        # Calculate tilt angles from rotation matrix
        # Extract Euler angles (approximate method)
        pitch = np.arcsin(-rmat[2, 0])  # Pitch (rotation around x-axis)
        yaw = np.arctan2(rmat[2, 1], rmat[2, 2])  # Yaw (rotation around y-axis)
        roll = np.arctan2(rmat[1, 0], rmat[0, 0])  # Roll (rotation around z-axis)
        
        # Store tilt angles (pitch and yaw are most important for perspective correction)
        tilt_angles.append({'pitch': pitch, 'yaw': yaw, 'roll': roll})
    
    return rvecs, tvecs, distances, tilt_angles


def draw_axis_on_marker(frame, rvecs, tvecs, camera_matrix, dist_coeffs, marker_length):
    """
    Draw coordinate axes on the ArUco marker to visualize its orientation
    """
    # This function intentionally left empty due to OpenCV version compatibility issues
    # The drawAxis function may not be available in all OpenCV versions
    pass

detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# Initialize variables for pose estimation
marker_length = KNOWN_MARKER_SIZE_MM / 1000.0  # Convert mm to meters for pose estimation

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

def detect_objects_with_yolo_in_roi(frame, roi_rect, yolo_detector, marker_corners=None, min_confidence=0.3):
    """
    Use YOLO to detect all objects within the ROI
    Returns: (detections_list, roi_frame)
    """
    x, y, w, h = roi_rect
    roi_frame = frame[y:y+h, x:x+w].copy()
    
    # Run YOLO detection on ROI
    result = yolo_detector.detect(roi_frame)
    
    if result['count'] == 0:
        return [], roi_frame
    
    # Filter detections by confidence and create contours
    valid_detections = []
    for detection in result['detections']:
        if detection['confidence'] >= min_confidence:
            # Create a contour from the detection bbox for measurement
            bbox = detection['bbox']
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            # Create a rectangular contour
            contour = np.array([
                [[x1, y1]],
                [[x2, y1]],
                [[x2, y2]],
                [[x1, y2]]
            ], dtype=np.int32)
            
            # Add contour to detection
            detection_with_contour = detection.copy()
            detection_with_contour['contour'] = contour
            valid_detections.append(detection_with_contour)
    
    return valid_detections, roi_frame

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
    
    width_mm = (width_px * pixel_to_mm_ratio) - 10 ####################
    height_mm = (height_px * pixel_to_mm_ratio) - 10###########################
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


def calculate_circle_properties(contour, pixel_to_mm_ratio):
    """
    Calculate properties for circular objects including diameter and perimeter
    """
    # Calculate minimum enclosing circle
    (x, y), radius_px = cv2.minEnclosingCircle(contour)
    
    # Calculate area-based radius for more accurate measurement
    area = cv2.contourArea(contour)
    radius_area_based_px = math.sqrt(area / math.pi)
    
    # Use the area-based radius for more accurate measurements
    radius_mm = radius_area_based_px * pixel_to_mm_ratio
    diameter_mm = 2 * radius_mm
    
    # Calculate perimeter (circumference) using both methods
    perimeter_contour_mm = cv2.arcLength(contour, closed=True) * pixel_to_mm_ratio
    circumference_mm = 2 * math.pi * radius_mm
    
    # Calculate circularity as a measure of how circular the object is
    perimeter = cv2.arcLength(contour, True)
    circularity = 0
    if perimeter > 0:
        circularity = (4 * math.pi * area) / (perimeter * perimeter)
    
    return {
        'shape_type': 'circle',
        'circle_radius_mm': radius_mm,
        'circle_diameter_mm': diameter_mm,
        'circle_circumference_mm': circumference_mm,
        'circle_area_mm2': area * (pixel_to_mm_ratio ** 2),
        'circle_perimeter_mm': perimeter_contour_mm,  # Perimeter from contour
        'circle_circularity': circularity,
        'shape_vertices': 0,
        'center': (x, y)
    }


def calculate_triangle_properties(approx, pixel_to_mm_ratio):
    """
    Calculate properties for triangular objects including side lengths, angles, and altitudes
    """
    # Get the 3 vertices of the triangle
    pts = approx.reshape(3, 2)
    
    # Calculate side lengths in pixels
    side_a_px = np.linalg.norm(pts[0] - pts[1])  # Side 0-1
    side_b_px = np.linalg.norm(pts[1] - pts[2])  # Side 1-2
    side_c_px = np.linalg.norm(pts[2] - pts[0])  # Side 2-0
    
    # Convert to mm
    side_a_mm = side_a_px * pixel_to_mm_ratio
    side_b_mm = side_b_px * pixel_to_mm_ratio
    side_c_mm = side_c_px * pixel_to_mm_ratio
    
    # Calculate interior angles using the law of cosines
    # Angle at vertex 0 (opposite to side_b)
    cos_angle_0 = (side_a_mm**2 + side_c_mm**2 - side_b_mm**2) / (2 * side_a_mm * side_c_mm)
    angle_0_rad = np.arccos(np.clip(cos_angle_0, -1.0, 1.0))
    angle_0_deg = np.degrees(angle_0_rad)
    
    # Angle at vertex 1 (opposite to side_c)
    cos_angle_1 = (side_a_mm**2 + side_b_mm**2 - side_c_mm**2) / (2 * side_a_mm * side_b_mm)
    angle_1_rad = np.arccos(np.clip(cos_angle_1, -1.0, 1.0))
    angle_1_deg = np.degrees(angle_1_rad)
    
    # Angle at vertex 2 (opposite to side_a)
    angle_2_deg = 180.0 - angle_0_deg - angle_1_deg
    
    # Calculate altitudes (slant heights) using area
    # Area using Heron's formula
    semi_perimeter = (side_a_mm + side_b_mm + side_c_mm) / 2
    area = math.sqrt(semi_perimeter * (semi_perimeter - side_a_mm) * 
                     (semi_perimeter - side_b_mm) * (semi_perimeter - side_c_mm))
    
    # Altitudes (heights) from each side
    altitude_a_mm = (2 * area) / side_a_mm if side_a_mm > 0 else 0
    altitude_b_mm = (2 * area) / side_b_mm if side_b_mm > 0 else 0
    altitude_c_mm = (2 * area) / side_c_mm if side_c_mm > 0 else 0
    
    return {
        'shape_type': 'triangle',
        'triangle_side_a_mm': side_a_mm,
        'triangle_side_b_mm': side_b_mm,
        'triangle_side_c_mm': side_c_mm,
        'triangle_angle_0_deg': angle_0_deg,
        'triangle_angle_1_deg': angle_1_deg,
        'triangle_angle_2_deg': angle_2_deg,
        'triangle_altitude_a_mm': altitude_a_mm,
        'triangle_altitude_b_mm': altitude_b_mm,
        'triangle_altitude_c_mm': altitude_c_mm,
        'triangle_area_mm2': area,
        'shape_vertices': 3,
        'circularity': 0  # Not a circle
    }


def calculate_quadrilateral_properties(approx, pixel_to_mm_ratio):
    """
    Calculate properties for quadrilateral objects including side lengths, angles, and diagonals
    """
    # Get the 4 vertices of the quadrilateral
    pts = approx.reshape(4, 2)
    
    # Calculate side lengths in pixels
    side_0_px = np.linalg.norm(pts[0] - pts[1])  # Side 0-1
    side_1_px = np.linalg.norm(pts[1] - pts[2])  # Side 1-2
    side_2_px = np.linalg.norm(pts[2] - pts[3])  # Side 2-3
    side_3_px = np.linalg.norm(pts[3] - pts[0])  # Side 3-0
    
    # Convert to mm
    side_0_mm = side_0_px * pixel_to_mm_ratio
    side_1_mm = side_1_px * pixel_to_mm_ratio
    side_2_mm = side_2_px * pixel_to_mm_ratio
    side_3_mm = side_3_px * pixel_to_mm_ratio
    
    # Calculate interior angles using the law of cosines
    # For each vertex, we calculate the angle between the two adjacent sides
    
    # Angle at vertex 0 (between sides 3 and 0)
    v0_3 = pts[0] - pts[3]  # Vector from vertex 3 to vertex 0
    v0_1 = pts[1] - pts[0]  # Vector from vertex 0 to vertex 1
    cos_angle_0 = np.dot(v0_3, v0_1) / (np.linalg.norm(v0_3) * np.linalg.norm(v0_1))
    angle_0_rad = np.arccos(np.clip(cos_angle_0, -1.0, 1.0))
    angle_0_deg = np.degrees(angle_0_rad)
    
    # Angle at vertex 1 (between sides 0 and 1)
    v1_0 = pts[0] - pts[1]  # Vector from vertex 0 to vertex 1
    v1_2 = pts[2] - pts[1]  # Vector from vertex 1 to vertex 2
    cos_angle_1 = np.dot(v1_0, v1_2) / (np.linalg.norm(v1_0) * np.linalg.norm(v1_2))
    angle_1_rad = np.arccos(np.clip(cos_angle_1, -1.0, 1.0))
    angle_1_deg = np.degrees(angle_1_rad)
    
    # Angle at vertex 2 (between sides 1 and 2)
    v2_1 = pts[1] - pts[2]  # Vector from vertex 1 to vertex 2
    v2_3 = pts[3] - pts[2]  # Vector from vertex 2 to vertex 3
    cos_angle_2 = np.dot(v2_1, v2_3) / (np.linalg.norm(v2_1) * np.linalg.norm(v2_3))
    angle_2_rad = np.arccos(np.clip(cos_angle_2, -1.0, 1.0))
    angle_2_deg = np.degrees(angle_2_rad)
    
    # Angle at vertex 3 (between sides 2 and 3)
    v3_2 = pts[2] - pts[3]  # Vector from vertex 2 to vertex 3
    v3_0 = pts[0] - pts[3]  # Vector from vertex 0 to vertex 3
    cos_angle_3 = np.dot(v3_2, v3_0) / (np.linalg.norm(v3_2) * np.linalg.norm(v3_0))
    angle_3_rad = np.arccos(np.clip(cos_angle_3, -1.0, 1.0))
    angle_3_deg = np.degrees(angle_3_rad)
    
    # Calculate diagonals
    diagonal_02_px = np.linalg.norm(pts[0] - pts[2])  # Diagonal 0-2
    diagonal_13_px = np.linalg.norm(pts[1] - pts[3])  # Diagonal 1-3
    
    diagonal_02_mm = diagonal_02_px * pixel_to_mm_ratio
    diagonal_13_mm = diagonal_13_px * pixel_to_mm_ratio
    
    # Calculate area using the shoelace formula
    area = 0.5 * abs(
        pts[0][0] * (pts[1][1] - pts[3][1]) +
        pts[1][0] * (pts[2][1] - pts[0][1]) +
        pts[2][0] * (pts[3][1] - pts[1][1]) +
        pts[3][0] * (pts[0][1] - pts[2][1])
    )
    area_mm2 = area * (pixel_to_mm_ratio ** 2)
    
    return {
        'shape_type': 'quadrilateral',
        'quad_side_0_mm': side_0_mm,
        'quad_side_1_mm': side_1_mm,
        'quad_side_2_mm': side_2_mm,
        'quad_side_3_mm': side_3_mm,
        'quad_angle_0_deg': angle_0_deg,
        'quad_angle_1_deg': angle_1_deg,
        'quad_angle_2_deg': angle_2_deg,
        'quad_angle_3_deg': angle_3_deg,
        'quad_diagonal_02_mm': diagonal_02_mm,
        'quad_diagonal_13_mm': diagonal_13_mm,
        'quad_area_mm2': area_mm2,
        'shape_vertices': 4,
        'circularity': 0  # Not a circle
    }


def calculate_geometric_properties(contour, pixel_to_mm_ratio):
    """
    Calculate geometric properties including slant heights and angles for shapes
    """
    # Check if the shape is circular using contour circularity
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    # Calculate circularity (4*pi*area/perimeter^2), circle has value close to 1
    if perimeter > 0:
        circularity = (4 * math.pi * area) / (perimeter * perimeter)
    else:
        circularity = 0
    
    # Check if the shape is circular based on circularity and aspect ratio
    if circularity > 0.7:  # Threshold for circularity
        props = calculate_circle_properties(contour, pixel_to_mm_ratio)
    else:
        # Approximate contour to polygon to get vertices for polygonal shapes
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        vertices = len(approx)
        
        props = {
            'shape_vertices': vertices,
            'shape_type': 'unknown',
            'circularity': circularity
        }
        
        # Classify shape based on number of vertices
        if vertices == 3:
            props['shape_type'] = 'triangle'
            # Calculate triangle properties
            triangle_props = calculate_triangle_properties(approx, pixel_to_mm_ratio)
            props.update(triangle_props)
        elif vertices == 4:
            props['shape_type'] = 'quadrilateral'
            # Calculate quadrilateral properties
            quad_props = calculate_quadrilateral_properties(approx, pixel_to_mm_ratio)
            props.update(quad_props)
        else:
            props['shape_type'] = f'{vertices}-gon'
    
    return props


def measure_object_with_tilt_correction(contour, pixel_to_mm_ratio, marker_tilt_angles):
    """
    Measure object dimensions with basic calculations (correction applied externally)
    """
    if contour is None:
        return None, None
    
    rect = cv2.minAreaRect(contour)
    center, size, angle = rect
    width_px, height_px = size
    
    # To ensure consistent width/height labeling regardless of orientation,
    # we'll use the bounding rectangle dimensions instead of rotated rectangle
    # This provides more intuitive width/height based on image coordinate system
    x, y, bbox_w, bbox_h = cv2.boundingRect(contour)
    
    # Compare the dimensions to decide which is width and which is height
    # Use the longer dimension as width and shorter as height for consistency
    if bbox_w >= bbox_h:
        width_mm = bbox_w * pixel_to_mm_ratio
        height_mm = bbox_h * pixel_to_mm_ratio
    else:
        width_mm = bbox_h * pixel_to_mm_ratio  # Use the longer dimension as width
        height_mm = bbox_w * pixel_to_mm_ratio   # Use the shorter dimension as height
        # Adjust angle by 90 degrees since we swapped dimensions
        angle = angle + 90 if abs(angle) <= 45 else angle - 90
    
    # For area and perimeter, use contour-based calculations
    area_px = cv2.contourArea(contour)
    area_mm2 = area_px * (pixel_to_mm_ratio ** 2)
    perimeter_px = cv2.arcLength(contour, closed=True)
    perimeter_mm = perimeter_px * pixel_to_mm_ratio
    
    bbox_width_mm = bbox_w * pixel_to_mm_ratio
    bbox_height_mm = bbox_h * pixel_to_mm_ratio
    
    aspect_ratio = width_mm / height_mm if height_mm > 0 else 0
    compactness = (4 * math.pi * area_px) / (perimeter_px ** 2) if perimeter_px > 0 else 0
    
    # Calculate geometric properties for circles, triangles, quadrilaterals, etc.
    geometric_props = calculate_geometric_properties(contour, pixel_to_mm_ratio)
    
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
    
    # Add geometric properties to measurements
    measurements.update(geometric_props)
    
    return measurements, rect

# Visualization functions
def draw_rotated_rectangle(frame, rect, roi_offset, color=(0, 255, 0), thickness=2):
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    box[:, 0] += roi_offset[0]
    box[:, 1] += roi_offset[1]
    cv2.drawContours(frame, [box], 0, color, thickness)
    return box

def draw_measurement_overlay(frame, measurements, position, object_label="OBJECT", distance_mm=None, is_multiple_objects=False, roi_rect=None):
    x, y = position
    # Reduce panel size when multiple objects are present to avoid crowding
    panel_width = 340 if is_multiple_objects else 380
    panel_height = 380 if is_multiple_objects else 480  # Reduced height for multiple objects
    
    # Ensure the overlay stays within frame boundaries and avoids ROI area
    frame_height, frame_width = frame.shape[:2]
    
    # Check if ROI information is provided to avoid overlap
    if roi_rect is not None:
        roi_x, roi_y, roi_width, roi_height = roi_rect
        
        # Adjust position to avoid ROI overlap
        if (x < roi_x + roi_width and x + panel_width > roi_x and 
            y < roi_y + roi_height and y + panel_height > roi_y):
            # If there's overlap, try positioning on the left side of the frame
            x = max(20, min(20, frame_width - panel_width - 20))
            y = max(80, min(y, frame_height - panel_height - 80))
        else:
            x = max(20, min(x, frame_width - panel_width - 20))
            y = max(80, min(y, frame_height - panel_height - 80))
    else:
        x = max(20, min(x, frame_width - panel_width - 20))
        y = max(80, min(y, frame_height - panel_height - 80))
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + panel_width, y + panel_height),
                  (20, 25, 30), -1)
    cv2.rectangle(overlay, (x, y), (x + panel_width, y + panel_height),
                  (14, 165, 233), 3)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
    
    cv2.putText(frame, f"📦 {object_label}", (x + 15, y + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7 if is_multiple_objects else 0.8, (14, 165, 233), 2)
    cv2.line(frame, (x + 15, y + 48), (x + panel_width - 15, y + 48),
             (14, 165, 233), 1)
    
    y_offset = y + 60 if is_multiple_objects else y + 75
    line_spacing = 24 if is_multiple_objects else 28  # Reduced spacing when multiple objects
    
    # Basic measurements (always shown)
    metrics = [
        ("Width (Rotated):", f"{measurements['width_mm']:.2f} mm", (0, 255, 255)),
        ("Height (Rotated):", f"{measurements['height_mm']:.2f} mm", (0, 255, 255)),
        ("", "", None),  # Spacer
        ("Area:", f"{measurements['area_mm2']:.2f} mm²", (0, 255, 0)),
        ("Perimeter:", f"{measurements['perimeter_mm']:.2f} mm", (255, 200, 0)),
        ("Rotation Angle:", f"{measurements['angle_deg']:.1f}°", (255, 100, 255)),
    ]
    
    # For multiple objects, only show essential geometric properties to save space
    if 'shape_type' in measurements and not is_multiple_objects:
        metrics.extend([
            ("", "", None),  # Spacer
            ("Shape Type:", f"{measurements['shape_type']}", (255, 255, 0)),
            ("Circularity:", f"{measurements.get('circularity', 0):.2f}", (255, 255, 0)),
        ])
        
        # Add circle-specific properties
        if measurements['shape_type'] == 'circle':
            metrics.extend([
                ("", "", None),  # Spacer
                ("Circle Properties:", "", (255, 100, 100)),
                ("Radius:", f"{measurements.get('circle_radius_mm', 0):.2f} mm", (255, 100, 100)),
                ("Diameter:", f"{measurements.get('circle_diameter_mm', 0):.2f} mm", (255, 100, 100)),
                ("Circumference:", f"{measurements.get('circle_circumference_mm', 0):.2f} mm", (255, 100, 100)),
            ])
        
        # Add triangle-specific properties
        elif measurements['shape_type'] == 'triangle':
            metrics.extend([
                ("", "", None),  # Spacer
                ("Triangle Sides:", "", (255, 150, 50)),
                ("Side A:", f"{measurements.get('triangle_side_a_mm', 0):.2f} mm", (255, 150, 50)),
                ("Side B:", f"{measurements.get('triangle_side_b_mm', 0):.2f} mm", (255, 150, 50)),
                ("Side C:", f"{measurements.get('triangle_side_c_mm', 0):.2f} mm", (255, 150, 50)),
                ("", "", None),  # Spacer
                ("Interior Angles:", "", (100, 200, 255)),
                ("Angle @V0:", f"{measurements.get('triangle_angle_0_deg', 0):.1f}°", (100, 200, 255)),
                ("Angle @V1:", f"{measurements.get('triangle_angle_1_deg', 0):.1f}°", (100, 200, 255)),
                ("Angle @V2:", f"{measurements.get('triangle_angle_2_deg', 0):.1f}°", (100, 200, 255)),
            ])
        
        # Add quadrilateral-specific properties
        elif measurements['shape_type'] == 'quadrilateral':
            metrics.extend([
                ("", "", None),  # Spacer
                ("Quad Sides:", "", (255, 150, 50)),
                ("Side 0-1:", f"{measurements.get('quad_side_0_mm', 0):.2f} mm", (255, 150, 50)),
                ("Side 1-2:", f"{measurements.get('quad_side_1_mm', 0):.2f} mm", (255, 150, 50)),
                ("Side 2-3:", f"{measurements.get('quad_side_2_mm', 0):.2f} mm", (255, 150, 50)),
                ("Side 3-0:", f"{measurements.get('quad_side_3_mm', 0):.2f} mm", (255, 150, 50)),
                ("", "", None),  # Spacer
                ("Interior Angles:", "", (100, 200, 255)),
                ("Angle 0:", f"{measurements.get('quad_angle_0_deg', 0):.1f}°", (100, 200, 255)),
                ("Angle 1:", f"{measurements.get('quad_angle_1_deg', 0):.1f}°", (100, 200, 255)),
                ("Angle 2:", f"{measurements.get('quad_angle_2_deg', 0):.1f}°", (100, 200, 255)),
                ("Angle 3:", f"{measurements.get('quad_angle_3_deg', 0):.1f}°", (100, 200, 255)),
            ])
    
    # When multiple objects are present, show only basic geometric properties
    elif 'shape_type' in measurements and is_multiple_objects:
        metrics.extend([
            ("", "", None),  # Spacer
            ("Shape Type:", f"{measurements['shape_type']}", (255, 255, 0)),
        ])
    
    # Add distance if available
    if distance_mm is not None:
        metrics.append(("Distance:", f"{distance_mm:.1f} mm", (100, 100, 255)))
    
    for i, (label, value, color) in enumerate(metrics):
        y_pos = y_offset + i * line_spacing
        if label:  # Skip spacers
            font_scale = 0.45 if is_multiple_objects else 0.5
            cv2.putText(frame, label, (x + 15, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (180, 180, 180), 1)
            font_scale_value = 0.5 if is_multiple_objects else 0.55
            cv2.putText(frame, value, (x + 150 if is_multiple_objects else x + 200, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_value, color, 2 if not is_multiple_objects else 1)

def draw_dimension_lines(frame, box, measurements):
    # Draw dimension lines and text on the object box
    mid_bottom = ((box[0] + box[1]) // 2).astype(int)
    cv2.putText(frame, f"{measurements['width_mm']:.1f}mm",
                tuple(mid_bottom + [0, 25]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    mid_left = ((box[0] + box[3]) // 2).astype(int)
    cv2.putText(frame, f"{measurements['height_mm']:.1f}mm",
                tuple(mid_left - [80, 0]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    
    # Draw geometric property indicators for triangles
    if measurements.get('shape_type') == 'triangle':
        # Draw lines representing the slant heights (altitudes) if available
        # Find the center of the triangle
        center_x = int(sum([point[0] for point in box]) / 4)
        center_y = int(sum([point[1] for point in box]) / 4)
        
        # Draw altitude indicators
        if 'triangle_altitude_a_mm' in measurements:
            # Draw a small line indicating the presence of altitude measurement
            cv2.line(frame, (center_x, center_y), (center_x + 20, center_y - 20), (0, 255, 255), 2)
            cv2.putText(frame, "SLANT", (center_x + 25, center_y - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    # Draw geometric property indicators for quadrilaterals
    elif measurements.get('shape_type') == 'quadrilateral':
        # Draw indicators for quadrilateral slant lengths
        # Find the center of the quadrilateral
        center_x = int(sum([point[0] for point in box]) / 4)
        center_y = int(sum([point[1] for point in box]) / 4)
        
        # Draw lines indicating slant properties
        if 'quad_side_0_mm' in measurements:
            # Draw a small line indicating the presence of side length measurement
            cv2.line(frame, (center_x, center_y), (center_x + 20, center_y - 20), (0, 255, 150), 2)
            cv2.putText(frame, "SIDE", (center_x + 25, center_y - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 150), 1)
    
    # Draw geometric property indicators for circles
    elif measurements.get('shape_type') == 'circle':
        # Draw indicators for circular properties
        # Get the center from measurements
        center_x = int(measurements['center'][0]) if 'center' in measurements and isinstance(measurements['center'], (tuple, list)) and len(measurements['center']) >= 2 else int(sum([point[0] for point in box]) / 4)
        center_y = int(measurements['center'][1]) if 'center' in measurements and isinstance(measurements['center'], (tuple, list)) and len(measurements['center']) >= 2 else int(sum([point[1] for point in box]) / 4)
        
        # Draw diameter indicators
        if 'circle_diameter_mm' in measurements:
            # Draw a line indicating the presence of diameter measurement
            cv2.line(frame, (center_x, center_y), (center_x + 20, center_y - 20), (100, 200, 255), 2)
            cv2.putText(frame, "DIA", (center_x + 25, center_y - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 255), 1)

def draw_compact_info_panel(frame, marker_size_px, object_size_px=None, marker_distance=None):
    """
    Draw a compact info panel in top right corner with pixel dimensions
    Args:
        marker_size_px: Tuple of (width, height) in pixels for marker
        object_size_px: Tuple of (width, height) in pixels for object (or None)
        marker_distance: Distance to marker in mm (or None)
    """
    panel_width = 300
    panel_height = 170  # Increased height to accommodate distance info
    margin = 15
    
    # Position in bottom right - ensuring it's away from ROI
    x = max(10, frame.shape[1] - panel_width - margin)  # Ensure minimum left margin
    y = max(100, frame.shape[0] - panel_height - 80)  # Ensure it's above status bar and away from ROI area
    
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
    
    # Distance to marker
    y_offset += line_spacing
    cv2.putText(frame, "Distance:", (x + 15, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    
    if marker_distance is not None:
        distance_text = f"{marker_distance:.1f} mm"
        text_color = (100, 100, 255)
    else:
        distance_text = "--- mm"
        text_color = (100, 100, 100)
    
    cv2.putText(frame, distance_text, (x + 150, y_offset),
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
    
    # Calculate weighted moving average for each measurement (more weight to recent values)
    smoothed = {}
    buffer_len = len(history['measurements_buffer'])
    
    for key in current_measurements:
        if key == 'center':
            # Handle center point separately with weighted average
            x_vals = [m['center'][0] if 'center' in m and isinstance(m['center'], (tuple, list)) and len(m['center']) >= 2 else 0 for m in history['measurements_buffer']]
            y_vals = [m['center'][1] if 'center' in m and isinstance(m['center'], (tuple, list)) and len(m['center']) >= 2 else 0 for m in history['measurements_buffer']]
            
            # Weighted average (more recent values have higher weights)
            total_weight = sum(range(1, buffer_len + 1))
            avg_x = sum(x_vals[i] * (i + 1) for i in range(buffer_len)) / total_weight
            avg_y = sum(y_vals[i] * (i + 1) for i in range(buffer_len)) / total_weight
            
            smoothed['center'] = (avg_x, avg_y)
        elif isinstance(current_measurements[key], (int, float)):
            # Only include values that have the key present
            values = [m[key] for m in history['measurements_buffer'] if key in m]
            if values:  # Only calculate average if there are values
                # Weighted average (more recent values have higher weights)
                total_weight = sum(range(1, len(values) + 1))
                weighted_sum = sum(values[i] * (i + 1) for i in range(len(values)))
                avg_val = weighted_sum / total_weight
                smoothed[key] = avg_val
            else:
                smoothed[key] = current_measurements[key]  # Use current value if no history
        else:
            # For non-numeric values, use the current value from the most recent measurement
            smoothed[key] = current_measurements[key]
    
    # Check for stability
    if len(history['measurements_buffer']) >= min_stable_frames:
        recent_measurements = history['measurements_buffer'][-min_stable_frames:]
        is_stable = True
        
        # Check stability for key measurements
        for key in ['width_mm', 'height_mm', 'area_mm2']:
            if key in current_measurements:
                # Only include values that have the key present
                values = [m[key] for m in recent_measurements if key in m]
                if values and len(values) > 1:
                    min_val = min(values)
                    max_val = max(values)
                    std_dev = np.std(values) if len(values) > 1 else 0
                    avg_val = sum(values) / len(values)
                    
                    # Calculate relative variation and standard deviation
                    if avg_val != 0:
                        variation = (max_val - min_val) / abs(avg_val)
                        if variation > stability_threshold or std_dev > stability_std_threshold:
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
            # Take median for more robustness against outliers
            for key in current_measurements:
                if key != 'center' and isinstance(current_measurements[key], (int, float)):
                    # Only include values that have the key present
                    values = [m[key] for m in history['measurements_buffer'] if key in m]
                    if values and len(values) >= 3:  # Need at least 3 values for good median
                        values.sort()
                        # Use median instead of mean for robustness
                        if len(values) % 2 == 0:
                            median_val = (values[len(values)//2 - 1] + values[len(values)//2]) / 2
                        else:
                            median_val = values[len(values)//2]
                        smoothed[key] = median_val
                    elif values:  # Fallback to weighted average if not enough values for median
                        total_weight = sum(range(1, len(values) + 1))
                        weighted_sum = sum(values[i] * (i + 1) for i in range(len(values)))
                        smoothed[key] = weighted_sum / total_weight
                    else:
                        smoothed[key] = current_measurements[key]  # Use current value if no history
    
    # Round values for stable display
    for key in ['width_mm', 'height_mm', 'area_mm2', 'perimeter_mm', 'angle_deg']:
        if key in smoothed:
            smoothed[key] = round(smoothed[key], 2)
    
    return smoothed



def main():
    """Run the interactive measurement application."""
    # Rebound here because the render loop assigns to them; without
    # `global` they would shadow the module-level copies that
    # apply_advanced_smoothing() reads.
    global measurement_history, smoothing_window, min_stable_frames
    global stability_threshold, stability_std_threshold

    configure()

    # Initialize webcam
    print(f"Selected marker size: {KNOWN_MARKER_SIZE_MM}x{KNOWN_MARKER_SIZE_MM}mm")
    print(f"Selected dictionary: {dict_name}")
    calibrate_camera_intrinsics()  # Show calibration guide
    print("Initializing camera with pose estimation enabled...")
    # Accepts a local camera, a phone streaming over the network, or a video
    # file. Point it at a phone without touching this code:
    #     set INSPECT_AR_SOURCE=http://192.168.1.5:8080/video
    try:
        cap = open_capture()
    except RuntimeError as e:
        print(f"Error: {e}")
        exit(1)

    print("INSPECT-AR - FROM MANUAL INSPECTIONS TO AI INSPECTOR")
    print("Controls: 'q' - Quit, 's' - Save screenshot, 'c' - Contrast, 'r' - Reset, 'u' - Unlock, 'f' - Freeze")

    # State variables
    pixel_to_mm_ratio = None
    calibrated = False
    use_contrast = True
    frame_count = 0
    detection_count = 0

    # Multi-object mode
    single_object_mode = False  # Set to False to enable multiple object detection, True for single object mode

    # Enhanced measurement history for advanced smoothing
    measurement_history = {}
    smoothing_window = 20  # Number of frames to average for stability (increased for better smoothing)
    min_stable_frames = 8  # Minimum frames to consider measurement stable
    stability_threshold = 0.010  # 1% threshold for stability (reduced for more sensitivity)
    stability_std_threshold = 0.5  # Standard deviation threshold for stability

    # YOLO detector initialization
    yolo_detector = YOLODetector()

    # Lock-on mechanism variables
    # Locking mechanism variables
    LOCK_THRESHOLD_SECONDS = 5.0  # Lock after 5 seconds of consistent detection
    is_locked = False  # Global lock state
    locked_objects = {}  # Dictionary to store locked objects by ID
    first_detection_times = {}  # Track detection start time for each object
    consistent_detection_timers = {}  # Track consistent detection time for each object
    lock_activation_time = None  # Time when the lock was activated

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
        cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 90),
                      (15, 15, 25), -1)
        cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 90),
                      (14, 165, 233), 2)
        cv2.putText(display_frame, "INSPECT-AR", 
                    (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1.2, (14, 165, 233), 2)
        cv2.putText(display_frame, "FROM MANUAL INSPECTIONS TO AI INSPECTOR", 
                    (20, 70), cv2.FONT_HERSHEY_DUPLEX, 0.6, (14, 165, 233), 1)

        # Estimate pose of markers for 3D information
        rvecs, tvecs, marker_distances, marker_tilt_angles = estimate_marker_pose(corners, ids, marker_length) if ids is not None else (None, None, None, None)

        if ids is not None and len(ids) > 0:
            # Calibrate using first marker
            if not calibrated:
                pixel_to_mm_ratio, avg_size = compute_pixel_to_mm_ratio(
                    corners[0], KNOWN_MARKER_SIZE_MM
                )
                calibrated = True

            # Apply perspective correction based on marker tilt angles
            if calibrated and marker_tilt_angles is not None and len(marker_tilt_angles) > 0:
                # Get tilt angles for the first marker
                tilt = marker_tilt_angles[0]
                pitch = abs(tilt['pitch'])
                yaw = abs(tilt['yaw'])

                # Calculate the effective tilt angle for perspective correction
                # Use the maximum of pitch and yaw as the primary tilt factor
                tilt_angle = max(pitch, yaw)

                # Calculate correction factor based on tilt
                # The cosine of the tilt angle is used to correct for perspective distortion
                cos_tilt = np.abs(np.cos(tilt_angle))  # Use absolute value to handle negative angles
                # Limit the correction factor to prevent extreme values when marker is nearly perpendicular
                cos_tilt = max(0.3, cos_tilt)  # Don't let it go below 0.3 to prevent extreme corrections
                perspective_correction_factor = 1.0 / cos_tilt

                # Only apply correction if tilt is significant (> 10 degrees)
                if tilt_angle > 0.175:  # About 10 degrees
                    # Also apply distance-based correction
                    distance_correction_factor = 1.0
                    if marker_distances is not None and len(marker_distances) > 0:
                        current_distance = marker_distances[0]  # Distance to first marker in mm
                        # Reference distance (when marker is at ideal position)
                        reference_distance = KNOWN_MARKER_SIZE_MM * 10  # Adjust as needed
                        distance_correction_factor = current_distance / reference_distance

                    # Combine both corrections
                    corrected_pixel_to_mm_ratio = pixel_to_mm_ratio * perspective_correction_factor * distance_correction_factor
                else:
                    corrected_pixel_to_mm_ratio = pixel_to_mm_ratio  # No significant tilt, use base ratio
            else:
                corrected_pixel_to_mm_ratio = pixel_to_mm_ratio

            # Calculate marker pixel size (continuously updates, not locked yet)
            corner_points = corners[0][0]
            marker_width_px = np.linalg.norm(corner_points[0] - corner_points[1])
            marker_height_px = np.linalg.norm(corner_points[1] - corner_points[2])
            current_marker_size_px = (marker_width_px, marker_height_px)

            # Draw detected markers
            cv2.aruco.drawDetectedMarkers(display_frame, corners, ids)

            # Draw coordinate axes on markers to visualize orientation
            # Note: Axis drawing temporarily disabled due to OpenCV version compatibility
            # if rvecs is not None and tvecs is not None:
            #    draw_axis_on_marker(display_frame, rvecs, tvecs, CAMERA_MATRIX, DIST_COEFFS, marker_length)

            # Define ROI near marker
            roi_rect, roi_center = define_roi_near_marker(corners[0], frame.shape)
            x, y, w, h = roi_rect

            # Draw ROI boundary
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
            cv2.putText(display_frame, "ROI", (x + 5, y + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

            # Detection locking logic
            if single_object_mode:
                # Single object detection mode - use global lock
                if not is_locked:
                    # Single object detection mode - use YOLO to detect best object in ROI
                    detections, roi_frame = detect_objects_with_yolo_in_roi(
                        frame, roi_rect, yolo_detector, marker_corners=corners[0]
                    )

                    # Get the best detection (highest confidence) for single object mode
                    detection = None
                    best_contour = None
                    if detections:
                        # Sort by confidence to get the best detection
                        detections_sorted = sorted(detections, key=lambda x: x['confidence'], reverse=True)
                        detection = detections_sorted[0]
                        best_contour = detection.get('contour')

                    if detection is not None and best_contour is not None:
                        detection_count += 1

                        # Start or continue timer for consistent detection
                        current_time = time.time()
                        detection_id = detection['id']
                        if detection_id not in first_detection_times:
                            first_detection_times[detection_id] = current_time
                            consistent_detection_timers[detection_id] = 0.0
                        else:
                            consistent_detection_timers[detection_id] = current_time - first_detection_times[detection_id]

                        # Use corrected pixel-to-mm ratio if available, otherwise use base ratio
                        current_pixel_to_mm_ratio = corrected_pixel_to_mm_ratio if 'corrected_pixel_to_mm_ratio' in locals() else pixel_to_mm_ratio

                        # Measure object with rotation and tilt correction
                        measurements, rect = measure_object_with_tilt_correction(best_contour, current_pixel_to_mm_ratio, marker_tilt_angles)

                        if measurements is not None:
                            # Apply advanced smoothing to reduce fluctuation
                            smoothed_measurements = apply_advanced_smoothing(detection['id'], measurements)

                            # Get object actual pixel size from rotated rect (not YOLO bbox)
                            # This is the actual measured dimension used in calculations
                            obj_width_px = smoothed_measurements['width_mm'] / current_pixel_to_mm_ratio
                            obj_height_px = smoothed_measurements['height_mm'] / current_pixel_to_mm_ratio
                            current_object_size_px = (obj_width_px, obj_height_px)

                            # Check if we should lock onto this object (5 seconds)
                            if consistent_detection_timers[detection_id] >= LOCK_THRESHOLD_SECONDS and not is_locked:
                                is_locked = True
                                lock_activation_time = time.time()  # Record when the lock was activated
                                locked_objects[detection_id] = {
                                    'measurements': smoothed_measurements.copy(),
                                    'rect': rect,
                                    'box': cv2.boxPoints(rect),
                                    'confidence': detection['confidence']
                                }
                                locked_objects[detection_id]['box'] = np.intp(locked_objects[detection_id]['box'])
                                locked_objects[detection_id]['box'][:, 0] += x
                                locked_objects[detection_id]['box'][:, 1] += y
                                # Lock BOTH marker and object pixel dimensions together
                                locked_marker_size_px = current_marker_size_px
                                locked_object_size_px = current_object_size_px
                                print(f"🔒 LOCKED onto object (ID: {detection['id']}, Confidence: {detection['confidence']:.2f}) after {consistent_detection_timers[detection_id]:.1f}s")
                                print(f"📏 Locked dimensions - Marker: {locked_marker_size_px[0]:.0f}x{locked_marker_size_px[1]:.0f}px, Object: {locked_object_size_px[0]:.0f}x{locked_object_size_px[1]:.0f}px")

                            # Draw rotated rectangle
                            box = draw_rotated_rectangle(display_frame, rect, (x, y), 
                                                        color=(0, 255, 0), thickness=3)

                            # Draw dimension lines with smoothed values
                            draw_dimension_lines(display_frame, box, smoothed_measurements)

                            # Draw measurement overlay - moved to left side to avoid ROI
                            distance_info = marker_distances[0] if marker_distances is not None and len(marker_distances) > 0 else None
                            draw_measurement_overlay(display_frame, smoothed_measurements, (20, 100), distance_mm=distance_info, is_multiple_objects=False, roi_rect=roi_rect)

                            # Draw center point
                            center_abs = (int(smoothed_measurements['center'][0]) + x, 
                                         int(smoothed_measurements['center'][1]) + y)
                            cv2.circle(display_frame, center_abs, 6, (0, 255, 0), -1)
                            cv2.circle(display_frame, center_abs, 10, (0, 255, 0), 2)

                            # Draw compact info panel in bottom right (away from ROI)
                            # If locked: show locked values, otherwise show current marker + no object yet
                            display_marker_size = locked_marker_size_px if is_locked else current_marker_size_px
                            display_object_size = locked_object_size_px if is_locked else None
                            display_distance = marker_distances[0] if marker_distances is not None and len(marker_distances) > 0 else None
                            draw_compact_info_panel(display_frame, display_marker_size, display_object_size, marker_distance=display_distance)

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
                            if consistent_detection_timers[detection_id] < LOCK_THRESHOLD_SECONDS:
                                status_text = f"✅ DETECTING... Lock in {LOCK_THRESHOLD_SECONDS - consistent_detection_timers[detection_id]:.1f}s"
                                status_color = (0, 255, 255)
                            else:
                                status_text = "🔒 LOCKED ON TARGET - Press 'u' to unlock"
                                status_color = (0, 255, 0)
                        else:
                            status_text = "⚠️ CONTOUR TOO SMALL"
                            status_color = (0, 165, 255)
                            if detection_id in first_detection_times:
                                del first_detection_times[detection_id]
                                del consistent_detection_timers[detection_id]
                    else:
                        status_text = "🔍 SEARCHING FOR OBJECT IN ROI (YOLO)..."
                        status_color = (255, 200, 0)
                        # Clear any tracking for this detection ID if no detection
                        detection_id = None

                        # Still show ROI for context
                        roi_frame = frame[y:y+h, x:x+w].copy()

                        # Draw info panel in bottom right: show current marker size (updating) + no object
                        display_distance = marker_distances[0] if marker_distances is not None and len(marker_distances) > 0 else None
                        draw_compact_info_panel(display_frame, current_marker_size_px, None, marker_distance=display_distance)
                else:
                    # Locked mode for single object - display frozen measurements
                    # Draw all locked objects
                    for idx, (obj_id, locked_obj) in enumerate(locked_objects.items()):
                        # Draw locked rotated rectangle
                        cv2.drawContours(display_frame, [locked_obj['box']], 0, (255, 0, 255), 3)

                        # Draw dimension lines with locked values
                        draw_dimension_lines(display_frame, locked_obj['box'], locked_obj['measurements'])

                        # Draw center point with locked measurements
                        center_abs = (int(locked_obj['measurements']['center'][0]) + x, 
                                     int(locked_obj['measurements']['center'][1]) + y)
                        cv2.circle(display_frame, center_abs, 6, (255, 0, 255), -1)
                        cv2.circle(display_frame, center_abs, 10, (255, 0, 255), 2)

                        # Draw measurement overlay for each locked object
                        distance_info = marker_distances[0] if marker_distances is not None and len(marker_distances) > 0 else None
                        # Use smaller table if in multiple object mode
                        # Position each overlay differently to avoid overlap
                        overlay_x = 20 + (idx % 3) * 400  # Position horizontally in 3 columns
                        overlay_y = 100 + (idx // 3) * 500  # Position vertically in rows
                        draw_measurement_overlay(display_frame, locked_obj['measurements'], (overlay_x, overlay_y), 
                                                object_label=f"OBJ {idx+1} LOCKED", distance_mm=distance_info, 
                                                is_multiple_objects=(not single_object_mode), roi_rect=roi_rect)

                    # Draw compact info panel in bottom right with permanently locked values
                    display_distance = marker_distances[0] if marker_distances is not None and len(marker_distances) > 0 else None
                    draw_compact_info_panel(display_frame, locked_marker_size_px, locked_object_size_px, marker_distance=display_distance)

                    # Status
                    status_text = f"🔒 {len(locked_objects)} OBJECTS LOCKED - Will remain locked until 'u' is pressed"
                    status_color = (255, 0, 255)

                    # Still show ROI for context (but no detection)
                    roi_frame = frame[y:y+h, x:x+w].copy()
            else:
                # Multiple object detection mode - detect and measure all objects in ROI
                # NOTE: In multiple object mode, we allow continuous detection even when some objects are locked
                detections, roi_frame = detect_objects_with_yolo_in_roi(
                    frame, roi_rect, yolo_detector, marker_corners=corners[0]
                )

                # Process all detected objects
                if detections:
                    detection_count += len(detections)

                    # Use corrected pixel-to-mm ratio if available, otherwise use base ratio
                    current_pixel_to_mm_ratio = corrected_pixel_to_mm_ratio if 'corrected_pixel_to_mm_ratio' in locals() else pixel_to_mm_ratio

                    # Process each detected object
                    for i, detection in enumerate(detections):
                        contour = detection.get('contour')
                        if contour is not None:
                            # Create a unique ID for this detection in multiple object mode
                            bbox = detection['bbox']
                            x1, y1, x2, y2 = [int(coord) for coord in bbox]

                            # Check if this detection is close to any locked object
                            detection_id = None
                            for locked_id, locked_obj in locked_objects.items():
                                # Calculate distance between centers
                                if 'center' in locked_obj['measurements']:
                                    locked_center_x = int(locked_obj['measurements']['center'][0]) + x
                                    locked_center_y = int(locked_obj['measurements']['center'][1]) + y
                                    detection_center_x = x1 + (x2 - x1) // 2
                                    detection_center_y = y1 + (y2 - y1) // 2

                                    distance = np.sqrt((locked_center_x - detection_center_x)**2 + (locked_center_y - detection_center_y)**2)

                                    # If the detection is close to a locked object, use the same ID
                                    if distance < 100:  # Threshold in pixels
                                        detection_id = locked_id
                                        break

                            # If no close locked object found, create a new ID
                            if detection_id is None:
                                bbox_str = f"{x1}_{y1}_{x2-x1}_{y2-y1}_{int(time.time())}"
                                detection_id = f"obj_{bbox_str}"

                            # Measure object with rotation and tilt correction
                            measurements, rect = measure_object_with_tilt_correction(contour, current_pixel_to_mm_ratio, marker_tilt_angles)

                            if measurements is not None:
                                # If this detection corresponds to a locked object, skip processing and just draw the locked version
                                if detection_id in locked_objects:
                                    # Draw the locked object (this will be done later in the display section)
                                    continue

                                # Start or continue timer for consistent detection for this object
                                current_time = time.time()
                                if detection_id not in first_detection_times:
                                    first_detection_times[detection_id] = current_time
                                    consistent_detection_timers[detection_id] = 0.0
                                else:
                                    consistent_detection_timers[detection_id] = current_time - first_detection_times[detection_id]

                                # Apply advanced smoothing to reduce fluctuation
                                smoothed_measurements = apply_advanced_smoothing(detection_id, measurements)

                                # Check if we should lock onto this object (5 seconds)
                                # In multiple object mode, we can lock individual objects independently
                                if consistent_detection_timers[detection_id] >= LOCK_THRESHOLD_SECONDS:
                                    # Only lock if not already locked
                                    if detection_id not in locked_objects:
                                        locked_objects[detection_id] = {
                                            'measurements': smoothed_measurements.copy(),
                                            'rect': rect,
                                            'box': cv2.boxPoints(rect),
                                            'confidence': detection['confidence']
                                        }
                                        locked_objects[detection_id]['box'] = np.intp(locked_objects[detection_id]['box'])
                                        locked_objects[detection_id]['box'][:, 0] += x
                                        locked_objects[detection_id]['box'][:, 1] += y
                                        print(f"🔒 LOCKED onto object {i+1} (ID: {detection_id}, Conf: {detection['confidence']:.2f}) after {consistent_detection_timers[detection_id]:.1f}s")

                                # Draw rotated rectangle for this object
                                # If object is locked, use locked rectangle, otherwise use current detection
                                if detection_id in locked_objects:
                                    # Use the locked box which is already offset
                                    cv2.drawContours(display_frame, [locked_objects[detection_id]['box']], 0, (255, 0, 255), 3)
                                    # Draw dimension lines with locked values
                                    draw_dimension_lines(display_frame, locked_objects[detection_id]['box'], locked_objects[detection_id]['measurements'])
                                    # Draw center point with locked measurements
                                    # The center in measurements is relative to ROI, so add ROI offset
                                    locked_center_abs = (int(locked_objects[detection_id]['measurements']['center'][0]) + x, 
                                                 int(locked_objects[detection_id]['measurements']['center'][1]) + y)
                                    cv2.circle(display_frame, locked_center_abs, 6, (255, 0, 255), -1)
                                    cv2.circle(display_frame, locked_center_abs, 10, (255, 0, 255), 2)
                                else:
                                    # Use current detection
                                    box = draw_rotated_rectangle(display_frame, rect, (x, y), 
                                                                color=(0, 255, 0), thickness=2)
                                    # Draw dimension lines with smoothed values
                                    draw_dimension_lines(display_frame, box, smoothed_measurements)
                                    # Draw center point
                                    center_abs = (int(smoothed_measurements['center'][0]) + x, 
                                                 int(smoothed_measurements['center'][1]) + y)
                                    cv2.circle(display_frame, center_abs, 4, (0, 255, 0), -1)
                                    cv2.circle(display_frame, center_abs, 7, (0, 255, 0), 1)

                                # Draw detection bbox from YOLO
                                bbox = detection['bbox']
                                bbox_x1, bbox_y1, bbox_x2, bbox_y2 = [int(coord) for coord in bbox]
                                cv2.rectangle(roi_frame, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), (0, 255, 255), 1)

                                # Draw confidence and detection info
                                conf_text = f"Conf: {detection['confidence']:.2f}"
                                cv2.putText(roi_frame, conf_text, (bbox_x1, bbox_y1 - 5),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

                                # Draw measurement overlay with unique label for each object
                                distance_info = marker_distances[0] if marker_distances is not None and len(marker_distances) > 0 else None
                                # Position each overlay differently to avoid overlap
                                overlay_x = 20 + (i % 3) * 400  # Position horizontally in 3 columns
                                overlay_y = 100 + (i // 3) * 500  # Position vertically in rows

                                # If object is locked, use locked measurements
                                if detection_id in locked_objects:
                                    draw_measurement_overlay(display_frame, locked_objects[detection_id]['measurements'], (overlay_x, overlay_y), 
                                                            object_label=f"OBJ {i+1} LOCKED", distance_mm=distance_info, is_multiple_objects=True, roi_rect=roi_rect)
                                else:
                                    draw_measurement_overlay(display_frame, smoothed_measurements, (overlay_x, overlay_y), 
                                                            object_label=f"OBJ {i+1}", distance_mm=distance_info, is_multiple_objects=True, roi_rect=roi_rect)

                                # Draw detection bbox from YOLO
                                bbox = detection['bbox']
                                bbox_x1, bbox_y1, bbox_x2, bbox_y2 = [int(coord) for coord in bbox]
                                cv2.rectangle(roi_frame, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), (0, 255, 255), 1)

                                # Draw confidence and detection info
                                conf_text = f"Conf: {detection['confidence']:.2f}"
                                cv2.putText(roi_frame, conf_text, (bbox_x1, bbox_y1 - 5),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

                    # Draw all locked objects that weren't detected in this frame
                    for locked_id, locked_obj in locked_objects.items():
                        # Check if this locked object was not just processed
                        was_detected_this_frame = False
                        for detection in detections:
                            contour = detection.get('contour')
                            if contour is not None:
                                bbox = detection['bbox']
                                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                                # Calculate distance between centers
                                if 'center' in locked_obj['measurements']:
                                    locked_center_x = int(locked_obj['measurements']['center'][0]) + x
                                    locked_center_y = int(locked_obj['measurements']['center'][1]) + y
                                    detection_center_x = x1 + (x2 - x1) // 2
                                    detection_center_y = y1 + (y2 - y1) // 2

                                    distance = np.sqrt((locked_center_x - detection_center_x)**2 + (locked_center_y - detection_center_y)**2)

                                    if distance < 100:  # Same threshold as used for ID assignment
                                        was_detected_this_frame = True
                                        break

                        # If this locked object wasn't detected this frame, draw it separately
                        if not was_detected_this_frame:
                            # Use the locked box which is already offset
                            cv2.drawContours(display_frame, [locked_obj['box']], 0, (255, 0, 255), 3)
                            # Draw dimension lines with locked values
                            draw_dimension_lines(display_frame, locked_obj['box'], locked_obj['measurements'])
                            # Draw center point with locked measurements
                            # The center in measurements is relative to ROI, so add ROI offset
                            locked_center_abs = (int(locked_obj['measurements']['center'][0]) + x, 
                                                 int(locked_obj['measurements']['center'][1]) + y)
                            cv2.circle(display_frame, locked_center_abs, 6, (255, 0, 255), -1)
                            cv2.circle(display_frame, locked_center_abs, 10, (255, 0, 255), 2)

                            # Draw measurement overlay for this locked object
                            distance_info = marker_distances[0] if marker_distances is not None and len(marker_distances) > 0 else None
                            # Position this overlay differently to avoid overlap with others
                            # Use a simple counter to position it
                            all_locked_ids = list(locked_objects.keys())
                            idx = all_locked_ids.index(locked_id)
                            overlay_x = 20 + (idx % 3) * 400  # Position horizontally in 3 columns
                            overlay_y = 100 + (idx // 3) * 500  # Position vertically in rows
                            draw_measurement_overlay(display_frame, locked_obj['measurements'], (overlay_x, overlay_y), 
                                                    object_label="LOCKED OBJ", distance_mm=distance_info, is_multiple_objects=True, roi_rect=roi_rect)

                    # Status for multiple object detection
                    locked_count = len(locked_objects)
                    if locked_count > 0:
                        status_text = f"✅ {len(detections)} OBJ DETECTED, {locked_count} LOCKED - Will remain locked until 'u' is pressed"
                        status_color = (255, 0, 255)  # Purple to indicate locked objects
                    else:
                        status_text = f"✅ DETECTED {len(detections)} OBJECTS IN ROI"
                        status_color = (0, 255, 0)

                    # Draw compact info panel in bottom right (away from ROI)
                    display_distance = marker_distances[0] if marker_distances is not None and len(marker_distances) > 0 else None
                    draw_compact_info_panel(display_frame, current_marker_size_px, None, marker_distance=display_distance)
                else:
                    status_text = "🔍 SEARCHING FOR OBJECTS IN ROI (YOLO)..."
                    status_color = (255, 200, 0)

                    # Still show ROI for context
                    roi_frame = frame[y:y+h, x:x+w].copy()

                    # Draw info panel in bottom right: show current marker size (updating) + no object
                    display_distance = marker_distances[0] if marker_distances is not None and len(marker_distances) > 0 else None
                    draw_compact_info_panel(display_frame, current_marker_size_px, None, marker_distance=display_distance)

            # Show ROI and binary mask in corner (for debugging) - moved to bottom left
            if roi_frame is not None and roi_frame.size > 0 and roi_frame.shape[0] > 0 and roi_frame.shape[1] > 0:
                roi_small = cv2.resize(roi_frame, (200, 150))
                # Position at bottom left, above the status bar
                y_pos = display_frame.shape[0] - 190
                display_frame[y_pos:y_pos+150, 10:210] = roi_small
                cv2.rectangle(display_frame, (10, y_pos-20), (210, y_pos+150), (255, 200, 0), 2)
                cv2.putText(display_frame, "ROI View", (15, y_pos-5),
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

        # Show frame in a normal resizable window
        cv2.namedWindow('INSPECT-AR - Industrial Object Measurement System', cv2.WINDOW_NORMAL)
        cv2.imshow('INSPECT-AR - Industrial Object Measurement System', display_frame)

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
            lock_activation_time = None  # Reset lock activation time
            locked_objects.clear()  # Clear all locked objects
            first_detection_times.clear()  # Clear all detection timers
            consistent_detection_timers.clear()  # Clear all consistent detection timers
            measurement_history.clear()
            locked_marker_size_px = None
            locked_object_size_px = None
        elif key == ord('u'):
            # Unlock detection - clear all locked objects and reset detection state
            is_locked = False
            lock_activation_time = None  # Reset lock activation time
            locked_objects.clear()  # Clear all locked objects
            first_detection_times.clear()  # Clear all detection timers
            consistent_detection_timers.clear()  # Clear all consistent detection timers
            measurement_history.clear()
            locked_object_size_px = None  # Clear object size but keep marker size
            print("🔓 Unlocked - resuming YOLO detection for all objects")
        elif key == ord('f'):
            cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()




if __name__ == "__main__":
    main()