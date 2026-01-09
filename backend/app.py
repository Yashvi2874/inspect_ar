"""
ZENITH INSPECT-AR Backend API
Flask server for aerospace industrial object dimension measurement and defect detection
"""

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from pathlib import Path

from models.aruco_calibration import ArucoCalibrator
from models.dimension_detector import DimensionDetector
from models.defect_detector import DefectDetector
from models.yolo_detector import YOLODetector
from utils.image_processor import ImageProcessor

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = Path('uploads')
RESULTS_FOLDER = Path('results')
MODELS_FOLDER = Path('models/weights')

UPLOAD_FOLDER.mkdir(exist_ok=True)
RESULTS_FOLDER.mkdir(exist_ok=True)
MODELS_FOLDER.mkdir(exist_ok=True)

app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'mp4', 'avi'}

# Initialize models
print("Loading models...")
aruco_calibrator = ArucoCalibrator()
dimension_detector = DimensionDetector(models_folder=MODELS_FOLDER)
defect_detector = DefectDetector(models_folder=MODELS_FOLDER)
yolo_detector = YOLODetector(models_folder=MODELS_FOLDER)
image_processor = ImageProcessor()

print("Models loaded successfully!")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'online',
        'service': 'ZENITH INSPECT-AR',
        'models': {
            'aruco': 'ready',
            'yolo': 'ready',
            'vit': 'ready'
        }
    })


@app.route('/api/calibrate', methods=['POST'])
def calibrate_camera():
    """Calibrate camera using ArUco marker"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Read image
    image = cv2.imread(filepath)
    
    # Detect ArUco markers and calibrate
    result = aruco_calibrator.calibrate(image)
    
    if result['success']:
        return jsonify({
            'success': True,
            'calibration_data': result['calibration'],
            'marker_count': result['marker_count'],
            'reference_size_mm': result['reference_size_mm']
        })
    else:
        return jsonify({
            'success': False,
            'error': result['error']
        }), 400


@app.route('/api/analyze', methods=['POST'])
def analyze_object():
    """Complete analysis: detection, dimension measurement, and defect detection"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400
    
    # Get calibration data if provided
    calibration_data = request.form.get('calibration_data', None)
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Read image
    image = cv2.imread(filepath)
    
    try:
        # Step 1: Detect ArUco markers for scale calibration
        aruco_result = aruco_calibrator.detect_and_calibrate(image)
        
        # Step 2: YOLO object detection and segmentation
        yolo_result = yolo_detector.detect(image)
        
        # Step 3: Dimension measurement
        dimension_result = dimension_detector.measure(
            image, 
            yolo_result['detections'],
            aruco_result['pixel_to_mm_ratio']
        )
        
        # Step 4: Defect detection using ViT
        defect_result = defect_detector.detect(
            image,
            yolo_result['detections']
        )
        
        # Combine results
        response = {
            'success': True,
            'calibration': {
                'markers_detected': aruco_result['marker_count'],
                'pixel_to_mm_ratio': aruco_result['pixel_to_mm_ratio'],
                'reference_size': aruco_result.get('reference_size_mm', 210)  # A4 width
            },
            'objects_detected': len(yolo_result['detections']),
            'detections': [],
            'overall_defect_status': 'PASS'
        }
        
        # Process each detected object
        for i, detection in enumerate(yolo_result['detections']):
            obj_id = detection['id']
            
            obj_result = {
                'id': obj_id,
                'class': detection['class'],
                'confidence': float(detection['confidence']),
                'bbox': detection['bbox'],
                'dimensions': dimension_result['objects'][obj_id],
                'defects': defect_result['objects'][obj_id]
            }
            
            response['detections'].append(obj_result)
            
            # Update overall status
            if defect_result['objects'][obj_id]['has_defects']:
                response['overall_defect_status'] = 'FAIL'
        
        # Calculate overall confidence
        response['confidence'] = f"{dimension_result['overall_confidence']:.1f}%"
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/train/status', methods=['GET'])
def training_status():
    """Get training status"""
    return jsonify({
        'yolo_trained': os.path.exists(MODELS_FOLDER / 'yolo_best.pt'),
        'vit_trained': os.path.exists(MODELS_FOLDER / 'vit_defect_best.pth'),
        'training_active': False
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 ZENITH INSPECT-AR Backend Server")
    print("="*60)
    print("ArUco Calibration: Ready")
    print("YOLO Detection: Ready")
    print("ViT Defect Detection: Ready")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
