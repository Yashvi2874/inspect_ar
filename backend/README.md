# ZENITH INSPECT-AR

Industrial measurement system using OpenCV and ArUco markers for precise dimension detection of objects.

## 🏗️ Project Structure

```
backend/
├── app.py                    # Flask REST API server
├── industrial_measurement.py  # Real-time measurement system
├── requirements.txt          # Python dependencies
└── aruco_markers/            # Generated ArUco marker files
```

## 🛠️ Features

- **ArUco Marker Calibration**: Precise camera calibration using printed markers
- **Real-time Dimension Measurement**: Width, height, area, and perimeter in mm
- **ROI-based Detection**: Automatic region of interest around markers
- **Rotation Handling**: minAreaRect for accurate measurements regardless of object orientation
- **Stable Measurements**: Advanced smoothing to reduce fluctuation
- **Visual Overlay**: Professional measurement panels with dimension lines

## 🚀 Quick Start

### Prerequisites
- Python 3.8+

### Installation

```bash
pip install -r backend/requirements.txt
```

### Running the Application

```bash
python backend/industrial_measurement.py
```

## 📊 Technical Specifications

### ArUco Calibration
- Dictionary: `DICT_6X6_250` (default)
- Marker size: Configurable (default 50mm)
- Pixel-to-mm conversion with enhanced accuracy

### Object Detection
- Method: Contour-based segmentation
- Preprocessing: Bilateral filter, adaptive thresholding
- Morphological operations for noise reduction

### Dimension Measurement
- **Width**: Longest dimension (mm)
- **Height**: Shortest dimension (mm) 
- **Area**: Surface area (mm²)
- **Perimeter**: Boundary length (mm)
- **Rotation Angle**: Object orientation (degrees)
- **Aspect Ratio**: Width/Height ratio

### ROI Configuration
- Size: 6× marker size (larger coverage)
- Offset: 1.2× marker width from marker
- Position: To the right of marker

## 📐 Measurement Pipeline

```
Camera Feed → ArUco Detection → Calibration → ROI Definition → Object Detection → Dimension Measurement → Smoothing → Display
```

## 🎮 Controls

- **q**: Quit application
- **s**: Save screenshot with measurements
- **c**: Toggle contrast enhancement
- **r**: Reset calibration
- **f**: Freeze frame

## 📞 Support

For technical support, contact the development team.

---

*Developed with ❤️ for industrial measurement applications*