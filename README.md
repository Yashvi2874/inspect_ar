INSPECT-AR: AI-Powered Industrial Object Inspection System From Manual Inspection to AI Inspector
HAL Aerospace Challenge
Aerothon Hackathon
Team ZENITH 
KJ Somaiya College of Engineering
Team Members
Yashasvi Gupta (16010123341) – Computer Engineering
Sai Parcha (16014223072) – AI & DS Engineering
Aastha Shah (16010523003) – Mechanical Engineering

Abstract
INSPECT-AR is a real-time, mobile-assisted AI-powered quality inspection prototype developed for aerospace component manufacturing under the HAL Aerothon challenge.
The system integrates classical computer vision with modern deep learning to achieve:
•	Real-time anomaly & defect detection
•	High-precision dimension measurement (±1–2 mm tolerance)
•	Live visual feedback with overlays at 15+ FPS
•	Exportable inspection reports (CSV/JSON)
Built using a smartphone camera (via IP Webcam), ArUco markers for dimensioning & orientation, YOLO for component localization, and Faster R-CNN for detailed defect classification, INSPECT-AR demonstrates a practical, low-cost transition from error-prone manual inspection to consistent, explainable, AI-augmented quality control.
Hackathon Objectives (Achieved)
1.	Detect anomalies & defects with visual feedback
2.	Perform real-time size measurement from live video (±1–2 mm accuracy)
3.	Deliver a working prototype with ≥15 FPS processing
Solution Implemented
Core Functional Modules
•	Live Video Feed → IP Webcam mobile app → HTTP MJPEG stream
•	Component Localization → YOLOv8 (fast & lightweight object detection)
•	Dimension Measurement → ArUco marker detection + single marker pose estimation + OpenCV PnP + calibrated pixel-to-mm conversion
•	Defect Detection → Faster R-CNN (ResNet-50-FPN) for multi-class defect classification
•	Orientation Awareness → ArUco pose estimation (helps conditional defect validation)
•	Visualization → Real-time overlays: bounding boxes and dimensions.
Uniqueness Highlights
•	Fully mobile-friendly (live camera feed from mobile can be connected to any device)
•	No internet required
•	Explainable AI through visual overlays
•	ArUco-based high-accuracy metrology without expensive hardware
•	 CV + deep learning pipeline optimized for standard laptops

Key Technologies Used
•	OpenCV → Frame processing, ArUco detection, pose estimation, drawing overlays, calibration
•	Ultralytics YOLOv8 → Fast component & missing part detection
•	Torchvision Faster R-CNN → High-accuracy defect classification (crack, dent, scratch, corrosion, etc.)
•	Python → Core orchestration, multi-threading for parallel inference
•	estimatePoseSingleMarkers function in aruco, for rotation and translation vectors
•	IP Webcam → Zero-cost, high-quality mobile camera streaming

Github link- https://github.com/Yashvi2874/inspect_ar

Datasets Used:
https://universe.roboflow.com/aircraft-defects/aircraft-defects-8plvs/dataset/1
https://universe.roboflow.com/dibya-dillip/aircraft-skin-defects-new-dataset/dataset/2
https://universe.roboflow.com/defectdatasets/neu-det-fquva/dataset/1
https://universe.roboflow.com/defectdatasets/magnatic-tile-defect/dataset/1

Future Enhancement Directions
•	Web/mobile dashboard interface
•	Automatic datasheet parsing & dynamic tolerance configuration
•	Active learning for continuous model improvement
•	Edge deployment (NVIDIA Jetson / mobile NNAPI)
•	Integration of thermal & ultrasonic modalities

Conclusion
INSPECT-AR successfully delivers a practical, working prototype that meets all core requirements of the HAL Aerothon Aerospace Challenge.
