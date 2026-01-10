<div  align="center">
  
# INSPECT-AR  
**AI-Powered Industrial Object Inspection System**  
From Manual Inspection to AI Inspector
![WhatsApp Image 2026-01-10 at 10 57 53](https://github.com/user-attachments/assets/37184715-7663-4e66-92bd-a9b3a7ec6347)

</div>

<p align="center">
  <strong>HAL Aerospace Challenge — Aerothon Hackathon</strong><br>
  <strong>Team ZENITH</strong> — KJ Somaiya College of Engineering
</p>

![WhatsApp Image 2026-01-08 at 17 17 02](https://github.com/user-attachments/assets/d0c6370f-2672-4198-9f87-c6644fe6cd58)

<p align="center">
  <img src="https://img.shields.io/badge/Hackathon-HAL%20Aerothon-blue?style=for-the-badge" alt="HAL Aerothon">
  <img src="https://img.shields.io/badge/Status-Prototype-success?style=for-the-badge" alt="Status">
  <img src="https://img.shields.io/badge/FPS-15%2B-green?style=for-the-badge" alt="Performance">
</p>

## Abstract

**INSPECT-AR** is a real-time, mobile-assisted AI-powered quality inspection prototype developed specifically for aerospace component manufacturing under the HAL Aerothon challenge.

The system combines classical computer vision with modern deep learning to deliver:

- Real-time anomaly & defect detection
- High-precision dimension measurement (±1–2 mm tolerance)
- Live visual feedback with AR-style overlays at **15+ FPS**
- Exportable inspection reports (CSV/JSON)

Built using a smartphone camera (via IP Webcam), ArUco markers for accurate dimensioning & orientation, YOLO for fast component localization, and Faster R-CNN for detailed defect classification — INSPECT-AR offers a practical, low-cost solution to transition from error-prone manual inspection to consistent, explainable, AI-augmented quality control.

## Hackathon Objectives (Achieved ✅)

1. Detect anomalies & defects with intuitive visual feedback  
2. Perform real-time size measurement from live video stream (±1–2 mm accuracy)  
3. Deliver a fully working prototype with ≥15 FPS processing speed

## Solution Architecture — Core Functional Modules

```mermaid
graph TD
    A[Smartphone Camera<br>IP Webcam MJPEG Stream] --> B[Live Video Feed]
    B --> C[YOLOv8<br>Component Localization & Missing Parts]
    B --> D[ArUco Marker Detection<br>Pose Estimation + PnP]
    D --> E[Pixel-to-mm Calibration<br>Dimension Measurement ±1-2mm]
    B --> F[Faster R-CNN<br>Multi-class Defect Classification]
    C --> G[Real-time Overlays<br>Bounding Boxes + Dimensions]
    F --> G
    E --> G
    G --> H[Inspection Report<br>CSV / JSON Export]
