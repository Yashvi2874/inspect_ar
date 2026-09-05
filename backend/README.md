# INSPECT-AR — backend

Two things live here: a working real-time dimension-measurement application, and
the in-progress integrated inspection pipeline.

See [../docs/KNOWN_ISSUES.md](../docs/KNOWN_ISSUES.md) for the current state of
each component and [../docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) for how they
fit together.

## Layout

```
backend/
├── industrial_measurement.py          # real-time ArUco measurement app (works)
├── integrated_inspection_pipeline.py  # combined pipeline (does not run — see Known Issues)
├── test_integrated_pipeline.py
├── example_usage.py
├── requirements.txt
├── models/
│   ├── yolo_detector.py               # YOLOv8 with contour fallback
│   ├── aruco_calibration.py           # marker detection → mm/px
│   ├── dimension_detector.py          # contour → dimensions
│   ├── vgg16_wrapper.py               # Keras defect classifier
│   ├── defect_detector.py             # PyTorch classifier (unused)
│   └── blip_wrapper.py                # BLIP captioner
└── utils/
    └── image_processor.py
```

## Running the measurement app

Must be run **from this directory** — it resolves `yolov8n.pt` relative to the
working directory.

```bash
cd backend
set PYTHONIOENCODING=utf-8      # Windows: console messages contain emoji
python industrial_measurement.py
```

It prompts on stdin for marker size and ArUco dictionary, then opens a webcam
window. Hold a printed marker in view to calibrate; objects placed beside it are
measured.

| Key | Action |
|---|---|
| `q` | quit |
| `s` | save screenshot |
| `c` | toggle contrast enhancement |
| `r` | reset calibration and clear locks |
| `u` | unlock locked objects |
| `f` | freeze frame |

Requires a webcam and a display.

## Technical notes

**Calibration.** `DICT_6X6_250` by default; marker size selectable from
47/50/40/30 mm at startup. The px→mm ratio is the median of the marker's four
side lengths. Measured error against a synthetic ground-truth marker: 0.14%.

**Camera intrinsics.** `CAMERA_MATRIX` and `DIST_COEFFS` near the top of
`industrial_measurement.py` are **placeholder values** (fx=fy=800, cx=640,
cy=360, zero distortion). Pose estimation and distance readings are approximate
until you substitute a real calibration for your camera.

**Object detection.** Adaptive Gaussian threshold plus a bilateral filter and
morphological open/close, restricted to an ROI positioned 1.2× marker-width to the
right of the marker and sized 6× the marker. YOLO is used inside the ROI when
available.

**Measurement.** `minAreaRect` for rotation-invariant width/height, with
shape-specific routines for circles, triangles and quadrilaterals, tilt correction
from the marker's pose, and a weighted moving average over recent frames.

Reported per object: width, height, area, perimeter, rotation angle, aspect ratio.

## Integrated pipeline

```bash
cd backend
python test_integrated_pipeline.py 1
```

This currently fails at import. The required fixes are listed in
[../docs/KNOWN_ISSUES.md](../docs/KNOWN_ISSUES.md).
