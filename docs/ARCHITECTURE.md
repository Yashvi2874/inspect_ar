# Architecture

INSPECT-AR is two independently-developed capabilities plus an attempt to merge
them. Understanding that split explains most of the codebase.

---

## Track A — Dimension measurement (mature)

A single-file interactive application: `backend/industrial_measurement.py` (~1550 lines).

```
webcam frame
   → ArUco marker detection            (cv2.aruco.ArucoDetector)
   → pose estimation + tilt angles     (solvePnP against CAMERA_MATRIX)
   → pixel→mm ratio from marker edges  (median of 4 side lengths)
   → ROI defined relative to marker
   → object detection inside ROI       (YOLO, or adaptive-threshold contours)
   → shape-specific geometry           (circle / triangle / quad / rotated rect)
   → tilt correction
   → temporal smoothing + object lock  (weighted moving average over N frames)
   → overlay render
```

Notable: it handles rotated objects via `minAreaRect`, corrects for marker tilt,
and locks a measurement once it has been stable for a number of frames.

This track owns its own copies of detection and calibration logic. It does **not**
import the `backend/models/*` modules other than `YOLODetector`.

---

## Track B — Defect classification and captioning (notebook-stage)

```
notebooks/Defect-Detection-of-Aircraft.ipynb
   ImageDataGenerator(rescale=1/255)
   → frozen ImageNet VGG16 (include_top=False)
   → Flatten → Dense(512) → Dropout(.3) → Dense(512) → Dropout(.3) → Dense(1, sigmoid)
   → binary crossentropy, Adam(1e-4), 5 epochs
   → test accuracy 0.6875          ← never saved to disk

notebooks/Captioning-of-Aircraft-Images.ipynb
   → stock Salesforce/blip-image-captioning-base, two hardcoded prompts
   → nothing trained, nothing saved
```

---

## Track C — The integration

`backend/integrated_inspection_pipeline.py` orchestrates thin re-implementations
of both tracks:

```
frame
 → ImageProcessor.preprocess()                    utils/image_processor.py
 → YOLODetector.detect()                          models/yolo_detector.py
 → for each detection:
      VGG16DefectDetector.detect()                models/vgg16_wrapper.py
      BLIPCaptioner.process_detections()          models/blip_wrapper.py
      DimensionDetector.measure()                 models/dimension_detector.py
 → merged results dict keyed by YOLO integer id
 → draw_results() / print_results()
```

Calibration is a separate, manual step: `pipeline.calibrate(frame)` must be
called with an ArUco marker visible before `dimensions` is populated at all.

### Where the seams are

The `models/*.py` modules are **not** the Track A code — they are simplified
rewrites. `dimension_detector.py` reimplements object segmentation in 8 lines
where `industrial_measurement.py` uses a tuned adaptive-threshold and morphology
chain, which is why the simplified version fails on dark-on-light scenes
(Known Issues #6).

Similarly `vgg16_wrapper.py` reconstructs the notebook's architecture from scratch
rather than loading a saved model — necessary, because the notebook never saved one.

The result dict is the other seam: producers and consumers disagree on both
nesting depth and key names (Known Issues #7).

---

## Data flow contract

`process_frame()` returns:

```python
{
    "timestamp": float,
    "frame_shape": tuple,
    "detections": [ {id, class, class_id, confidence, bbox, area}, ... ],
    "defects":    { obj_id: {predicted_class, class_id, confidence, has_defect} },
    "captions":   { obj_id: {caption, bbox} },
    "dimensions": { obj_id: {width_mm, height_mm, area_mm2, perimeter_mm, ...} },
    "processing_time": float,
}
```

`dimensions` is documented here as it is *intended* to be. The current code
assigns the producer's outer envelope instead, so in practice the key holds
`{"objects": {...}, "overall_confidence": …}` and never renders.

---

## Dependencies between modules

```
integrated_inspection_pipeline
├── models.yolo_detector        → ultralytics
├── models.vgg16_wrapper        → tensorflow / keras
├── models.blip_wrapper         → transformers, torch
├── models.dimension_detector   → opencv
├── models.aruco_calibration    → opencv-contrib
├── utils.image_processor       → opencv
└── industrial_measurement      ← should not be here; see Known Issues #1

models.defect_detector          → torch, torchvision   (orphaned, nothing imports it)
```
