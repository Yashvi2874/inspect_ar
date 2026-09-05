# Architecture

INSPECT-AR grew from two independently-built capabilities that were later
joined. Understanding that split explains most of the codebase.

---

## Track A — Dimension measurement

A single-file interactive application: `backend/industrial_measurement.py`
(~1,550 lines). This is the original hackathon demo and the most mature code in
the project.

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

Two things are worth separating here, because they have very different
reliability:

- The **pixel-to-mm ratio** comes only from the marker's side lengths in the
  image. It never uses the camera intrinsics. This is why its measured error is
  0.14%.
- The **distance and tilt** readings come from `solvePnP`, which does use the
  intrinsics — and those are placeholder values (fx = fy = 800). Treat them as
  approximate.

This track owns its own copies of detection and calibration logic. It does not
import `backend/models/*` other than `YOLODetector`.

---

## Track B — Damage detection

Two detectors that answer different questions, and are useful together.

### Supervised — `models/defect_detector.py`

A torchvision Faster R-CNN with a ResNet50-FPN backbone and an 8-class head,
trained on two concatenated Roboflow COCO exports (5,672 images, 6,219 boxes).

It localises and labels in a single pass, so it needs no upstream object
detector — which is why the pipeline runs it directly on the frame rather than
behind YOLO.

Loading is strict (`strict=True`). An earlier version rebuilt a ResNet18
classifier instead, matched none of the checkpoint's 295 tensors, swallowed the
failure, and ran inference on random weights. Strict loading makes that class of
error impossible to miss.

One subtlety: torchvision picks the normalisation layer from its weight
arguments — `FrozenBatchNorm2d` when weights are requested, plain `BatchNorm2d`
when they are not. The checkpoint was trained with the former. Building with
both weight arguments `None` (to avoid a needless 100 MB download) therefore
produces a subtly different module tree, and `load_state_dict` will *not* catch
it, because BatchNorm silently defaults a missing `num_batches_tracked` to 0.
`_freeze_batchnorm()` converts them explicitly.

### Unsupervised — `models/anomaly_autoencoder.py`

A convolutional autoencoder trained only on defect-free aircraft skin. It
reconstructs normal surface texture well and damage badly, so per-pixel
reconstruction error doubles as a defect map.

```
frame
  → tile into overlapping 64×64 patches
  → encode → 128×8×8 bottleneck → decode
  → error = α·MSE + (1-α)·(1 - SSIM)      per pixel
  → average overlapping tiles back into one full-resolution map
  → normalise (99th percentile) → threshold → contour → regions
                                 → JET colormap → heatmap overlay
```

Both error measures are used because they fail differently: MSE reacts to
intensity change (a bright scratch on dark metal), SSIM to structural change (a
dent that deforms a reflection but barely shifts pixel values).

The bottleneck is deliberately narrow — 8,192 values against 12,288 input
values. Too wide and the network learns to copy its input, reconstructing
defects perfectly and detecting nothing.

**Training data** is the interesting part. No dataset here has a healthy class.
But defect boxes cover only ~8.4% of each labelled image, so
`backend/training/train_autoencoder.py` harvests patches that overlap no defect
box, with a 16 px safety margin because labels are drawn tight and cracks run
past them. Contaminated training data would teach the autoencoder to
reconstruct damage, and it would then report none.

---

## Track C — Integration

`backend/integrated_inspection_pipeline.py` orchestrates the stages:

```
frame
 → ImageProcessor.preprocess()                   utils/image_processor.py
 → DefectDetector.detect_frame()                 models/defect_detector.py
      (or YOLODetector.detect(), optional)
 → for each detection:
      DimensionDetector.measure()                models/dimension_detector.py
      BLIPCaptioner.process_detections()         models/blip_wrapper.py
 → merged results dict keyed by detection id
 → draw_results() / print_results()
```

Calibration is a separate, manual step: `pipeline.calibrate(frame)` must be
called with an ArUco marker visible before `dimensions` is populated at all. It
returns `False` when no marker is found, and `pixel_to_mm_ratio` stays `None` —
so an uncalibrated measurement is absent rather than wrong.

The detector runs on the **original** frame, not the preprocessed one, because
`ImageProcessor.preprocess` collapses the image to grayscale and back to three
identical channels, and the detector was trained on colour.

### Defaults

| Stage | Default | Why |
|---|---|---|
| `use_defect_detector` | **on** | The only stage trained on aircraft damage |
| `use_yolo` | off | Stock COCO weights, no damage classes |
| `use_vgg16` | off | No trained Keras checkpoint exists |
| `use_blip` | on, but guarded | ~1 GB download; segfaults in some environments |
| `use_dimensions` | on | Cheap, and inert until calibrated |

`models.vgg16_wrapper` is imported lazily inside `__init__` rather than at
module scope, because importing it pulls in the whole TensorFlow runtime, and
TensorFlow and PyTorch in one process segfault on some setups.

---

## Track D — HTTP API

`backend/app.py` exposes the pipeline over HTTP. It is built on
`IntegratedInspectionPipeline` rather than on individually wired models, so it
inherits the pipeline's fixes instead of repeating its bugs.

```
POST /api/analyze   →  calibrate() → process_frame() → JSON (+ optional annotated JPEG)
POST /api/anomaly   →  AnomalyDetector.detect()      → JSON (+ optional heatmap JPEG)
POST /api/calibrate →  ArucoCalibrator.calibrate()   → px-to-mm ratio
GET  /api/health    →  which checkpoints are loadable
GET  /api/models    →  which checkpoints exist on disk
```

Models load lazily on first use, so importing the module and answering
`/api/health` never waits on a 165 MB checkpoint.

Uploads are decoded in memory (`cv2.imdecode`) rather than written to disk. The
original hackathon API saved every upload permanently, which grew without bound
and let two uploads of the same filename overwrite each other.

---

## Data flow contract

`process_frame()` returns:

```python
{
    "timestamp": float,
    "frame_shape": tuple,
    "detections": [ {id, class, class_id, confidence, bbox, area, untrained_class}, ... ],
    "defects":    { obj_id: {predicted_class, class_id, confidence, has_defect, untrained_class} },
    "captions":   { obj_id: {caption, bbox} },
    "dimensions": { obj_id: {width_mm, height_mm, area_mm2, perimeter_mm, ...} },
    "dimension_confidence": float | None,
    "processing_time": float,
}
```

`dimensions` is keyed by the same integer ids as `detections`. An earlier
version assigned the producer's outer envelope (`{"objects": ..., "overall_confidence": ...}`)
to this key, so every `obj_id in results["dimensions"]` test compared an int
against the strings `"objects"` and `"overall_confidence"` and was always false.
Measured dimensions could never reach the display.

---

## Module dependencies

```
integrated_inspection_pipeline
├── models.defect_detector       → torch, torchvision
├── models.yolo_detector         → ultralytics          (optional)
├── models.vgg16_wrapper         → tensorflow / keras   (optional, lazy import)
├── models.blip_wrapper          → transformers, torch  (optional, guarded)
├── models.dimension_detector    → opencv
├── models.aruco_calibration     → opencv-contrib
└── utils.image_processor        → opencv

app                              → flask, flask-cors
├── integrated_inspection_pipeline
└── models.anomaly_autoencoder   → torch

training.train_autoencoder       → torch, opencv
└── models.anomaly_autoencoder

industrial_measurement           → opencv-contrib, ultralytics
                                   (standalone; not imported by anything)
```
