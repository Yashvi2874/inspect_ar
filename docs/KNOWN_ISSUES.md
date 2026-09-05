# Known Issues

Every item here was reproduced by running the code, not inferred by reading it.

The original audit of this project found 14 defects. Ten have since been fixed;
they are kept below under [Fixed](#fixed) rather than deleted, because the
history of what was wrong is part of what makes the current numbers credible.

**Open** issues still affect output today. **Fixed** issues are resolved in the
current tree.

---

## Open

### 1. The detector's class labels are unreliable
`backend/models/defect_detector.py`

The checkpoint is a genuine trained Faster R-CNN, and it localises damage on
aircraft skin well. Its *labels* are a different matter, for two reasons that
both originate in the training script:

- **Class 1 is a collision.** Two Roboflow COCO exports were concatenated with
  their raw `category_id` values used directly as labels, with no remapping.
  One dataset's id 1 means "defect" (1,448 boxes); the other's means "crack"
  (996 boxes). They are indistinguishable in the trained model, which is why
  the class is reported as `crack_or_defect`.
- **Classes 6 and 7 were never trained.** No annotation in either dataset
  carries those ids, yet the model still emits them, sometimes above 0.9
  confidence. They are surfaced as `untrained_6` / `untrained_7` with an
  `untrained_class: true` flag so a caller can discard them.

The `f1 = 0.642` stored in the checkpoint was computed **class-agnostically** —
`calculate_metrics` matched boxes by IoU and never compared labels. Re-running
that same evaluation code against the saved weights does not reproduce it.

It also boxes rivets and fasteners as damage.

**Fix:** retrain with category ids remapped to a single consistent space, and a
metric that compares labels.

---

### 2. Dimension measurement is polarity sensitive
`backend/models/dimension_detector.py:40`

Uses `cv2.THRESH_BINARY | cv2.THRESH_OTSU`, which selects **bright** pixels. For
a dark object on a light background the largest contour can be the background.

Measured against a synthetic 300×160 px object at 0.5 mm/px (truth 150.0 × 80.0 mm):

| Scenario | Result | Error |
|---|---|---|
| Bright object, padded bbox | 150.0 × 80.0 mm | correct |
| Dark object, tight bbox | 149.5 × 79.5 mm | 0.3% / 0.6% |
| Dark object, padded bbox | 189.5 × 119.5 mm | 26% / 49% |

**Fix:** use `THRESH_BINARY_INV`, or reuse the adaptive-threshold routine in
`industrial_measurement.py:detect_object_in_roi`, which handles both polarities.

---

### 3. Preprocessing discards colour
`backend/utils/image_processor.py:40`

`preprocess()` converts BGR → grayscale, applies CLAHE, then converts back to
3-channel BGR, leaving `R == G == B` at every pixel. With
`enhance_contrast=False` (line 44) it returns a **2-D array**, which makes
`cv2.cvtColor(..., COLOR_BGR2GRAY)` raise in `dimension_detector.py:37` — where
a blanket `except` swallows it into all-zero measurements.

The Faster R-CNN detector is unaffected: the pipeline deliberately runs it on
the original colour frame.

**Fix:** keep colour, or apply CLAHE to the L channel of LAB.

---

### 4. `industrial_measurement.py` cannot be imported
`backend/industrial_measurement.py`

No `if __name__ == "__main__":` guard across ~1,550 lines. At module scope it
calls `input()` twice (lines 35, 45), opens `cv2.VideoCapture(0)` (line 967) and
enters an infinite render loop (line 1012). Importing it therefore launches the
whole webcam application, or dies with `EOFError` under non-interactive stdin.

The pipeline no longer imports it, so this blocks nothing today — but it makes
the file impossible to unit-test or reuse.

**Fix:** wrap the body in a `main()` behind a `__main__` guard.

---

### 5. Pose estimation uses placeholder camera intrinsics
`backend/industrial_measurement.py:58-65`

`CAMERA_MATRIX` is hard-coded to fx = fy = 800, cx = 640, cy = 360, with zero
distortion, and the file says so itself. Everything derived from `solvePnP` —
the distance readout and the pitch/yaw/roll used for tilt correction — is
therefore approximate.

This does **not** affect the headline accuracy figure. The pixel-to-mm ratio
comes from the marker's side lengths in the image and never touches the
intrinsics, which is why 0.14% holds.

**Fix:** run a real chessboard calibration for your camera and substitute the
result.

---

### 6. The anomaly autoencoder over-fires on texture
`backend/models/anomaly_autoencoder.py`

It reliably scores damage above clean skin (1.53× on validation, 8 of 10
images), but it also fires on rough or weathered surface, and returns more
regions than there are defects.

**Fix:** raise the threshold, train longer on more patches, or require a
minimum region area. It is a screening tool, not a localiser.

---

### 7. BLIP captioning segfaults on load
`backend/models/blip_wrapper.py`

Constructing `BLIPCaptioner` crashes the process natively — not a Python
exception, so it cannot be caught by the pipeline's `try/except`. The pipeline
defaults it off, and `test_integrated_pipeline.py` honours `INSPECT_AR_NO_BLIP=1`.

Also note it fetches ~1 GB from Hugging Face on first use, so it needs network.

---

### 8. No trained Keras classifier exists
`backend/models/vgg16_wrapper.py`

The VGG16 path builds its architecture correctly but has no weights to load: no
`.keras`, `.h5` or `.weights.h5` file exists anywhere in the project or its git
history. Predictions from it would be from a randomly initialised head, so the
wrapper now says so loudly and the pipeline leaves it off by default.

---

### 9. YOLO ships stock COCO weights
`backend/models/yolo_detector.py:28`

`YOLO('yolov8n.pt')` — no aircraft-damage classes. On real dataset images it
reports things like `clock` and `cat`, and finds nothing at all on most.
Disabled by default in the pipeline.

Also: `YOLODetector` accepts a `models_folder` argument and never uses it,
resolving `yolov8n.pt` against the current working directory instead.

---

## Fixed

These were real and are resolved in the current tree.

| # | Was | Fix |
|---|---|---|
| 1 | Pipeline could not be imported — `import industrial_measurement` ran a webcam app at import time; the alias was used zero times | Import deleted. `integrated_inspection_pipeline` now imports cleanly |
| 2 | `tf.no_grad()` in `vgg16_wrapper.py:112` — a PyTorch call in TensorFlow code, `AttributeError` on every inference | Removed; Keras builds no tape under `training=False` |
| 3 | `best_model.pth` loaded into the wrong architecture — 0 of 295 tensors matched, failure swallowed, **inference ran on random weights** | Rebuilt around `fasterrcnn_resnet50_fpn` with `strict=True`. Also converts BatchNorm → FrozenBatchNorm to match how the checkpoint was trained |
| 4 | Uncalibrated measurements reported as successful — a missing marker returned ratio `1.0`, so pixel counts were labelled "mm" | `detect_and_calibrate` returns a `success` flag and `None`; the pipeline tests the flag |
| 5 | Dimensions could never reach the display — the producer's envelope was assigned whole, so `obj_id in results["dimensions"]` was always false | Assigns `.get("objects", {})`; renderer reads `width_mm`/`height_mm` |
| 6 | Class labels contradicted the training data — `["normal", "defect"]` meant a crack was reported as *not* a defect | Corrected to `["crack", "dent"]`; both are damage, so `has_defect` is always true |
| 7 | Hardcoded paths broke from the documented working directory | All anchored to `Path(__file__).resolve().parent` |
| 8 | Emoji in `print()` raised `UnicodeEncodeError` under a cp1252 console — worst inside an exception handler, which crashed while reporting the original error | Replaced with ASCII markers |
| 9 | `test_integrated_pipeline.py` never called `calibrate()`, so the "dimension" test exercised no dimension code | Calibration added, with an explicit message when no marker is present |
| 10 | `example_usage.py` used `np` before importing it, and read `width`/`area` keys the producer never emitted | Import moved to module scope; keys corrected |

Also fixed, found during the same work:

- `blip_wrapper.py` imported TensorFlow and never used it, loading the entire TF
  runtime alongside PyTorch — a contributor to the segfaults. Removed.
- `test_integrated_pipeline.py` called `mkdir(exist_ok=True)` without
  `parents=True`, so a missing parent raised `FileNotFoundError`.
- The API's `cv2.imdecode` could raise rather than return `None`, turning a bad
  upload into an empty HTTP 500. Now a 400 with a message.
- `train_autoencoder.py` stored numpy **views** of harvested patches, keeping
  every 1.2 MB source image alive and exhausting memory. Now copies.

---

## Checked and cleared

- The ArUco API in use is the modern OpenCV 4.7+ `ArucoDetector`, correct on 4.11.
- `pixel_to_mm_ratio` is consistently millimetres-per-pixel in both
  implementations; the arithmetic in `dimension_detector.measure()` is
  dimensionally correct.
- The `/255.0` normalisation in `vgg16_wrapper` matches the notebook's training
  rescale.
- No secrets, API keys or credentials appear anywhere in git history.
