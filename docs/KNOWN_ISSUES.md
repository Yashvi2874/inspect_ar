# Known Issues

Every item here was reproduced by running the code, not inferred by reading it.

The original audit of this project found 14 defects. Twelve have since been
fixed;
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

**Fix — written and tested, needs a GPU run.**
`backend/training/train_detector.py` retrains with both causes addressed:
every dataset declares an explicit mapping into one shared 7-class label space
(so v2's "defect" is its own class, not merged into "crack"), the head is sized
from that space so no class goes untrained, metrics are class-aware, per-class
precision and recall are printed each epoch, and `class_names` is saved into
the checkpoint so no future loader has to guess what the indices mean.

Verified on the real annotations: the remap yields defect 1448 | crack 996 |
dent 1346 | missing-head 990 | paint-off 789 | scratch 222, with no collision.
The metric was checked against perfectly-located boxes carrying deliberately
wrong labels — the old class-agnostic measure scores 0.500 there, the new
class-aware one scores 0.000.

Only the training run itself remains; it is impractical on CPU.

---

### 2. Preprocessing discards colour
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

### 3. Pose estimation uses placeholder camera intrinsics
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

### 4. The anomaly autoencoder over-fires on texture
`backend/models/anomaly_autoencoder.py`

It reliably scores damage above clean skin (1.70× on validation, 8 of 10
images), but it also fires on rough or weathered surface, and returns more
regions than there are defects.

**Fix:** raise the threshold, train longer on more patches, or require a
minimum region area. It is a screening tool, not a localiser.

---

### 5. BLIP captions are generic, and it is memory-hungry
`backend/models/blip_wrapper.py`

Two separate things, previously conflated in this document.

**Not a defect:** BLIP loads and runs correctly. An earlier version of these
notes claimed it "segfaults on load". That was wrong — the crashes were caused
by other jobs on the machine consuming RAM at the time, not by BLIP. Verified
by loading it alongside the Faster R-CNN detector: both fit.

**Real constraint:** materialising its weights needs roughly 2 GB of free RAM,
and when that is unavailable the process dies with a native allocation failure
that no `try`/`except` can catch. `BLIPCaptioner` therefore checks available
memory before loading and stands down with a message instead. The pipeline
drops the captioner entirely when it did not load, so the captioning step is
skipped rather than writing "Model not loaded" against every detection.

**Real limitation:** the captions are generic. This is stock
`blip-image-captioning-base` with no fine-tuning on damage, so it describes the
photograph ("a picture of a camera lens") rather than the defect. Fine-tuning
it is on the roadmap.

---

### 6. No trained Keras classifier exists
`backend/models/vgg16_wrapper.py`

The VGG16 path builds its architecture correctly but has no weights to load: no
`.keras`, `.h5` or `.weights.h5` file exists anywhere in the project or its git
history. Predictions from it would be from a randomly initialised head, so the
wrapper now says so loudly and the pipeline leaves it off by default.

---

### 7. YOLO ships stock COCO weights
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
| 11 | Dimension measurement was polarity sensitive — `THRESH_BINARY` always keeps the brighter group, so a dark part on a light panel measured the panel (26% / 49% error) | Polarity is now inferred by comparing the ROI border with its centre, then `THRESH_BINARY_INV` is used when the object is the darker one. All four dark/bright x tight/padded cases now measure 149.5 x 79.5 mm against a 150 x 80 truth |
| 12 | `industrial_measurement.py` had no `__main__` guard — importing it prompted on stdin and opened a webcam, so it could not be reused or tested | Prompts moved into `configure()`, the application body into `main()`, behind a guard. Module-level defaults let it import with stdin closed; the five smoothing globals `apply_advanced_smoothing()` reads stay at module scope |

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
