# Known Issues

Every item here was reproduced by running the code, not inferred by reading it.
Line numbers refer to the current tree.

**Blockers** stop execution or make output meaningless. **Major** issues produce
silently wrong results.

---

## Blockers

### 1. The integrated pipeline cannot be imported
`backend/integrated_inspection_pipeline.py:18` — `import industrial_measurement as im`

`industrial_measurement.py` has no `if __name__ == "__main__":` guard. At module
scope it calls `input()` twice (lines 35, 45), opens `cv2.VideoCapture(0)`, and
enters an infinite render loop. Importing the pipeline therefore starts the whole
standalone webcam app; under a non-interactive stdin it dies with `EOFError`.

The alias `im` is referenced **zero** times.

**Fix:** delete line 18. Separately, wrap `industrial_measurement.py`'s body in a
`main()` behind a `__main__` guard so it can be imported safely at all.

---

### 2. Defect inference crashes on a PyTorch idiom
`backend/models/vgg16_wrapper.py:112` — `with tf.no_grad():`

`tf.no_grad()` does not exist; that is PyTorch. TensorFlow 2.17.1 raises
`AttributeError: module 'tensorflow' has no attribute 'no_grad'`. The call sits in
`process_frame` with no try/except, so it takes the whole pipeline down.

**Fix:** delete the context manager. Keras inference with `training=False` already
builds no gradient tape.

---

### 3. No trained defect classifier exists in this repository
`notebooks/Defect-Detection-of-Aircraft.ipynb`

The notebook trains a Keras VGG16 and then never calls `model.save()` or
`save_weights()`. No `.keras`, `.h5`, or `.weights.h5` file exists anywhere in the
tree or in git history.

**Fix:** add `model.save("backend/models/defect_vgg16.keras")` after `model.fit`.

---

### 4. `best_model.pth` is the wrong kind of model entirely
`backend/models/best_model.pth` (165 MB)

Inspecting the checkpoint: 295 state-dict tensors under `backbone.body.*`,
`rpn.*`, `roi_heads.*`; `roi_heads.box_predictor.cls_score` is `(8, 1024)`;
metadata `epoch=23`, `f1=0.642`. It is a **torchvision Faster R-CNN
ResNet50-FPN object detector with 8 classes** — not a VGG16, not a ResNet18, and
not a binary classifier. It matches neither notebook.

Both loaders fail and both swallow the failure:

- `vgg16_wrapper.py:42` — Keras 3 rejects the `.pth` outright, then
  `except` → *"Using randomly initialized model"*.
- `defect_detector.py:79` — `load_state_dict` raises on key mismatch. Measured
  overlap between checkpoint keys and model keys: **0 of 295**.

Inference therefore runs on **random weights**. Observed confidences of 0.51,
0.57, 0.61 are coin flips.

**Fix:** either load the Faster R-CNN with its real architecture, or produce a
genuine classifier checkpoint per issue #3.

---

### 5. Uncalibrated measurements are reported as successful
`backend/models/aruco_calibration.py:96`

When no marker is found, `detect_and_calibrate()` returns
`pixel_to_mm_ratio = 1.0` instead of signalling failure. The pipeline's
`calibrate()` only checks that the key exists, so it prints
*"Calibration successful"* and every subsequent measurement is a **pixel count
labelled "mm"**.

**Fix:** propagate the `success` flag and return `False`.

---

### 6. Dimension measurement measures the background
`backend/models/dimension_detector.py:40`

Uses `cv2.THRESH_BINARY` (non-inverted), which selects **bright** pixels. For a
dark object on a light background — the common case for aircraft skin — the
largest contour is the background, not the object.

Measured against a synthetic 300×160 px object at 0.5 mm/px (truth 150.0 × 80.0 mm):

| Scenario | Result | Error |
|---|---|---|
| Dark object, tight bbox | 0.0 × 0.0 mm | silent total failure |
| Dark object, padded bbox | 189.5 × 119.5 mm | 26% / 49% |
| Bright object, padded bbox | 150.0 × 80.0 mm | correct |

**Fix:** use `THRESH_BINARY_INV`, or better, reuse the adaptive-threshold routine
already written in `industrial_measurement.py:detect_object_in_roi`, which handles
this correctly.

---

### 7. Dimensions can never reach the display
`backend/integrated_inspection_pipeline.py:188`

`DimensionDetector.measure()` returns `{'objects': {...}, 'overall_confidence': …}`
but the pipeline assigns that whole envelope to `results["dimensions"]`. The
renderer then tests `obj_id in results["dimensions"]`, where `obj_id` is an int and
the only keys are `"objects"` and `"overall_confidence"`. Always false.

Compounding it, `draw_results` (line 273) reads `width`/`height`/`area` while the
producer emits `width_mm`/`height_mm`/`area_mm2`.

**Fix:** assign `dimension_results.get("objects", {})` and align the key names.

---

## Major

### 8. Object detection has no aircraft-damage classes
`backend/models/yolo_detector.py:28` — `YOLO('yolov8n.pt')`

Stock COCO weights. On real dataset images it reports classes like `clock` and
`surfboard`, and finds nothing at all on half the images tested — which
short-circuits `process_frame` at line 162 and returns an empty result.

Because every later stage crops to a YOLO box, a wrong box means the classifier
and captioner analyse the wrong pixels. This is the project's central
limitation and no amount of plumbing repair fixes it.

**Fix:** train YOLOv8 on annotated damage bounding boxes.

---

### 9. Class labels contradict what the model was trained on
`backend/models/vgg16_wrapper.py:34` — `class_names = ["normal", "defect"]`

The notebook uses `flow_from_directory(class_mode='binary')` on `crack/` and
`dent/`, giving `crack=0`, `dent=1`. **Both classes are defects**; there is no
"normal" class in the dataset. So `has_defect = class_id == 1` really means
"is a dent", and a crack is reported as *not a defect*.

---

### 10. Preprocessing destroys colour before the models see it
`backend/utils/image_processor.py:40`

`preprocess()` converts BGR → grayscale, applies CLAHE, then converts back to
3-channel BGR. The result has `R == G == B` at every pixel. YOLO, VGG16 and BLIP
all receive a colour-free image; the classifier was trained on full-colour RGB.

With `enhance_contrast=False` (line 44) it returns a **2-D array**, which breaks
the downstream `cv2.cvtColor(..., COLOR_BGR2RGB)` calls outright.

---

### 11. Hardcoded paths are wrong from the documented working directory
`integrated_inspection_pipeline.py:46`, `test_integrated_pipeline.py:50,112`

The docs say `cd backend && python test_integrated_pipeline.py`, but the code uses
`Path("backend/models")`, `Path("aircraft_damage_dataset_v1/test")` and
`Path("backend/test_output")` — all resolving relative to the repo root. From
`backend/` they become `backend/backend/models` etc.

Consequences: the model folder is missing (silent fallback to random weights), the
dataset is not found (the test benchmarks a **random-noise image** and then prints
`PERFORMANCE OK`), and the output directory raises `FileNotFoundError`.

`example_usage.py:26` has the same bug in the opposite direction.

**Fix:** anchor every path to `Path(__file__).resolve().parent`.

---

### 12. Emoji in `print()` crash on Windows
Throughout `backend/`

Under the default cp1252 console codepage, printing `✅`/`⚠️`/`📌` raises
`UnicodeEncodeError`. This bites hardest at `vgg16_wrapper.py:45`, where the emoji
is *inside the exception handler* — so the handler crashes while reporting the
original error.

**Fix:** set `PYTHONIOENCODING=utf-8`, or drop the emoji.

---

### 13. `defect_detector.py` is dead code
Never imported anywhere. It duplicates `vgg16_wrapper.py` in PyTorch and, if
revived, would also run on random weights (see #4).

---

### 14. `test_integrated_pipeline.py` does not test what it claims
It constructs the pipeline with `use_dimensions=True` and draws with
`show_dimensions=True`, but never calls `calibrate()`. Step 4 is skipped on every
frame, so the "dimension" test exercises no dimension code.

---

## Not defects (checked and cleared)

- The ArUco API is the **modern** OpenCV 4.7+ one (`ArucoDetector`) and works
  correctly on 4.11.
- `pixel_to_mm_ratio` is consistently mm-per-pixel in both implementations; the
  arithmetic in `dimension_detector.measure()` is dimensionally correct.
- No secrets, API keys or credentials appear anywhere in git history.
