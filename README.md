<div align="center">

# INSPECT-AR

**From Manual Inspection to AI Inspector**

Computer-vision inspection for aircraft surfaces - measure a part in millimetres
from a single camera, and find damage on it.

🥉 **Third Prize - HAL AEROTHON '26**
National-Level Aerospace Innovation Hackathon
Hindustan Aeronautics Limited × IIIT Dharwad · 9-10 January 2026

<img src="docs/images/win-team.jpg" alt="Team Zenith with the Third Prize cheque at IIIT Dharwad" width="330">
<img src="docs/images/getting.jpeg" alt="Third Prize being presented at HAL AEROTHON '26" width="430">

</div>

---

## What this is

Point a camera at an aircraft panel with a **printed ArUco marker** beside it -
a square black-and-white pattern, like a QR code, whose real-world size you
already know. In the frame below the marker is 47 mm wide and occupies about
172 pixels, so the software derives a scale of 0.2672 millimetres per pixel and
can then measure anything else in view without touching it.

On top of that scale sit two independent damage detectors, which answer
different questions:

- A **supervised** detector, trained on labelled aircraft damage, that names
  what it finds - crack, dent, missing head, paint-off, scratch.
- An **unsupervised** autoencoder, trained only on undamaged skin, that flags
  anything unlike normal surface - including damage nobody ever labelled.

Built in 24 hours at HAL AEROTHON '26 by Team Zenith, and developed since.

![INSPECT-AR measuring a gear and running damage detection](docs/images/integration.jpeg)

*The live application, uncropped. The printed ArUco marker sets the scale at
0.2672 mm/px against a 47 mm reference, the gear is measured at 51.1 x 32.4 mm
without contact, and the damage detector overlays its findings on the same
frame. Detector class indices are shown raw here - see [Results](#results) for
why those labels are not yet trustworthy.*

---

## Status of each component

Measured on this machine, not estimated. Every number below is reproducible
with the commands in [Running](#running).

| Component | State | Notes |
|---|---|---|
| ArUco calibration (px → mm) | **Working** | 0.14% error against a synthetic ground-truth marker |
| Standalone measurement app | **Working** | Real-time overlay; needs a webcam and a printed marker |
| Faster R-CNN damage detector | **Working** | Real trained weights; localises well, class labels are unreliable - see [Results](#results) |
| Anomaly autoencoder + heatmap | **Working** | Unsupervised; defect regions score 1.7× clean skin |
| Integrated pipeline | **Working** | ~1.8-2.0 s per 640×640 frame, CPU-only |
| HTTP API | **Working** | 5 endpoints, `backend/app.py` |
| Dimension measurement | **Working** | Polarity-aware thresholding: 0.3% / 0.6% error on both dark-on-light and bright-on-dark |
| BLIP captioning | **Working** | Needs ~2 GB free RAM; skips itself cleanly below that. Captions are generic - the model is not fine-tuned on damage |
| VGG16 classifier | **Legacy, off by default** | No trained Keras checkpoint exists anywhere in the project |
| YOLOv8 | **Off by default** | Ships stock COCO weights, which have no damage classes |

---

## Architecture

```mermaid
graph TD
    A[Camera frame or image file] --> B[ArUco marker detection]
    B --> C["Pixel-to-mm ratio<br/>(median of 4 marker side lengths)"]
    A --> D[Faster R-CNN ResNet50-FPN<br/>supervised damage detection]
    A --> E[Conv autoencoder<br/>unsupervised anomaly detection]
    D --> F[Contour dimensioning per region]
    C --> F
    E --> G[Reconstruction-error heatmap]
    F --> H[Annotated overlay + JSON]
    G --> H
```

The scale in **C** is derived purely from the marker's side lengths in the
image. It never needs the camera's intrinsic parameters, which is why it is the
most reliable number this project produces.

The standalone app additionally applies a tilt correction derived from
`solvePnP` pose estimation. That path *does* use camera intrinsics, and those
are **placeholder values** (fx = fy = 800), so distance and orientation readings
are approximate until you substitute a real calibration for your camera.

---

## Results

### ArUco calibration

| Measurement | Value |
|---|---|
| Calibration error | **0.14%** (synthetic 100 px marker declared 50 mm) |
| Dimension accuracy | **149.5 × 79.5 mm** against a 150 × 80 mm truth (0.3% / 0.6%), on dark-on-light and bright-on-dark alike |

### Supervised detector

Faster R-CNN ResNet50-FPN, 8-class head, trained for 23 epochs on two
concatenated Roboflow exports - 5,672 images, 6,219 boxes.

**Read this before trusting its labels.** The checkpoint records `f1 = 0.642`,
but that figure was computed *class-agnostically* - the training script matched
boxes by IoU and never compared predicted labels to ground-truth labels.
Re-running that same evaluation code against the saved weights does not
reproduce it. Two further issues are real and worth stating:

- **Class 1 is a collision.** The two source datasets were concatenated without
  remapping category IDs, so class 1 mixes one dataset's generic "defect" with
  the other's "crack". It is reported as `crack_or_defect` for that reason.
- **Classes 6 and 7 were never trained** - no annotation in either dataset
  carries those IDs - yet they still fire, sometimes above 0.9 confidence. They
  are labelled `untrained_6` / `untrained_7` and flagged in the output.

In practice it is a strong *region proposer* on aircraft skin and a weak
*classifier*. It also boxes rivets and fasteners as damage.

**The cause is fixed, the retrain is not.**
`backend/training/train_detector.py` maps every dataset into one shared 7-class
space so the generic "defect" stays separate from "crack", sizes the head from
that space so no class goes untrained, scores class-aware metrics, and saves
`class_names` into the checkpoint. Verified against the real annotations -
defect 1448, crack 996, dent 1346, missing-head 990, paint-off 789,
scratch 222, no collision. Only the training run remains, and it needs a GPU;
until then the checkpoint above is what ships.

### Unsupervised anomaly detection

No dataset here has a "healthy" class - every image is of damage. So clean
training data is **harvested**: defect boxes cover only ~8.4% of each labelled
image, and the training script keeps only patches overlapping no defect box
(plus a 16 px safety margin, because labels are drawn tight and cracks run past
them).

Trained for 20 epochs on 2,000 harvested clean patches; best validation loss
0.02245, which is also the reference error the detector scores anomalies
against.

| Metric | Value |
|---|---|
| Defect-region score ÷ clean-skin score | **1.70×** |
| Validation images where defects scored higher | **8 of 10** |
| Strongest single image | **3.96×** |
| Mean score inside labelled defects | 0.03970 |
| Mean score outside them | 0.02816 |

![Anomaly heatmap: input, reconstruction-error heatmap, and detected regions](docs/images/anomaly-heatmap.png)

*Left: input. Middle: per-pixel reconstruction error - the crack is the hot red
line. Right: green is the human label, magenta is what the autoencoder found
without ever being shown a defect.*

It over-fires on rough or weathered texture and returns more regions than there
are defects. Treat it as a screening tool that says "look here", not a precise
localiser.

### Throughput

~1.8-2.0 s per 640×640 frame (≈0.5 FPS) with detection and dimensioning
enabled, on a CPU-only laptop. There is no GPU measurement in this repository.

---

## Setup

```bash
git clone https://github.com/Yashvi2874/inspect_ar.git
cd inspect_ar

python -m venv venv
venv\Scripts\activate           # Windows
# source venv/bin/activate      # Linux / macOS

pip install -r requirements.txt
```

`opencv-contrib-python` is required - `cv2.aruco` is not in the base
`opencv-python` package.

### Windows console encoding

```bash
set PYTHONIOENCODING=utf-8      # without it, console output can die on encoding
```

Optionally, `set INSPECT_AR_NO_BLIP=1` skips captioning entirely. It is not
required - BLIP checks available memory and stands itself down when there is
too little, rather than crashing.

### Model weights

**No weights are distributed with this repository.** They exceed GitHub's file
size limit and are gitignored.

| File | Size | How to obtain |
|---|---|---|
| `backend/models/best_model.pth` | 165 MB | `python backend/training/train_detector.py` (needs a GPU), or copy your own |
| `backend/models/anomaly_autoencoder.pth` | 1 MB | `python backend/training/train_autoencoder.py` |
| `yolov8n.pt` | 6 MB | Downloaded automatically by `ultralytics` |

A fresh clone therefore **cannot run the detector** until you supply or train a
checkpoint. `GET /api/models` reports which are present.

---

## Running

### Standalone measurement app (the original demo)

```bash
cd backend
python industrial_measurement.py
```

Answer the two prompts, hold a printed marker in view to calibrate, then place
the object beside it.

| Key | Action |
|---|---|
| `q` | quit |
| `s` | save screenshot |
| `c` | toggle contrast enhancement |
| `r` | reset calibration |
| `u` | unlock locked objects |
| `f` | freeze frame |

Needs a webcam and a display for the interactive window. The module is also
importable: it sits behind a `__main__` guard, so its measurement helpers can be
reused without launching the application.

```python
from industrial_measurement import compute_pixel_to_mm_ratio
ratio, _ = compute_pixel_to_mm_ratio(marker_corners, known_size_mm=47.0)
```

### Integrated pipeline

```bash
cd backend
python test_integrated_pipeline.py 1
```

Processes sample images and writes annotated results to `backend/test_output/`.

### HTTP API

```bash
python backend/app.py           # http://127.0.0.1:5000
```

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/health` | GET | Service status and which checkpoints are loadable |
| `/api/models` | GET | Which checkpoints exist on disk |
| `/api/calibrate` | POST | Image → pixel-to-mm ratio from an ArUco marker |
| `/api/analyze` | POST | Image → damage detections, plus dimensions when calibrated |
| `/api/anomaly` | POST | Image → unsupervised anomaly regions and heatmap |

All uploads use the multipart field name `file`.

```bash
curl -F "file=@panel.jpg" -F "heatmap=1" http://127.0.0.1:5000/api/anomaly
```

The server binds to loopback only. Set `INSPECT_AR_HOST=0.0.0.0` to expose it,
and do not combine that with `INSPECT_AR_DEBUG=1` - the Werkzeug debugger
executes arbitrary code from the browser.

### Training the autoencoder

```bash
python backend/training/train_autoencoder.py --epochs 20
```

Harvests defect-free patches automatically and writes
`backend/models/anomaly_autoencoder.pth`.

---

## Known limitations

Found by auditing our own code after the hackathon. Full detail with file and
line references in [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md).

- The shipped detector's class labels are unreliable: two of its eight classes
  were never trained yet still fire, and class 1 merges two different source
  categories. `backend/training/train_detector.py` fixes both causes; the
  replacement checkpoint has not been trained yet, so the old one still ships.
- `scratch` has only 222 training boxes against `dent`'s 1,346. Any retrain
  should expect weak recall on it until that imbalance is addressed.
- Preprocessing converts to grayscale and back, discarding colour for every
  stage except the detector, which runs on the original frame.
- Pose-derived distance and tilt use placeholder camera intrinsics. The
  pixel-to-mm scale does not, which is why 0.14% still holds.
- BLIP captions are generic. It is stock `blip-image-captioning-base`, never
  fine-tuned on damage, so it describes the picture rather than the defect.
- The anomaly autoencoder over-fires on rough texture.

---

## Roadmap

- [ ] Train and ship the remapped detector with
      `backend/training/train_detector.py`. The collision fix, the class-aware
      metric and the 7-class label space are written and verified; only the
      run remains. Measured at 7.5 min/epoch on an RTX 4070 Laptop at batch 6
      and 640px, so a full 25 epochs is about three hours - not the cloud job
      this once assumed.
- [ ] Address the class imbalance before trusting per-class numbers - weighted
      sampling, or more `scratch` data
- [ ] Fine-tune BLIP on damage descriptions so captions are about the defect
- [ ] OCR for part traceability - read serial numbers and stencilled IDs so
      every detection is logged against the component it belongs to
- [ ] UAV-mounted capture
- [ ] Explore non-destructive testing modalities for subsurface inspection -
      ultrasonic, eddy-current, thermography, borescope

---

## Team

**Team Zenith** - KJ Somaiya College of Engineering

| Member | GitHub |
|---|---|
| Yashasvi Gupta | [@Yashvi2874](https://github.com/Yashvi2874) |
| Aastha Shah | [@aasthans-2508](https://github.com/aasthans-2508) |
| Sai Parcha | [@SaiParcha](https://github.com/SaiParcha) |

Mentors: **Dr. Shailesh Nikam** and **Mr. Kaustubh Kulkarni**, for guidance on
system architecture, backend structuring, ArUco integration and model
optimisation.

Thanks to **HAL** and **IIIT Dharwad** for the platform.

---

## Acknowledgements

- **YOLOv8** - [Ultralytics](https://github.com/ultralytics/ultralytics).
  Licensed **AGPL-3.0**, which is more restrictive than this project's MIT
  licence; if you redistribute a work that imports it, that licence governs.
- **Faster R-CNN, ResNet50-FPN** - [torchvision](https://github.com/pytorch/vision)
- **BLIP** - [Salesforce Research](https://github.com/salesforce/BLIP)
- **ArUco** - [OpenCV](https://docs.opencv.org/)
- **Damage datasets** - Roboflow Universe aircraft-skin-defect exports,
  published under **CC BY 4.0**, which requires attribution.
- **Training notebooks** - the VGG16 and BLIP notebooks this project's early
  wrappers were derived from originate as **IBM Skills Network** guided-lab
  material, obtained via
  [asitdave/Aircraft-Defect-Detection-and-Automated-Image-Captioning](https://github.com/asitdave/Aircraft-Defect-Detection-and-Automated-Image-Captioning).
  That repository publishes no licence, so those notebooks are **not
  redistributed here**. `backend/models/vgg16_wrapper.py` and `blip_wrapper.py`
  note their origin in their headers.

---

## License

**MIT** - see [LICENSE](LICENSE). This covers the original code of this project
(`backend/` and `docs/`).

It does **not** cover third-party components, which carry their own terms:
the datasets (CC BY 4.0), Ultralytics YOLOv8 (AGPL-3.0), and the training
notebooks described above.
