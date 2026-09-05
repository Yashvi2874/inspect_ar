# INSPECT-AR — AI-Assisted Aircraft Damage Inspection

Computer-vision prototype for aircraft surface inspection: measure the physical
dimensions of a region of interest using an ArUco reference marker, and
classify/caption aircraft damage from photographs.

> **Status: work-in-progress prototype.** The dimension-measurement app works.
> The defect-classification and integrated-pipeline paths do **not** currently run
> end to end. See [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md) before relying on
> any output — it lists every known defect with file and line references.

---

## What actually works today

| Capability | State | Notes |
|---|---|---|
| ArUco calibration (px → mm) | **Working** | 0.14% error against a synthetic ground-truth marker |
| Standalone dimension app | **Working** | `backend/industrial_measurement.py`, needs a webcam + printed marker |
| YOLO object detection | **Runs, wrong domain** | Stock COCO weights — no aircraft-damage classes |
| VGG16 defect classification | **Broken** | Crashes on `tf.no_grad()`; no trained weights exist in the repo |
| BLIP captioning | **Runs, not fine-tuned** | Generic pretrained captions, ~1 GB download on first use |
| Integrated pipeline | **Broken** | Cannot be imported; see Known Issues #1 |

---

## Repository layout

```
inspect_ar/
├── README.md
├── LICENSE
├── requirements.txt
├── backend/
│   ├── industrial_measurement.py       # standalone ArUco measurement app (works)
│   ├── integrated_inspection_pipeline.py
│   ├── test_integrated_pipeline.py
│   ├── example_usage.py
│   ├── models/
│   │   ├── yolo_detector.py            # YOLOv8 + contour fallback
│   │   ├── aruco_calibration.py        # marker detection → mm/px ratio
│   │   ├── dimension_detector.py       # contour → dimensions
│   │   ├── vgg16_wrapper.py            # Keras defect classifier
│   │   ├── defect_detector.py          # PyTorch classifier (unused)
│   │   └── blip_wrapper.py             # BLIP captioner
│   └── utils/image_processor.py
├── notebooks/
│   ├── Defect-Detection-of-Aircraft.ipynb
│   ├── Captioning-of-Aircraft-Images.ipynb
│   └── executed_Defect-Detection-of-Aircraft.ipynb
├── docs/
│   ├── ARCHITECTURE.md
│   └── KNOWN_ISSUES.md
└── aircraft_damage_dataset_v1/         # not in git — see .gitkeep
```

---

## Setup

```bash
git clone https://github.com/Yashvi2874/inspect_ar.git
cd inspect_ar

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

pip install -r requirements.txt
```

`opencv-contrib-python` is required — the ArUco module is not in the base
`opencv-python` package.

On Windows, run with UTF-8 output or the emoji in the console messages will
raise `UnicodeEncodeError` under the default cp1252 codepage:

```bash
set PYTHONIOENCODING=utf-8
```

### Model weights

Nothing large is committed to git.

- **YOLOv8n** — downloaded automatically by `ultralytics` on first run.
- **BLIP** — downloaded automatically from Hugging Face on first run (~1 GB).
- **`backend/models/best_model.pth`** — a Faster R-CNN checkpoint that is **not**
  loadable by either wrapper in this repo. See Known Issues #4.

### Dataset

Binary `dent` / `crack` image folders, ~446 images:

```
aircraft_damage_dataset_v1/{train,valid,test}/{dent,crack}/
```

Downloaded by the first cells of
[notebooks/Defect-Detection-of-Aircraft.ipynb](notebooks/Defect-Detection-of-Aircraft.ipynb).

---

## Running

### Dimension measurement (the working path)

```bash
cd backend
python industrial_measurement.py
```

Answer the two console prompts (marker size, ArUco dictionary), hold a printed
marker in view to calibrate, then measure objects beside it.

| Key | Action |
|---|---|
| `q` | quit |
| `s` | save screenshot |
| `c` | toggle contrast enhancement |
| `r` | reset calibration |
| `u` | unlock locked objects |
| `f` | freeze frame |

Requires a webcam and a display — it is an interactive OpenCV GUI app, not a library.

### Integrated pipeline

```bash
cd backend
python test_integrated_pipeline.py 1
```

**This does not currently work.** It fails at import. The fixes required are
enumerated in [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md).

### Training the defect classifier

```bash
jupyter notebook notebooks/Defect-Detection-of-Aircraft.ipynb
```

Transfer-learns a frozen ImageNet VGG16 with a small dense head, 5 epochs,
binary `crack` vs `dent`. **The notebook does not save the model** — add a
`model.save()` call at the end if you want a reusable checkpoint.

---

## Measured results

From the executed notebook and from instrumented runs — not estimates.

| Metric | Value | Source |
|---|---|---|
| Defect classifier, test accuracy | **0.6875** | `executed_…ipynb`, 50-image test set |
| Defect classifier, best val accuracy | 0.72 | same, epoch 1 of 5 |
| Defect classifier, final train accuracy | 0.88 | same, epoch 5 |
| ArUco calibration error | 0.14% | synthetic 100 px = 50 mm marker |
| Pipeline throughput (all stages, CPU) | 0.13–0.24 FPS | 640×640 images, no GPU |
| Dataset size | 300 train / 96 valid / 50 test | `flow_from_directory` output |

The train/test gap (0.88 vs 0.69) indicates overfitting on a small dataset.

---

## Roadmap

The prototype's central limitation is that **no model in this repo is trained to
find aircraft damage**. Object detection uses stock COCO weights. In priority order:

- [ ] Fix the blocking defects in [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md)
- [ ] Train a real damage detector (YOLOv8 on annotated damage boxes)
- [ ] Persist the classifier from the notebook and load it correctly
- [ ] Expand the dataset well beyond 446 images
- [ ] Fine-tune BLIP on damage descriptions
- [ ] Damage severity scoring
- [ ] Web UI / REST API

---

## Acknowledgements

YOLO — [Ultralytics](https://github.com/ultralytics/ultralytics) ·
VGG16 — [Keras Applications](https://keras.io/api/applications/) ·
BLIP — [Salesforce Research](https://github.com/salesforce/BLIP) ·
ArUco — [OpenCV](https://docs.opencv.org/) ·
Dataset — IBM / Roboflow aircraft damage set

## License

MIT — see [LICENSE](LICENSE).
