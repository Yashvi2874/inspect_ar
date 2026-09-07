"""
Retrain the Faster R-CNN damage detector, with the label bugs fixed.

Why this script exists
----------------------
The checkpoint currently shipped as best_model.pth was trained by a script that
concatenated two Roboflow COCO exports and used their raw `category_id` values
directly as labels, with no remapping. Their id spaces overlap:

    dataset v2  id 1 -> "defect"   (generic, 1448 boxes)
    dataset v4  id 1 -> "crack"    (996 boxes)

So the trained class 1 is a blend of two different things and cannot be
untangled after the fact. Classes 6 and 7 of the 8-class head were never given
a single training box, yet the model still emits them, sometimes above 0.9
confidence.

The reported f1 = 0.642 was also computed class-agnostically: predictions were
matched to ground truth by IoU alone, and the labels were never compared. It
measured localisation, not classification, which is why it looked healthy while
the labels were not.

What this script changes
------------------------
1. Every dataset declares an explicit mapping from its own category ids into
   ONE shared label space. Nothing is inferred from raw ids.
2. Metrics are class-aware: a prediction only counts as a true positive when it
   overlaps a ground-truth box AND names the same class. The class-agnostic
   figure is still reported beside it, so the two can be compared directly.
3. Per-class precision and recall are reported every epoch, so a class that is
   quietly failing is visible rather than hidden inside an average.
4. The head is sized from the label space, so there are no untrained classes.

Running it
----------
This is a GPU job. On CPU it is impractically slow -- roughly 5,600 images per
epoch through a ResNet50-FPN.

    python backend/training/train_detector.py --epochs 25 --batch-size 4

On Google Colab, mount the dataset and point --data at it. Check the device
line in the output says cuda before leaving it running.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

DEFAULT_DATA = REPO_ROOT.parent / "defects_3dataset" / "data"

# --------------------------------------------------------------------------
# The shared label space
# --------------------------------------------------------------------------
# Index 0 is background, which torchvision reserves. Every dataset below maps
# into these names and nothing else.
CLASS_NAMES = [
    "__background__",   # 0
    "crack",            # 1
    "dent",             # 2
    "missing-head",     # 3
    "paint-off",        # 4
    "scratch",          # 5
    "defect",           # 6  generic, from the single-class export
]
NUM_CLASSES = len(CLASS_NAMES)
NAME_TO_ID = {n: i for i, n in enumerate(CLASS_NAMES)}

# Per-dataset mappings: {directory name: {its category_id: shared name}}.
# "defect" from v2 is deliberately its OWN class, not merged into "crack" --
# merging them is precisely the bug this script exists to fix.
DATASET_MAPS = {
    "aircraft-skin-defects-new-dataset.v2-size-640-stock-defect-class.coco": {
        1: "defect",
    },
    "aircraft-skin-defects-new-dataset.v4-classification-isolated-5-classes-grayscale-prep.coco": {
        1: "crack",
        2: "dent",
        3: "missing-head",
        4: "paint-off",
        5: "scratch",
    },
}


# --------------------------------------------------------------------------
class DefectDataset(Dataset):
    """A Roboflow COCO split, with category ids remapped into the shared space."""

    def __init__(self, split_dir: Path, category_map: dict):
        self.split_dir = Path(split_dir)
        ann_file = self.split_dir / "_annotations.coco.json"
        if not ann_file.is_file():
            raise FileNotFoundError(f"No COCO annotations at {ann_file}")

        data = json.loads(ann_file.read_text(encoding="utf-8"))
        self.images = {i["id"]: i["file_name"] for i in data["images"]}

        self.targets = defaultdict(list)
        unmapped = set()
        for a in data["annotations"]:
            x, y, w, h = a["bbox"]
            if w <= 1 or h <= 1:
                continue
            name = category_map.get(a["category_id"])
            if name is None:
                # Loud, not silent. An unmapped id means the mapping above is
                # out of date with the dataset, which is how the original bug
                # happened in the first place.
                unmapped.add(a["category_id"])
                continue
            self.targets[a["image_id"]].append(
                ([x, y, x + w, y + h], NAME_TO_ID[name])
            )
        if unmapped:
            raise ValueError(
                f"{self.split_dir.parent.name}: category ids {sorted(unmapped)} "
                "have no entry in DATASET_MAPS. Add them before training."
            )

        # Only images that actually carry a box are useful to Faster R-CNN.
        self.ids = [i for i in self.images if self.targets.get(i)]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        # Absolute path: these dataset paths run ~253 characters and break
        # Windows' 260-char MAX_PATH when resolved against a deeper cwd.
        path = (self.split_dir / self.images[img_id]).resolve()
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            return None

        tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        boxes = [b for b, _ in self.targets[img_id]]
        labels = [l for _, l in self.targets[img_id]]
        return tensor, {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    return tuple(zip(*batch)) if batch else None


def build_splits(data_root: Path, split: str):
    sets = []
    for dirname, cmap in DATASET_MAPS.items():
        d = data_root / dirname / split
        if d.is_dir():
            sets.append(DefectDataset(d, cmap))
        else:
            print(f"  [warn] missing split: {d}")
    if not sets:
        raise SystemExit(f"No '{split}' data found under {data_root}")
    return ConcatDataset(sets)


# --------------------------------------------------------------------------
def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = ((a[2] - a[0]) * (a[3] - a[1]) +
             (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


def evaluate(preds, gts, iou_thr=0.5, score_thr=0.5):
    """
    Class-aware detection metrics, with the class-agnostic figure alongside.

    A true positive requires BOTH sufficient overlap and a matching label. The
    original script compared boxes only, which is why its f1 said nothing about
    whether the labels were right.
    """
    tp = fp = fn = 0
    tp_agnostic = 0
    per_class = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for p, t in zip(preds, gts):
        keep = p["scores"].cpu().numpy() > score_thr
        pb = p["boxes"].cpu().numpy()[keep]
        pl = p["labels"].cpu().numpy()[keep]
        tb = t["boxes"].cpu().numpy()
        tl = t["labels"].cpu().numpy()

        matched, matched_agnostic = set(), set()

        for box, label in zip(pb, pl):
            hit = hit_agnostic = False
            for i, (gt_box, gt_label) in enumerate(zip(tb, tl)):
                if iou(box, gt_box) < iou_thr:
                    continue
                if i not in matched_agnostic and not hit_agnostic:
                    matched_agnostic.add(i)
                    tp_agnostic += 1
                    hit_agnostic = True
                if i not in matched and label == gt_label:
                    matched.add(i)
                    tp += 1
                    per_class[int(label)]["tp"] += 1
                    hit = True
                    break
            if not hit:
                fp += 1
                per_class[int(label)]["fp"] += 1

        fn += len(tb) - len(matched)
        for i, gt_label in enumerate(tl):
            if i not in matched:
                per_class[int(gt_label)]["fn"] += 1

    def prf(tp_, fp_, fn_):
        pr = tp_ / (tp_ + fp_) if tp_ + fp_ else 0.0
        rc = tp_ / (tp_ + fn_) if tp_ + fn_ else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        return pr, rc, f1

    precision, recall, f1 = prf(tp, fp, fn)
    _, _, f1_agnostic = prf(tp_agnostic, fp, fn)
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "f1_agnostic": f1_agnostic,
        "tp": tp, "fp": fp, "fn": fn,
        "per_class": {CLASS_NAMES[k]: prf(v["tp"], v["fp"], v["fn"])
                      for k, v in sorted(per_class.items())},
    }


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", type=Path,
                    default=BACKEND_DIR / "models" / "detector_remapped.pth")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print("  [warn] No GPU. ~5,600 images per epoch through ResNet50-FPN on")
        print("         CPU will take many hours. Consider Colab.")

    print(f"Label space ({NUM_CLASSES} classes): {CLASS_NAMES}")

    train_ds = build_splits(args.data, "train")
    val_ds = build_splits(args.data, "valid")
    print(f"  train: {len(train_ds)} images   valid: {len(val_ds)} images")

    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=args.workers)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=args.workers)

    # COCO-pretrained weights, then a head sized to OUR label space.
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=8, gamma=0.1)

    best_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for batch in train_loader:
            if batch is None:
                continue
            images, targets = batch
            images = [i.to(device) for i in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss = sum(model(images, targets).values())
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += loss.item()
            seen += 1
        scheduler.step()

        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                images, targets = batch
                preds += model([i.to(device) for i in images])
                gts += targets

        m = evaluate(preds, gts)
        print(f"\nepoch {epoch:2d}/{args.epochs}  loss {total / max(seen,1):.4f}")
        print(f"  class-AWARE     P {m['precision']:.3f}  R {m['recall']:.3f}  F1 {m['f1']:.3f}")
        print(f"  class-agnostic  F1 {m['f1_agnostic']:.3f}   <- what the old script reported")
        for name, (pr, rc, f1) in m["per_class"].items():
            print(f"    {name:16s} P {pr:.3f}  R {rc:.3f}  F1 {f1:.3f}")

        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "f1": m["f1"],
                "f1_agnostic": m["f1_agnostic"],
                # Saved WITH the checkpoint, so no future loader has to guess
                # what the label indices mean. Their absence is what made the
                # original checkpoint so hard to interpret.
                "class_names": CLASS_NAMES,
                "num_classes": NUM_CLASSES,
            }, args.out)
            print(f"  saved -> {args.out.name}")

    print(f"\nDone. Best class-aware F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
