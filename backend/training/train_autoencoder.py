"""
Train the unsupervised anomaly autoencoder on DEFECT-FREE aircraft skin.

Where defect-free data comes from
---------------------------------
No dataset in this project has a "healthy" class — every image is of damage.
But the v2 Roboflow export annotates damage as boxes on 640x640 photographs of
aircraft skin, and those boxes cover only about 8.4% of the pixel area. The
other ~91.6% IS defect-free aircraft skin; it is simply not labelled as
anything.

So clean training data is harvested rather than downloaded: slide a window over
each image and keep only the patches that overlap NO annotated defect box. A
safety margin is added around every box, because Roboflow boxes are drawn tight
and a crack often continues a few pixels past its label.

This matters more than it sounds. If defective pixels leak into training, the
autoencoder learns to reconstruct damage, and at inference it will happily
rebuild a crack and report no anomaly.

Usage
-----
    cd inspect_ar
    set PYTHONIOENCODING=utf-8
    python backend/training/train_autoencoder.py --epochs 15

Writes backend/models/anomaly_autoencoder.pth.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from models.anomaly_autoencoder import (  # noqa: E402
    PATCH_SIZE,
    ConvAutoencoder,
    reconstruction_loss,
    ssim_map,
)

# The annotated-detection dataset lives outside the repo and outside version
# control. Override with --data if yours sits elsewhere.
DEFAULT_DATA = (
    REPO_ROOT.parent
    / "defects_3dataset" / "data"
    / "aircraft-skin-defects-new-dataset.v2-size-640-stock-defect-class.coco"
)

OUTPUT = BACKEND_DIR / "models" / "anomaly_autoencoder.pth"


# --------------------------------------------------------------------------
def harvest_clean_patches(split_dir, patch=PATCH_SIZE, stride=32, margin=16, limit=None):
    """
    Collect patches that overlap no annotated defect, with a safety margin.

    Args:
        split_dir: a Roboflow COCO split holding _annotations.coco.json.
        patch: square patch side in pixels.
        stride: step between patch origins. Smaller yields more, more correlated.
        margin: pixels to grow every defect box by before testing overlap.
        limit: stop after this many patches (keeps memory bounded).

    Returns:
        uint8 array (N, patch, patch, 3) in RGB.
    """
    split_dir = Path(split_dir)
    ann_file = split_dir / "_annotations.coco.json"
    if not ann_file.is_file():
        raise FileNotFoundError(f"No COCO annotations at {ann_file}")

    data = json.loads(ann_file.read_text(encoding="utf-8"))

    boxes_by_image = {}
    for a in data["annotations"]:
        x, y, w, h = a["bbox"]
        boxes_by_image.setdefault(a["image_id"], []).append(
            (x - margin, y - margin, x + w + margin, y + h + margin)
        )

    patches = []
    skipped = 0
    for info in data["images"]:
        # Absolute path: these dataset paths run ~253 characters, and resolving
        # them against a deeper working directory breaks Windows' 260-char
        # MAX_PATH limit, which surfaces as a silent imread -> None.
        image = cv2.imread(str((split_dir / info["file_name"]).resolve()))
        if image is None:
            skipped += 1
            continue

        boxes = boxes_by_image.get(info["id"], [])
        height, width = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for y in range(0, height - patch + 1, stride):
            for x in range(0, width - patch + 1, stride):
                px1, py1, px2, py2 = x, y, x + patch, y + patch
                # Reject on ANY intersection, not on centre-in-box: a patch
                # holding a defect corner is still contaminated.
                if any(px1 < bx2 and px2 > bx1 and py1 < by2 and py2 > by1
                       for bx1, by1, bx2, by2 in boxes):
                    continue
                # .copy() is load-bearing: a numpy slice is a VIEW, so
                # storing one keeps the whole 1.2 MB parent image alive and
                # the harvest exhausts memory after a few hundred images.
                patches.append(rgb[py1:py2, px1:px2].copy())

                if limit and len(patches) >= limit:
                    print(f"  reached limit of {limit} patches")
                    return np.stack(patches)

        del image, rgb

    if skipped:
        print(f"  warning: {skipped} images unreadable (check MAX_PATH)")
    if not patches:
        raise RuntimeError("No clean patches found — is the margin too large?")
    return np.stack(patches)


# --------------------------------------------------------------------------
class PatchDataset(Dataset):
    """Clean patches, scaled to [0,1], with flip/rotate augmentation."""

    def __init__(self, patches, augment=True):
        self.patches = patches
        self.augment = augment

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        p = self.patches[idx]
        if self.augment:
            # Aircraft skin has no canonical orientation, so these are all
            # equally valid views of normal surface. Cheap extra data.
            if random.random() < 0.5:
                p = np.fliplr(p)
            if random.random() < 0.5:
                p = np.flipud(p)
            k = random.randint(0, 3)
            if k:
                p = np.rot90(p, k)
        p = np.ascontiguousarray(p, dtype=np.float32) / 255.0
        return torch.from_numpy(p).permute(2, 0, 1)


# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--limit", type=int, default=4000,
                        help="max clean patches to harvest (memory bound)")
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--threads", type=int, default=0,
                        help="torch CPU threads; 0 leaves the default")
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Harvesting defect-free patches from {args.data.name} ...")
    train_patches = harvest_clean_patches(
        args.data / "train", stride=args.stride, limit=args.limit)
    print(f"  train: {len(train_patches)} clean patches of {PATCH_SIZE}x{PATCH_SIZE}")

    val_patches = harvest_clean_patches(
        args.data / "valid", stride=args.stride, limit=max(256, args.limit // 8))
    print(f"  valid: {len(val_patches)} clean patches")

    train_loader = DataLoader(PatchDataset(train_patches), batch_size=args.batch_size,
                              shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(PatchDataset(val_patches, augment=False),
                            batch_size=args.batch_size, num_workers=0)

    model = ConvAutoencoder().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=2, factor=0.5)

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = n = 0
        for batch in train_loader:
            batch = batch.to(device)
            loss = reconstruction_loss(batch, model(batch))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            running += loss.item() * batch.size(0)
            n += batch.size(0)
        train_loss = running / max(n, 1)

        # Validation doubles as calibration: the mean error on CLEAN patches is
        # the reference level the detector reports anomalies relative to.
        model.eval()
        running = n = 0
        errors = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                running += reconstruction_loss(batch, out).item() * batch.size(0)
                n += batch.size(0)
                sq = ((batch - out) ** 2).mean(dim=1)
                dis = (1.0 - ssim_map(batch, out)).mean(dim=1)
                errors.append((0.5 * sq + 0.5 * dis).mean().item())
        val_loss = running / max(n, 1)
        normal_error = float(np.mean(errors)) if errors else 0.0

        scheduler.step(val_loss)
        flag = ""
        if val_loss < best:
            best = val_loss
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "normal_error": normal_error,
                "patch_size": PATCH_SIZE,
            }, args.out)
            flag = "  <- saved"
        print(f"epoch {epoch:2d}/{args.epochs}  train {train_loss:.5f}  "
              f"val {val_loss:.5f}  clean-error {normal_error:.5f}{flag}")

    print(f"\nDone. Best val loss {best:.5f}. Wrote {args.out}")


if __name__ == "__main__":
    main()
