"""
Defect Detector Module for ZENITH INSPECT-AR

Loads the trained Faster R-CNN damage detector and runs inference on frames.

The checkpoint (`best_model.pth`) is a torchvision Faster R-CNN with a
ResNet50-FPN backbone and an 8-class head, trained by
`defects_3dataset/train_defect_detection.py`. It is a *detector*: it localises
damage and labels it in a single pass, so it does not need YOLO to hand it
boxes first.

Earlier versions of this file rebuilt a ResNet18 classifier instead, which
matched none of the checkpoint's 295 tensors and silently fell back to random
weights. Loading here is strict: a mismatch raises rather than degrading to
noise.
"""

import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops.misc import FrozenBatchNorm2d


# --------------------------------------------------
# Class map
# --------------------------------------------------
# The training script concatenated two Roboflow COCO exports and used their raw
# `category_id` values as labels, with no remapping. Their id spaces overlap, so
# class 1 is a genuine mixture of two source categories:
#
#   dataset v2 "stock-defect-class"   -> id 1 = "defect"  (1448 boxes)
#   dataset v4 "5-classes"            -> id 1 = "crack"   ( 996 boxes)
#
# Classes 6 and 7 exist in the head but were never trained: no annotation in
# either dataset carries those ids. Predictions for them are meaningless.
#
# Verified against data/*/train/_annotations.coco.json on 2026-09-04.
NUM_CLASSES = 8

CLASS_NAMES = {
    0: "background",
    1: "crack_or_defect",   # see collision note above
    2: "dent",
    3: "missing-head",
    4: "paint-off",
    5: "scratch",
    6: "untrained_6",
    7: "untrained_7",
}

# Classes the checkpoint never saw a single training box for.
UNTRAINED_CLASS_IDS = frozenset({6, 7})

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "best_model.pth"


# --------------------------------------------------
# Model construction
# --------------------------------------------------
def _freeze_batchnorm(module: torch.nn.Module) -> torch.nn.Module:
    """
    Replace every BatchNorm2d with FrozenBatchNorm2d, in place.

    torchvision picks the normalisation layer from its weight arguments:
    `fasterrcnn_resnet50_fpn` uses FrozenBatchNorm2d when either `weights` or
    `weights_backbone` is set, and plain BatchNorm2d when both are None. The
    training script passed weights="DEFAULT", so the checkpoint is a
    FrozenBatchNorm model and carries no `num_batches_tracked` buffers.

    Building with both set to None (to avoid a ~100 MB download we would
    immediately overwrite) would therefore give a subtly different module tree.
    load_state_dict would not catch it: BatchNorm's loader silently defaults a
    missing `num_batches_tracked` to 0 instead of reporting a missing key. The
    two are numerically identical under eval(), but a BatchNorm model put into
    train() would start updating its running statistics and corrupt the
    weights. Converting here keeps the architecture faithful.
    """
    for name, child in module.named_children():
        if isinstance(child, torch.nn.BatchNorm2d):
            frozen = FrozenBatchNorm2d(child.num_features, eps=child.eps)
            frozen.weight.data = child.weight.data.clone()
            frozen.bias.data = child.bias.data.clone()
            frozen.running_mean.data = child.running_mean.data.clone()
            frozen.running_var.data = child.running_var.data.clone()
            setattr(module, name, frozen)
        else:
            _freeze_batchnorm(child)
    return module


def _head_width(state_dict, default: int = NUM_CLASSES) -> int:
    """Read the classifier head's width straight from the saved weights.

    Sizing the model from the file rather than from a constant means a
    checkpoint with a different number of classes loads instead of raising a
    shape mismatch.
    """
    weight = state_dict.get("roi_heads.box_predictor.cls_score.weight")
    return int(weight.shape[0]) if weight is not None else default


def build_defect_model(num_classes: int = NUM_CLASSES) -> torch.nn.Module:
    """
    Rebuild the exact architecture used during training.

    Must stay in step with defects_3dataset/train_defect_detection.py:get_model,
    which swaps the stock 91-class COCO box predictor for an 8-class one.
    """
    # Both weight arguments are None: every tensor is about to be overwritten
    # from the checkpoint, so downloading pretrained weights first would be
    # wasted bandwidth. See _freeze_batchnorm for why the conversion follows.
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    _freeze_batchnorm(model)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# --------------------------------------------------
# Defect Detector Class
# --------------------------------------------------
class DefectDetector:
    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        num_classes: int = NUM_CLASSES,
        device: Optional[str] = None,
        score_threshold: float = 0.5,
    ):
        """
        Args:
            checkpoint_path: path to best_model.pth. Defaults to the copy that
                sits beside this file.
            num_classes: size of the detection head. Must match the checkpoint.
            device: "cpu" or "cuda". Auto-detected when omitted.
            score_threshold: minimum confidence for a box to be reported.

        Raises:
            FileNotFoundError: the checkpoint is not on disk.
            RuntimeError: the checkpoint does not fit the architecture.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.score_threshold = score_threshold

        checkpoint_path = Path(checkpoint_path or DEFAULT_CHECKPOINT)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Defect checkpoint not found: {checkpoint_path}\n"
                "It is gitignored (*.pth). Train one with "
                "backend/training/train_detector.py, or copy your own."
            )

        # Read the checkpoint BEFORE building the model, so the head is sized
        # from what the file actually contains rather than from a constant.
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            self.epoch = checkpoint.get("epoch")
            self.f1 = checkpoint.get("f1")
        else:
            state_dict = checkpoint
            self.epoch = None
            self.f1 = None

        # Checkpoints from train_detector.py carry their own class_names, so
        # nothing has to guess what index 3 means. The original 8-class
        # checkpoint does not, and falls back to the legacy colliding map.
        saved_names = checkpoint.get("class_names") if isinstance(checkpoint, dict) else None
        if saved_names:
            self.class_names = {i: n for i, n in enumerate(saved_names)}
            num_classes = checkpoint.get("num_classes", len(saved_names))
            self.untrained_ids = frozenset()
            self.self_describing = True
        else:
            self.class_names = dict(CLASS_NAMES)
            num_classes = _head_width(state_dict, num_classes)
            self.untrained_ids = UNTRAINED_CLASS_IDS
            self.self_describing = False

        self.model = build_defect_model(num_classes)

        # strict=True on purpose. A silent partial load is what made every
        # previous prediction noise; a mismatch must be loud.
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

        print(f"[ok] Defect detector loaded: {checkpoint_path.name}")
        print(f"     Architecture: Faster R-CNN ResNet50-FPN, {num_classes} classes")
        if self.self_describing:
            # Written by train_detector.py, whose f1 compares labels as well as
            # boxes, so the number means what it appears to mean.
            if self.epoch is not None:
                print(f"     Trained to epoch {self.epoch}, class-aware f1 {self.f1:.4f}")
            print(f"     Classes: {', '.join(list(self.class_names.values())[1:])}")
        else:
            # Legacy checkpoint. Its f1 was computed class-agnostically -- boxes
            # matched by IoU, labels never compared -- so it measures
            # localisation only, and two of its classes were never trained.
            if self.epoch is not None:
                print(f"     Trained to epoch {self.epoch}, localisation f1 {self.f1:.4f}")
            print("     [warn] Legacy label space: class 1 merges 'crack' and "
                  "'defect', and classes 6-7 were never trained.")
            print("     [warn] Retrain with backend/training/train_detector.py "
                  "for trustworthy labels.")
        print(f"     Device: {self.device}")

    # --------------------------------------------------
    # Whole-frame detection (the model's native mode)
    # --------------------------------------------------
    def detect_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Find damage anywhere in the frame.

        This is what the model was trained to do. It needs no prior object
        detector.

        Args:
            frame: BGR image, as returned by cv2.

        Returns:
            A list of detections, each:
                {id, class, class_id, confidence, bbox: [x1,y1,x2,y2],
                 area, untrained_class}
            sorted by confidence, highest first.
        """
        if frame is None or frame.size == 0:
            return []

        tensor = self._preprocess_frame(frame)

        with torch.no_grad():
            prediction = self.model([tensor])[0]

        boxes = prediction["boxes"].cpu().numpy()
        scores = prediction["scores"].cpu().numpy()
        labels = prediction["labels"].cpu().numpy()

        results = []
        for idx, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            if score < self.score_threshold:
                continue

            x1, y1, x2, y2 = (float(v) for v in box)
            class_id = int(label)

            results.append({
                "id": idx,
                "class": self.class_names.get(class_id, f"class_{class_id}"),
                "class_id": class_id,
                "confidence": float(score),
                "bbox": [x1, y1, x2, y2],
                "area": (x2 - x1) * (y2 - y1),
                # Flag rather than drop: the caller decides what to do with a
                # prediction from a head that saw no training data.
                "untrained_class": class_id in self.untrained_ids,
            })

        results.sort(key=lambda d: d["confidence"], reverse=True)
        return results

    # --------------------------------------------------
    # ROI-scoped detection (pipeline compatibility)
    # --------------------------------------------------
    def detect(self, frame: np.ndarray, detections: List[Dict]) -> Dict:
        """
        Report damage within regions another detector already found.

        Kept so the existing pipeline, which passes YOLO boxes, still works.
        Prefer detect_frame() — running the detector on the whole frame is both
        faster and more accurate than cropping first, because cropping discards
        the surrounding context the model was trained on.

        Args:
            frame: BGR image
            detections: upstream detections, each with a "bbox" and an "id"

        Returns:
            {"objects": {obj_id: {predicted_class, class_id, confidence,
                                  has_defect, bbox, untrained_class}}}
        """
        results: Dict[str, Dict] = {"objects": {}}

        if frame is None or frame.size == 0:
            return results

        height, width = frame.shape[:2]

        for det in detections:
            obj_id = det["id"]
            x1, y1, x2, y2 = (int(v) for v in det["bbox"])

            # Clamp to the frame; an out-of-bounds slice yields an empty array.
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            found = self.detect_frame(roi)
            if not found:
                continue

            best = found[0]
            results["objects"][obj_id] = {
                "predicted_class": best["class"],
                "class_id": best["class_id"],
                "confidence": best["confidence"],
                # Every trained class is a kind of damage; there is no "normal"
                # class in either source dataset. A detection at all means
                # damage was found.
                "has_defect": True,
                # Translate the ROI-local box back into frame coordinates.
                "bbox": [
                    best["bbox"][0] + x1, best["bbox"][1] + y1,
                    best["bbox"][2] + x1, best["bbox"][3] + y1,
                ],
                "untrained_class": best["untrained_class"],
            }

        return results

    # --------------------------------------------------
    # Preprocessing
    # --------------------------------------------------
    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        BGR uint8 image -> CHW float tensor in [0, 1].

        Matches DefectDataset.__getitem__ in the training script: RGB, scaled by
        1/255, no resize and no mean/std normalisation (Faster R-CNN does its
        own resizing and normalisation internally).
        """
        # A 2-D frame reaches here when ImageProcessor runs with
        # enhance_contrast=False; promote it rather than letting cvtColor raise.
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = rgb.astype(np.float32) / 255.0
        chw = np.transpose(rgb, (2, 0, 1))

        return torch.from_numpy(chw).to(self.device)
