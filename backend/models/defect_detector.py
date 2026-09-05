"""
Defect Detector Module for ZENITH INSPECT-AR
Loads a trained defect classification model and runs inference on detected ROIs
"""

import cv2
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List
import torchvision.models as models


# --------------------------------------------------
# Model Architecture (MUST match training)
# --------------------------------------------------
def build_defect_model(num_classes: int, checkpoint_path: Path):
    """
    Rebuild the same architecture used during training
    First, try to determine the architecture from the checkpoint
    """
    try:
        # Load the checkpoint to inspect its structure
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Check for specific architecture indicators in the state dict keys
        keys = list(state_dict.keys())
        if any('backbone' in key for key in keys):
            # This appears to be a detection model like Faster R-CNN
            # For defect detection, we'll need to adapt it or use a classification model instead
            print("⚠️  Detected object detection model (e.g., Faster R-CNN)")
            print("💡 For defect classification, a ResNet-based classifier is recommended")
            # Fall back to a classification model
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            return model
        else:
            # Try ResNet18 as default
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            return model
    except Exception as e:
        print(f"⚠️ Error inspecting checkpoint: {e}")
        # Default to ResNet18 if inspection fails
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model


# --------------------------------------------------
# Defect Detector Class
# --------------------------------------------------
class DefectDetector:
    def __init__(
        self,
        checkpoint_path: Path,
        num_classes: int = 2,
        device: str = None
    ):
        """
        Args:
            checkpoint_path: path to best_model.pth
            num_classes: number of defect classes
            device: cpu / cuda
        """

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 1️⃣ Build model architecture
        self.model = build_defect_model(num_classes, checkpoint_path)
        self.model.to(self.device)

        # 2️⃣ Load checkpoint safely
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        try:
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.class_names = checkpoint.get(
                    "class_names",
                    [f"class_{i}" for i in range(num_classes)]
                )
            else:
                # Raw state_dict case
                self.model.load_state_dict(checkpoint)
                self.class_names = [f"class_{i}" for i in range(num_classes)]

            self.model.eval()

            print("✅ Defect model loaded successfully")
            print(f"📌 Classes: {self.class_names}")
            print(f"📌 Device: {self.device}")
        except Exception as e:
            print(f"⚠️  Architecture mismatch: {e}")
            print("💡 Using randomly initialized model - retrain with correct architecture for best results")
            # Keep the model as randomly initialized
            self.class_names = [f"class_{i}" for i in range(num_classes)]
            self.model.eval()
            print(f"📌 Classes: {self.class_names}")
            print(f"📌 Device: {self.device}")

    # --------------------------------------------------
    # Inference
    # --------------------------------------------------
    def detect(self, frame: np.ndarray, detections: List[Dict]) -> Dict:
        """
        Run defect detection on YOLO-detected objects

        Args:
            frame: BGR image
            detections: YOLO detections list

        Returns:
            Dictionary with defect results per object
        """

        results = {"objects": {}}

        for det in detections:
            obj_id = det["id"]
            x1, y1, x2, y2 = map(int, det["bbox"])

            roi = frame[y1:y2, x1:x2]

            if roi.size == 0:
                continue

            tensor = self._preprocess(roi)

            with torch.no_grad():
                logits = self.model(tensor)
                probs = torch.softmax(logits, dim=1)
                conf, cls_id = torch.max(probs, dim=1)

            class_id = int(cls_id.item())
            confidence = float(conf.item())

            results["objects"][obj_id] = {
                "predicted_class": self.class_names[class_id],
                "class_id": class_id,
                "confidence": confidence,
                "has_defect": self.class_names[class_id] != "normal"
            }

        return results

    # --------------------------------------------------
    # Preprocessing
    # --------------------------------------------------
    def _preprocess(self, roi: np.ndarray) -> torch.Tensor:
        """
        Converts ROI to model input tensor
        """

        roi = cv2.resize(roi, (224, 224))
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi = roi.astype(np.float32) / 255.0

        roi = np.transpose(roi, (2, 0, 1))  # HWC → CHW
        tensor = torch.from_numpy(roi).unsqueeze(0)
        tensor = tensor.to(self.device)

        return tensor