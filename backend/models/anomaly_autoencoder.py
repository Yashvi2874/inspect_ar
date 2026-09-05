"""
Unsupervised defect detection by reconstruction error, for ZENITH INSPECT-AR.

The idea in one paragraph
------------------------
A convolutional autoencoder is a network that squeezes an image through a narrow
bottleneck and then tries to rebuild it. Train it ONLY on defect-free aircraft
skin and it becomes good at reproducing normal surface texture — rivets, panel
lines, paint, glare — and nothing else. Show it a crack and it cannot rebuild
that crack, because it never learned what one looks like. The gap between input
and reconstruction is therefore large exactly where the surface is abnormal, so
that per-pixel gap doubles as a defect map.

This is unsupervised: no damage labels are used in training. That is the point.
It answers a different question from the supervised Faster R-CNN detector in
defect_detector.py, which can only find the five damage classes it was labelled
for. The autoencoder flags anything unlike normal skin, including damage types
nobody annotated.

Error signal
------------
Two measures are combined, because they fail in different ways:

- MSE (mean squared error) is per-pixel. It reacts strongly to a bright scratch
  on dark metal, and weakly to a shallow dent that barely shifts pixel values.
- SSIM (structural similarity) compares local mean, variance and covariance over
  a sliding window. It reacts to structural change — a dent that deforms a
  reflection — even when pixel values barely move.

Using only one of them misses half the defect population, so the score map is
`alpha * MSE + (1 - alpha) * (1 - SSIM)`.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "anomaly_autoencoder.pth"

# Patch size the model is trained and run at. Small on purpose: the network must
# learn local surface texture, not the global layout of a particular photograph.
PATCH_SIZE = 64


# --------------------------------------------------------------------------
# SSIM (structural similarity)
# --------------------------------------------------------------------------
def _gaussian_window(window_size: int, sigma: float, channels: int) -> torch.Tensor:
    """A normalised 2-D Gaussian kernel, one per channel, for depthwise conv."""
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(1)
    window_2d = g @ g.t()
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()


def ssim_map(
    a: torch.Tensor,
    b: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """
    Per-pixel SSIM between two batches in [0, 1], shaped (N, C, H, W).

    Returns a map in roughly [-1, 1] where 1 means locally identical. Reduce it
    with .mean() for a scalar score, or keep the map to localise where the two
    images disagree structurally.
    """
    channels = a.shape[1]
    window = _gaussian_window(window_size, sigma, channels).to(a.device, a.dtype)
    pad = window_size // 2

    mu_a = F.conv2d(a, window, padding=pad, groups=channels)
    mu_b = F.conv2d(b, window, padding=pad, groups=channels)

    mu_a_sq, mu_b_sq, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b

    sigma_a = F.conv2d(a * a, window, padding=pad, groups=channels) - mu_a_sq
    sigma_b = F.conv2d(b * b, window, padding=pad, groups=channels) - mu_b_sq
    sigma_ab = F.conv2d(a * b, window, padding=pad, groups=channels) - mu_ab

    # Stabilising constants from Wang et al. 2004, for data range 1.0.
    c1, c2 = 0.01 ** 2, 0.03 ** 2

    numerator = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a_sq + mu_b_sq + c1) * (sigma_a + sigma_b + c2)
    return numerator / denominator


# --------------------------------------------------------------------------
# The autoencoder
# --------------------------------------------------------------------------
class ConvAutoencoder(nn.Module):
    """
    A compact convolutional autoencoder for 64x64 RGB patches.

    The bottleneck is deliberately narrow (128 x 8 x 8 = 8,192 values against
    64*64*3 = 12,288 input values). Too wide and the network learns to copy its
    input pixel-for-pixel, reconstructing defects perfectly and detecting
    nothing. Too narrow and it blurs normal texture, firing everywhere.
    """

    def __init__(self, in_channels: int = 3, base: int = 32):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base, 3, stride=2, padding=1),      # 64 -> 32
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base * 2, 3, stride=2, padding=1),         # 32 -> 16
            nn.BatchNorm2d(base * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1),     # 16 -> 8
            nn.BatchNorm2d(base * 4),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base * 4, base * 2, 4, stride=2, padding=1),  # 8 -> 16
            nn.BatchNorm2d(base * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1),      # 16 -> 32
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base, in_channels, 4, stride=2, padding=1),   # 32 -> 64
            nn.Sigmoid(),   # inputs are scaled to [0, 1], so bound the output too
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def reconstruction_loss(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Scalar training loss: alpha * MSE + (1 - alpha) * (1 - mean SSIM)."""
    mse = F.mse_loss(reconstructed, original)
    ssim = ssim_map(original, reconstructed).mean()
    return alpha * mse + (1.0 - alpha) * (1.0 - ssim)


# --------------------------------------------------------------------------
# Inference wrapper
# --------------------------------------------------------------------------
class AnomalyDetector:
    """
    Runs the trained autoencoder over a full frame and returns a defect heatmap.

    A frame is far larger than the 64x64 patches the model knows, so it is
    processed as overlapping tiles and the per-tile error maps are averaged back
    into one full-resolution map. Overlapping (stride < patch) avoids the grid
    seams that tiling otherwise leaves across the output.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        device: Optional[str] = None,
        alpha: float = 0.5,
        stride: int = PATCH_SIZE // 2,
    ):
        """
        Args:
            checkpoint_path: trained weights. Defaults to the file beside this one.
            device: "cpu" or "cuda"; auto-detected when omitted.
            alpha: weight of MSE against (1 - SSIM) in the score map.
            stride: tile step in pixels. Smaller is smoother and slower.

        Raises:
            FileNotFoundError: no checkpoint. Train one with
                backend/training/train_autoencoder.py.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.alpha = alpha
        self.stride = stride

        checkpoint_path = Path(checkpoint_path or DEFAULT_CHECKPOINT)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"No autoencoder checkpoint at {checkpoint_path}.\n"
                "Train one with:  python backend/training/train_autoencoder.py"
            )

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state = checkpoint.get("model_state_dict", checkpoint)

        self.model = ConvAutoencoder()
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()

        # Error level seen on defect-free validation patches. Scores are reported
        # relative to it, so "2.0" reads as "twice as unlike normal skin as clean
        # skin is". Without it, raw error values mean nothing on their own.
        self.normal_error = float(checkpoint.get("normal_error", 0.0)) or None
        self.epoch = checkpoint.get("epoch")

        print(f"[ok] Anomaly autoencoder loaded: {checkpoint_path.name}")
        if self.epoch is not None:
            print(f"     Trained {self.epoch} epochs on defect-free patches")
        if self.normal_error:
            print(f"     Reference error on clean skin: {self.normal_error:.5f}")

    # ----------------------------------------------------------------------
    def score_map(self, frame: np.ndarray) -> np.ndarray:
        """
        Per-pixel anomaly score for a BGR frame.

        Returns a float32 array shaped (H, W). Higher means less like normal
        aircraft skin. Values are raw error; use normalise_map or
        detect for something directly displayable.
        """
        if frame is None or frame.size == 0:
            return np.zeros((0, 0), dtype=np.float32)

        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Pad so the last tile is never truncated.
        pad_h = (PATCH_SIZE - height % PATCH_SIZE) % PATCH_SIZE
        pad_w = (PATCH_SIZE - width % PATCH_SIZE) % PATCH_SIZE
        if pad_h or pad_w:
            rgb = np.pad(rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

        padded_h, padded_w = rgb.shape[:2]
        accumulated = np.zeros((padded_h, padded_w), dtype=np.float32)
        counts = np.zeros((padded_h, padded_w), dtype=np.float32)

        ys = list(range(0, padded_h - PATCH_SIZE + 1, self.stride))
        xs = list(range(0, padded_w - PATCH_SIZE + 1, self.stride))

        # Batch a whole row of tiles at a time: one GPU/CPU call per row keeps
        # peak memory bounded while staying far faster than tile-by-tile.
        for y in ys:
            tiles = np.stack([rgb[y:y + PATCH_SIZE, x:x + PATCH_SIZE] for x in xs])
            batch = torch.from_numpy(tiles).permute(0, 3, 1, 2).to(self.device)

            with torch.no_grad():
                reconstructed = self.model(batch)
                sq_err = ((batch - reconstructed) ** 2).mean(dim=1)
                dissimilarity = (1.0 - ssim_map(batch, reconstructed)).mean(dim=1)
                error = self.alpha * sq_err + (1.0 - self.alpha) * dissimilarity

            error = error.cpu().numpy()
            for i, x in enumerate(xs):
                accumulated[y:y + PATCH_SIZE, x:x + PATCH_SIZE] += error[i]
                counts[y:y + PATCH_SIZE, x:x + PATCH_SIZE] += 1.0

        counts[counts == 0] = 1.0
        return (accumulated / counts)[:height, :width]

    # ----------------------------------------------------------------------
    @staticmethod
    def normalise_map(score: np.ndarray, percentile: float = 99.0) -> np.ndarray:
        """
        Scale a score map to [0, 1] for display.

        Clipped at a high percentile rather than the maximum, so one extreme
        pixel (a specular highlight, a dead sensor pixel) cannot flatten the
        rest of the map to near-zero.
        """
        if score.size == 0:
            return score
        low = float(score.min())
        high = float(np.percentile(score, percentile))
        if high <= low:
            return np.zeros_like(score)
        return np.clip((score - low) / (high - low), 0.0, 1.0)

    # ----------------------------------------------------------------------
    def heatmap(
        self,
        frame: np.ndarray,
        score: Optional[np.ndarray] = None,
        opacity: float = 0.5,
    ) -> np.ndarray:
        """Blend a JET-coloured anomaly map over the frame, for display."""
        if score is None:
            score = self.score_map(frame)
        normalised = self.normalise_map(score)
        coloured = cv2.applyColorMap((normalised * 255).astype(np.uint8), cv2.COLORMAP_JET)
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        return cv2.addWeighted(frame, 1.0 - opacity, coloured, opacity, 0.0)

    # ----------------------------------------------------------------------
    def detect(self, frame: np.ndarray, threshold: float = 0.6, min_area: int = 64) -> Dict:
        """
        Full unsupervised pass: score, threshold, and box the anomalous regions.

        Args:
            frame: BGR image.
            threshold: cut on the NORMALISED map, in [0, 1]. Higher is stricter.
            min_area: ignore blobs smaller than this many pixels, which are
                usually sensor noise rather than damage.

        Returns:
            {
              "regions": [{id, bbox, area, peak_score, mean_score}, ...],
              "score_map": float32 (H, W) raw scores,
              "heatmap": BGR overlay,
              "anomaly_ratio": fraction of the frame above threshold,
              "relative_error": mean error as a multiple of clean-skin error,
            }
        """
        score = self.score_map(frame)
        if score.size == 0:
            return {"regions": [], "score_map": score, "heatmap": frame,
                    "anomaly_ratio": 0.0, "relative_error": None}

        normalised = self.normalise_map(score)
        mask = (normalised >= threshold).astype(np.uint8)

        # Close small gaps so one defect yields one region, not a constellation.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            window = score[y:y + h, x:x + w]
            regions.append({
                "id": len(regions),
                "bbox": [float(x), float(y), float(x + w), float(y + h)],
                "area": float(area),
                "peak_score": float(window.max()) if window.size else 0.0,
                "mean_score": float(window.mean()) if window.size else 0.0,
            })

        regions.sort(key=lambda r: r["peak_score"], reverse=True)
        for i, r in enumerate(regions):
            r["id"] = i

        relative = None
        if self.normal_error:
            relative = float(score.mean() / self.normal_error)

        return {
            "regions": regions,
            "score_map": score,
            "heatmap": self.heatmap(frame, score),
            "anomaly_ratio": float(mask.mean()),
            "relative_error": relative,
        }
