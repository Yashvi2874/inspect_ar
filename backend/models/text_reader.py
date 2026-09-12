"""
Read text stamped or printed on aircraft components.

Why this exists
---------------
A defect report that says "crack found, 12.4 mm" is only half useful. The
question an inspector actually has to answer is *which part* it was found on.
Aircraft components carry that answer already: serial numbers, part numbers and
stencilled labels are printed directly on them.

Reading that text turns an observation into a record. This was suggested by the
judges at HAL AEROTHON '26 during the second evaluation, and it is the missing
half of traceability.

How it works
------------
OCR (Optical Character Recognition) converts pixels of text into characters.
This uses EasyOCR, which runs two neural networks in sequence: a **detector**
that finds where text is in the image, and a **recogniser** that reads each
region. That two-stage design is what lets it handle text at an angle, on a
curved surface, or against a noisy metal background, which a classical
threshold-and-match approach cannot.

Aircraft stencilling is a hostile case for OCR: low contrast, curved panels,
glare, paint wear. So the reader preprocesses for contrast, and every result
carries a confidence so a caller can discard uncertain reads rather than record
a wrong part number, which is worse than recording none.
"""

import re
from typing import Dict, List, Optional

import cv2
import numpy as np


# Serial and part numbers are capitals, digits and separators. Lowercase in a
# result usually means the OCR has misread noise, so its absence is a signal.
SERIAL_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9\-/\. ]{3,}$")


def looks_like_serial(text: str) -> bool:
    """
    Whether a read looks like a part or serial number rather than a phrase.

    The shape test alone is not enough. Aircraft panels also carry stencilled
    warnings, and "INSPECT BEFORE FLIGHT" is uppercase with separators too, so a
    pattern match by itself flags it as a part number. Requiring at least one
    digit separates identifiers from instructions: real part and serial numbers
    are numbered, warnings are not.
    """
    upper = text.upper()
    if not SERIAL_PATTERN.match(upper):
        return False
    return any(character.isdigit() for character in upper)

# Below this, a read is more likely to be wrong than right. Recording a wrong
# part number is worse than recording none, so the default is deliberately high.
DEFAULT_CONFIDENCE = 0.40


class TextReader:
    """Finds and reads text in an image, with confidence for every result."""

    def __init__(
        self,
        languages: Optional[List[str]] = None,
        use_gpu: bool = False,
        min_confidence: float = DEFAULT_CONFIDENCE,
    ):
        """
        Args:
            languages: EasyOCR language codes. Defaults to English, which is
                what aircraft part marking uses.
            use_gpu: run on CUDA if available. CPU is fine for still frames.
            min_confidence: results below this are dropped.

        Raises:
            ImportError: easyocr is not installed.
        """
        try:
            import easyocr
        except ImportError as e:
            raise ImportError(
                "TextReader needs easyocr. Install it with: pip install easyocr"
            ) from e

        self.min_confidence = min_confidence

        # Downloads detection and recognition weights (~64 MB) on first use,
        # then caches them. Constructing this is slow; do it once and reuse.
        print("[..] Loading OCR models (first run downloads ~64 MB)...")
        self.reader = easyocr.Reader(languages or ["en"], gpu=use_gpu, verbose=False)
        print("[ok] OCR reader ready")

    # ------------------------------------------------------------------
    @staticmethod
    def _preprocess(image: np.ndarray) -> np.ndarray:
        """
        Raise local contrast so stencilled text separates from bare metal.

        CLAHE (Contrast Limited Adaptive Histogram Equalisation) equalises
        brightness in small tiles rather than across the whole image, which is
        what makes it work on a panel that is glaring at one end and shadowed at
        the other. A global adjustment would blow out the bright end.
        """
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    # ------------------------------------------------------------------
    def read(self, image: np.ndarray, preprocess: bool = True) -> List[Dict]:
        """
        Find and read every piece of text in the image.

        Args:
            image: BGR frame.
            preprocess: apply contrast enhancement first. Worth leaving on for
                metal surfaces; turn it off for clean printed labels.

        Returns:
            A list of {text, confidence, bbox, looks_like_serial}, ordered by
            confidence, highest first. bbox is [x1, y1, x2, y2].
        """
        if image is None or image.size == 0:
            return []

        target = self._preprocess(image) if preprocess else image

        # readtext returns (box_of_4_points, text, confidence) per result.
        try:
            raw = self.reader.readtext(target)
        except Exception as e:
            print(f"[warn] OCR failed on this frame: {e}")
            return []

        results = []
        for box, text, confidence in raw:
            if confidence < self.min_confidence:
                continue

            cleaned = text.strip()
            if not cleaned:
                continue

            # EasyOCR gives four corner points, which may be a rotated
            # quadrilateral. Reduce to an axis-aligned box for the caller.
            xs = [float(p[0]) for p in box]
            ys = [float(p[1]) for p in box]

            results.append({
                "text": cleaned,
                "confidence": round(float(confidence), 4),
                "bbox": [min(xs), min(ys), max(xs), max(ys)],
                "looks_like_serial": looks_like_serial(cleaned),
            })

        results.sort(key=lambda r: r["confidence"], reverse=True)
        return results

    # ------------------------------------------------------------------
    def read_serials(self, image: np.ndarray) -> List[str]:
        """Just the strings that look like part or serial numbers."""
        return [r["text"] for r in self.read(image) if r["looks_like_serial"]]

    # ------------------------------------------------------------------
    def read_near_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        margin: int = 60,
    ) -> Dict:
        """
        Read text around each detected defect, to tie a finding to a component.

        Args:
            frame: BGR image.
            detections: upstream detections, each with an "id" and a "bbox".
            margin: pixels to expand each box by before reading. Stencilling
                sits *beside* damage, not on top of it, so reading only inside
                the defect box would find nothing.

        Returns:
            {"objects": {obj_id: {"text": [...], "serials": [...]}}}
        """
        out: Dict[str, Dict] = {"objects": {}}
        if frame is None or frame.size == 0:
            return out

        height, width = frame.shape[:2]

        for det in detections:
            x1, y1, x2, y2 = (int(v) for v in det["bbox"])
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(width, x2 + margin)
            y2 = min(height, y2 + margin)
            if x2 <= x1 or y2 <= y1:
                continue

            found = self.read(frame[y1:y2, x1:x2])
            if not found:
                continue

            out["objects"][det["id"]] = {
                "text": [f["text"] for f in found],
                "serials": [f["text"] for f in found if f["looks_like_serial"]],
            }

        return out

    # ------------------------------------------------------------------
    @staticmethod
    def draw(frame: np.ndarray, results: List[Dict]) -> np.ndarray:
        """Draw read text onto a copy of the frame, for demos and debugging."""
        out = frame.copy() if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        for r in results:
            x1, y1, x2, y2 = (int(v) for v in r["bbox"])
            # Serials in green, other text in amber, so the useful reads stand out.
            colour = (0, 220, 0) if r["looks_like_serial"] else (0, 180, 255)
            cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(out, f"{r['text']} {r['confidence']:.2f}",
                        (x1, max(14, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)

        return out
