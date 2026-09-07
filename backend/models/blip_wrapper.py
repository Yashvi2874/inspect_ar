"""
BLIP Image Captioning Wrapper
Extracted from Captioning-of-Aircraft-Images.ipynb
"""

import cv2
import numpy as np
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from pathlib import Path
from typing import Dict, Optional


class BLIPCaptioner:
    """
    BLIP-based image captioning and summarization for aircraft damage.
    Generates natural language descriptions of detected damage.
    """
    
    def __init__(self, device: str = None):
        """
        Initialize BLIP captioner.
        
        Args:
            device: 'cpu' or 'cuda' (handled by transformers library)
        """
        self.device = device or "cpu"

        # BLIP needs roughly 1.5-2 GB of RAM to materialise its weights. When
        # that is not available the load does not raise a Python exception --
        # the process dies with a native allocation failure, which no
        # try/except can catch. Checking first turns an uncatchable crash into
        # an ordinary degraded start.
        if not self._enough_memory():
            self.processor = None
            self.model = None
            return

        try:
            print("[..] Loading BLIP model and processor...")
            self.processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
            self.model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
            print("[ok] BLIP model loaded")
        except Exception as e:
            print(f"[warn] Could not load BLIP, captions disabled: {e}")
            self.processor = None
            self.model = None

    @property
    def available(self) -> bool:
        """True when the model really loaded and captions will be meaningful."""
        return self.model is not None and self.processor is not None

    # Headroom in bytes required before attempting the load.
    REQUIRED_MEMORY = 2_000_000_000

    @classmethod
    def _enough_memory(cls) -> bool:
        """True when there is plausibly enough free RAM to load BLIP.

        Returns True when psutil is unavailable: refusing to load on a machine
        we cannot measure would be worse than trying.
        """
        try:
            import psutil
        except ImportError:
            return True

        available = psutil.virtual_memory().available
        if available >= cls.REQUIRED_MEMORY:
            return True

        print(f"[warn] Only {available / 1e9:.1f} GB RAM available; BLIP needs "
              f"about {cls.REQUIRED_MEMORY / 1e9:.1f} GB.")
        print("[warn] Skipping BLIP rather than risking a native crash. "
              "Close other programs, or run with INSPECT_AR_NO_BLIP=1.")
        return False
    
    def generate_caption(self, image: np.ndarray) -> str:
        """
        Generate a caption for the image.
        
        Args:
            image: BGR image (numpy array)
        
        Returns:
            Generated caption string
        """
        if self.model is None or self.processor is None:
            return "Model not loaded"
        
        try:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image_rgb)
            
            # Set prompt for caption
            prompt = "This is a picture of"
            
            # Process inputs
            inputs = self.processor(
                images=pil_image,
                text=prompt,
                return_tensors="pt"
            )
            
            # Generate caption
            output = self.model.generate(**inputs, max_length=50)
            
            # Decode result
            caption = self.processor.decode(output[0], skip_special_tokens=True)
            
            return caption
        except Exception as e:
            print(f"Error generating caption: {e}")
            return "Error processing image"
    
    def generate_summary(self, image: np.ndarray) -> str:
        """
        Generate a detailed summary for the image.
        
        Args:
            image: BGR image (numpy array)
        
        Returns:
            Generated summary string
        """
        if self.model is None or self.processor is None:
            return "Model not loaded"
        
        try:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image_rgb)
            
            # Set prompt for summary
            prompt = "This is a detailed photo showing"
            
            # Process inputs
            inputs = self.processor(
                images=pil_image,
                text=prompt,
                return_tensors="pt"
            )
            
            # Generate summary
            output = self.model.generate(**inputs, max_length=100)
            
            # Decode result
            summary = self.processor.decode(output[0], skip_special_tokens=True)
            
            return summary
        except Exception as e:
            print(f"Error generating summary: {e}")
            return "Error processing image"
    
    def generate_text(self, image: np.ndarray, task: str = "caption") -> str:
        """
        Generate caption or summary based on task.
        
        Args:
            image: BGR image (numpy array)
            task: "caption" or "summary"
        
        Returns:
            Generated text
        """
        if task.lower() == "caption":
            return self.generate_caption(image)
        elif task.lower() == "summary":
            return self.generate_summary(image)
        else:
            return "Invalid task. Use 'caption' or 'summary'"
    
    def process_detections(self, frame: np.ndarray, detections: list) -> Dict:
        """
        Generate captions for detected objects.
        
        Args:
            frame: BGR image
            detections: List of detection dictionaries
        
        Returns:
            Dictionary with captions for each detection
        """
        results = {"captions": {}}
        
        for det in detections:
            obj_id = det.get("id", 0)
            bbox = det.get("bbox", [0, 0, frame.shape[1], frame.shape[0]])
            x1, y1, x2, y2 = map(int, bbox)
            
            # Ensure valid ROI
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            roi = frame[y1:y2, x1:x2]
            
            if roi.size == 0:
                continue
            
            # Generate caption for this ROI
            caption = self.generate_caption(roi)
            
            results["captions"][obj_id] = {
                "caption": caption,
                "bbox": [x1, y1, x2, y2]
            }
        
        return results
