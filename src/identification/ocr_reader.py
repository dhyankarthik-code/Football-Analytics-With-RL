"""
OCR Reader module for extracting player jersey numbers.
"""
from typing import Optional
import numpy as np
import cv2
from paddleocr import PaddleOCR
from src.utils.logger import get_logger

logger = get_logger("OCRReader")

class OCRReader:
    def __init__(self, config):
        """
        Initialize PaddleOCR for jersey number extraction.
        """
        self.config = config.identification
        if not self.config.ocr_enabled:
            logger.info("OCR is disabled in config.")
            return

        lang = self.config.ocr_model_lang
        logger.info(f"Initializing PaddleOCR (lang={lang})...")
        
        # Suppress Paddle logs
        self.ocr = PaddleOCR(use_angle_cls=True, lang=lang, device="cpu") 

    def predict(self, frame: np.ndarray, bbox: np.ndarray) -> Optional[int]:
        """
        Attempt to read jersey number from a player bounding box.
        """
        if not self.config.ocr_enabled:
            return None
            
        crop = self._get_crop(frame, bbox)
        if crop is None or crop.size == 0:
            return None
            
        # Preprocessing for better OCR
        processed_crop = self._preprocess(crop)
        
        try:
            result = self.ocr.ocr(processed_crop, cls=True)
            if not result or result[0] is None:
                return None
                
            # Parse result: list of [box, (text, score)]
            # We want the text with highest confidence
            for line in result[0]:
                text, score = line[1]
                if score > 0.6 and text.isdigit():
                    # Jersey numbers typically 1-99
                    num = int(text)
                    if 0 <= num <= 99:
                        return num
                        
        except Exception:
            # OCR errors are common on blur, ignore
            pass
            
        return None

    def _get_crop(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = map(int, bbox)
        h, w, _ = frame.shape
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        return frame[y1:y2, x1:x2]

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """Apply filters to enhance number visibility."""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Contrast Limited Adaptive Histogram Equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Convert back to BGR for Paddle
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
