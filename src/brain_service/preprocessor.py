"""Stage 7 — Crop preprocessing for VLM inference.

Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) on the
L channel in LAB color space to maximise contrast regardless of marker
quality or lighting, before the crop is sent to GOT-OCR 2.0.
"""

from __future__ import annotations

import cv2
import numpy as np


def preprocess_crop(bgr: np.ndarray) -> np.ndarray:
    """Enhance a board crop for VLM inference via CLAHE.

    Args:
        bgr: BGR uint8 entity crop from the clean board composite.

    Returns:
        BGR uint8 image with enhanced local contrast.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
