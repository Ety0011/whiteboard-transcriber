"""Stage 1 — Spatial Registration.

Detects the whiteboard quadrilateral (via Canny + HoughLinesP contour
analysis, or ArUco corner markers) and applies a perspective homography
so every downstream frame maps to a flat, canonical board view at a
fixed output resolution (e.g. 1280×720).

The computed homography matrix is cached and only recomputed every N
seconds or when corner displacement exceeds a threshold.

Key OpenCV APIs: cv2.getPerspectiveTransform, cv2.findHomography,
cv2.warpPerspective, cv2.aruco.detectMarkers.
"""

from __future__ import annotations

import numpy as np


def process(frame: np.ndarray) -> np.ndarray:
    """Warp *frame* to remove perspective distortion.

    Args:
        frame: BGR uint8 image captured from the camera.

    Returns:
        Perspective-corrected BGR uint8 image at the canonical board resolution.
    """
    raise NotImplementedError
