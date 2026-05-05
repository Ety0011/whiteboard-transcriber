"""Stage 3 — Surface Reconstruction.

Maintains a clean, unobstructed view of the whiteboard surface using
OpenCV's MOG2 Gaussian Mixture background subtractor. Person-region
pixels are masked out before feeding frames to MOG2 so that people
standing still do not corrupt the background model.

The composite output is: background_image where mask==1, current_frame
where mask==0.

Key OpenCV API: cv2.createBackgroundSubtractorMOG2,
bg_subtractor.apply(), bg_subtractor.getBackgroundImage().
"""

from __future__ import annotations

import numpy as np


def process(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Update the background model and return the clean board composite.

    Args:
        frame: BGR uint8 warped frame from Stage 1.
        mask:  Binary person mask from Stage 2 (uint8, 0=board, 1=person).

    Returns:
        BGR uint8 composite image showing the board surface without people.
    """
    raise NotImplementedError
