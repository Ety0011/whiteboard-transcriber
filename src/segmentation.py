"""Stage 2 — Person Segmentation.

Produces a binary mask that marks pixels belonging to people (arms,
torso, hands holding markers) so they can be excluded from the
background model in Stage 3.

Library: MediaPipe Selfie Segmentation (model_selection=1, landscape).
Important: MediaPipe expects RGB input — always convert from BGR before
calling segmenter.process(), or the mask will be silently incorrect.

The output mask is uint8 with values 0 (board) and 1 (person).
"""

from __future__ import annotations

import numpy as np


def process(frame: np.ndarray) -> np.ndarray:
    """Compute a binary person mask for *frame*.

    Args:
        frame: BGR uint8 image (perspective-corrected, from Stage 1).

    Returns:
        Binary mask as uint8 ndarray with shape ``(H, W)``.
        Pixel value 1 means "person present"; 0 means "board visible".
    """
    raise NotImplementedError
