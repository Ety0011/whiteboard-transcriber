"""Stage 4 — Change Detection (pipeline gate).

Finds regions of the board that have new or modified ink since the last
processed frame. This stage acts as a gate: if no regions survive the
filter, Stages 5–7 are skipped entirely for the current cycle.

Algorithm:
    1. Convert current and previous composites to grayscale.
    2. cv2.absdiff to produce a difference image.
    3. cv2.threshold (or adaptive) to binarise.
    4. cv2.morphologyEx (open + close) to remove noise.
    5. cv2.findContours + cv2.boundingRect to extract changed regions.
    6. Filter by minimum area (> 400 px²).
    7. Perceptual hash deduplication via imagehash.phash — regions whose
       hash matches an already-processed entry in the session hash table
       are skipped.

Key libraries: OpenCV (cv2), NumPy, imagehash.
"""

from __future__ import annotations

import numpy as np


def process(current: np.ndarray, previous: np.ndarray) -> list[dict]:
    """Detect changed regions between *current* and *previous* composites.

    Args:
        current:  BGR uint8 board composite for the current frame.
        previous: BGR uint8 board composite for the previous processed frame.

    Returns:
        List of region dicts, each with keys:
            ``x``, ``y``, ``w``, ``h``  — bounding box in board coordinates,
            ``hash``                     — perceptual hash (ImageHash object).
        Returns an empty list if no meaningful changes are detected.
    """
    raise NotImplementedError
