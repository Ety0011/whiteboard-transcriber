"""Pipeline orchestrator.

Chains Stages 1–7 sequentially for a single input frame. Stage 4 acts
as a gate: if no changed regions are detected, Stages 5–7 are skipped
and None is returned, saving the cost of layout classification and OCR.

This module is called from the processing thread in main.py. All
long-lived model objects (EasyOCR Reader, TrOCR model, YOLO model,
MediaPipe segmenter, MOG2 subtractor) should be initialised once and
passed in — not re-created per frame.
"""

from __future__ import annotations

import numpy as np


def process(frame: np.ndarray) -> str | None:
    """Run one full pipeline cycle on *frame*.

    Args:
        frame: Latest BGR uint8 frame pulled from the camera queue.

    Returns:
        Path to the updated Markdown file as a string if the cycle
        produced output, or ``None`` if Stage 4 detected no changes.
    """
    raise NotImplementedError
