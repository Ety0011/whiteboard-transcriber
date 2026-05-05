"""Stage 5 — Layout Classification.

Classifies each changed region (from Stage 4) as one of:
text_block, diagram, table, or equation.

Model: DocLayout-YOLO (purpose-built for document layout, pre-trained on
DocLayNet + D4LA) or YOLOv11n fine-tuned on whiteboard data. Both are
loaded via the Ultralytics API. The model is loaded once at module
initialisation — do not reload per frame.

Key library: ultralytics (YOLO).
"""

from __future__ import annotations

import numpy as np


def process(composite: np.ndarray, regions: list[dict]) -> list[dict]:
    """Classify each region in *regions* by its content type.

    Args:
        composite: BGR uint8 board composite (clean surface from Stage 3).
        regions:   List of region dicts from Stage 4
                   (keys: ``x``, ``y``, ``w``, ``h``, ``hash``).

    Returns:
        The same list of region dicts with an additional ``label`` key
        (str: ``"text_block"``, ``"diagram"``, ``"table"``, ``"equation"``).
    """
    raise NotImplementedError
