"""Block data type — a spatially coherent group of detected text lines."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .text_detector import TextLine


@dataclass
class Block:
    """A spatially coherent group of text lines forming one semantic unit.

    Attributes:
        bbox: Tight axis-aligned bounding box over all constituent lines,
            shape (4,) int32: x1, y1, x2, y2 in rectified 1920×1080 space.
        confidence: Maximum detection confidence among constituent lines.
        lines: Individual TextLine anchors that make up this block.
    """

    bbox: np.ndarray
    confidence: float
    lines: list[TextLine] = field(default_factory=list)
