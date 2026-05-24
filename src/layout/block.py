"""Shared data types and abstract base for text-line grouping strategies."""

from abc import ABC, abstractmethod
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


class TextLineClusterer(ABC):
    """Abstract interface for strategies that cluster TextLines into Blocks."""

    @abstractmethod
    def group(self, lines: list[TextLine]) -> list[Block]:
        """Group detected text lines into spatially coherent blocks.

        Args:
            lines: Text lines produced by Stage 5 (PaddleOCR detection).

        Returns:
            List of Blocks, each covering one or more adjacent lines.
        """

    @staticmethod
    def compute_bbox(lines: list[TextLine]) -> np.ndarray:
        """Return the tight axis-aligned bbox enclosing all *lines*.

        Args:
            lines: Non-empty list of TextLine objects.

        Returns:
            Shape (4,) int32 array: x1, y1, x2, y2.
        """
        boxes = np.array([line.bbox for line in lines])
        return np.array(
            [
                np.min(boxes[:, 0]),
                np.min(boxes[:, 1]),
                np.max(boxes[:, 2]),
                np.max(boxes[:, 3]),
            ],
            dtype=np.int32,
        )
