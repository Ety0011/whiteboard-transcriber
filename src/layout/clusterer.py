"""Abstract base for text-line clustering strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .text_detector import TextLine
from .block import Block


class BaseTextLineClusterer(ABC):
    """Abstract interface for strategies that cluster TextLines into Blocks."""

    @abstractmethod
    def cluster(self, lines: list[TextLine]) -> list[Block]:
        """Cluster detected text lines into spatially coherent blocks.

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
