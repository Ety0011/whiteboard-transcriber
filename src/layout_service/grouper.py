from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from .text_line_detector import TextLine


@dataclass
class Block:
    bbox: np.ndarray              # (4,) int32: x1, y1, x2, y2
    confidence: float
    lines: list[TextLine] = field(default_factory=list)


class TextLineGrouper(ABC):
    """Unified interface for text line aggregation strategies."""

    @abstractmethod
    def group(self, lines: list[TextLine]) -> list[Block]:
        """Groups detected text lines into structurally coherent blocks."""
        pass

    @staticmethod
    def compute_bbox(lines: list[TextLine]) -> np.ndarray:
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
