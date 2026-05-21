from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .anchor_detector import Anchor


@dataclass
class EntityGroup:
    anchors: list[Anchor]
    bbox: np.ndarray  # (4,) int32: min(x1), min(y1), max(x2), max(y2)
    confidence: float  # max confidence across constituent anchors


class LayoutAggregatorStrategy(ABC):
    """
    Unified interface for layout parsing and anchor aggregation strategies.
    Defines stable contract for production routing.
    """

    @abstractmethod
    def group(self, anchors: list[Anchor]) -> list[EntityGroup]:
        """Groups scattered anchors into structurally coherent macro blocks."""
        pass

    @staticmethod
    def compute_macro_bbox(anchors: list[Anchor]) -> np.ndarray:
        """Helper to dynamically calculate encapsulating box coordinates."""
        boxes = np.array([a.bbox for a in anchors])
        return np.array(
            [
                np.min(boxes[:, 0]),
                np.min(boxes[:, 1]),
                np.max(boxes[:, 2]),
                np.max(boxes[:, 3]),
            ],
            dtype=np.int32,
        )
