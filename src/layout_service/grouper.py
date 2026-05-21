from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from .text_line_detector import Anchor


@dataclass
class Block:
    poly: np.ndarray        # (N, 2) int32 — detector polygon (bbox corners for anchor-based)
    bbox: np.ndarray        # (4,) int32: x1, y1, x2, y2 — axis-aligned
    label: str              # "TEXT" | "MATH" | "TABLE" | "DIAGRAM"
    confidence: float       # detector confidence [0, 1]
    anchors: list[Anchor] = field(default_factory=list)  # constituent text-line anchors; [] for non-anchor detectors


class AnchorGrouper(ABC):
    """Unified interface for anchor aggregation strategies."""

    @abstractmethod
    def group(self, anchors: list[Anchor]) -> list[Block]:
        """Groups scattered anchors into structurally coherent blocks."""
        pass

    @staticmethod
    def compute_macro_bbox(anchors: list[Anchor]) -> np.ndarray:
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

    @staticmethod
    def compute_macro_poly(anchors: list[Anchor]) -> np.ndarray:
        boxes = np.array([a.bbox for a in anchors])
        x1 = int(np.min(boxes[:, 0]))
        y1 = int(np.min(boxes[:, 1]))
        x2 = int(np.max(boxes[:, 2]))
        y2 = int(np.max(boxes[:, 3]))
        return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
