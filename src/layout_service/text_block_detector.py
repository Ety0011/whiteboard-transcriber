import numpy as np

from .grouper import AnchorGrouper, Block
from .text_line_detector import TextLineDetector
from .base import BaseLayoutDetector


class TextBlockDetector(BaseLayoutDetector):
    """
    Composes TextLineDetector (PP-OCRv5_server_det) with any AnchorGrouper.
    Bridges the anchor-based detection into the BaseLayoutDetector list[Block] contract.
    """

    def __init__(
        self,
        strategy: AnchorGrouper,
        box_thresh: float = 0.6,
        unclip_ratio: float = 1.2,
    ):
        self.strategy = strategy
        self.box_thresh = box_thresh
        self.unclip_ratio = unclip_ratio
        self.anchor_detector: TextLineDetector | None = None

    def load(self) -> None:
        print(
            f"[TextBlockDetector] Spawning TextLineDetector "
            f"with strategy={type(self.strategy).__name__}..."
        )
        self.anchor_detector = TextLineDetector(
            box_thresh=self.box_thresh, unclip_ratio=self.unclip_ratio
        )

    def detect(self, frame: np.ndarray) -> list[Block]:
        result = self.anchor_detector.detect(frame)
        anchors = result.anchors

        if not anchors:
            return []

        blocks = self.strategy.group(anchors)
        return sorted(blocks, key=lambda b: b.poly[:, 1].min())
