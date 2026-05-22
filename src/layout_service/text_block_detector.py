import logging

import numpy as np

from .grouper import TextLineGrouper, Block
from .text_line_detector import TextLineDetector
from .base import BaseLayoutDetector

log = logging.getLogger(__name__)


class TextBlockDetector(BaseLayoutDetector):
    """
    Composes TextLineDetector (PP-OCRv5_server_det) with any TextLineGrouper.
    Bridges text-line detection into the BaseLayoutDetector list[Block] contract.
    """

    def __init__(
        self,
        strategy: TextLineGrouper,
        box_thresh: float = 0.6,
        unclip_ratio: float = 1.2,
    ):
        self.strategy = strategy
        self.box_thresh = box_thresh
        self.unclip_ratio = unclip_ratio
        self.line_detector: TextLineDetector | None = None

    def load(self) -> None:
        log.info(
            "TextBlockDetector: loading TextLineDetector with strategy=%s",
            type(self.strategy).__name__,
        )
        self.line_detector = TextLineDetector(
            box_thresh=self.box_thresh, unclip_ratio=self.unclip_ratio
        )
        self.line_detector.load()

    def detect(self, frame: np.ndarray) -> list[Block]:
        lines = self.line_detector.detect(frame)
        if not lines:
            return []
        return sorted(self.strategy.group(lines), key=lambda b: b.bbox[1])

    def shutdown(self) -> None:
        if self.line_detector is not None:
            self.line_detector.shutdown()
