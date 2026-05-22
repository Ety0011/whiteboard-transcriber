"""Composite layout detector: PP-OCRv5_server_det + any TextLineGrouper strategy."""

import logging

import numpy as np

from .base import BaseLayoutDetector
from .block import Block, TextLineGrouper
from .text_detector import TextLineDetector

log = logging.getLogger(__name__)


class TextBlockDetector(BaseLayoutDetector):
    """Compose TextLineDetector with a pluggable TextLineGrouper.

    Bridges Stage 5 text-line detection and Stage 6 grouping into the
    BaseLayoutDetector list[Block] contract expected by Discovery.

    Args:
        strategy: Grouping algorithm to apply to detected text lines.
        box_thresh: Minimum confidence for PaddleOCR to report a text line.
        unclip_ratio: Expansion ratio applied to detected polygon outlines.
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
        """Instantiate and load TextLineDetector inside the worker subprocess."""
        log.info("loading with strategy=%s", type(self.strategy).__name__)
        self.line_detector = TextLineDetector(
            box_thresh=self.box_thresh, unclip_ratio=self.unclip_ratio
        )
        self.line_detector.load()

    def detect(self, frame: np.ndarray) -> list[Block]:
        """Detect text lines then group them into blocks, sorted top-to-bottom.

        Args:
            frame: BGR uint8 clean board composite from Stage 4.

        Returns:
            List of Blocks sorted ascending by bbox y1.
        """
        lines = self.line_detector.detect(frame)
        if not lines:
            return []
        return sorted(self.strategy.group(lines), key=lambda b: b.bbox[1])

    def shutdown(self) -> None:
        """Propagate shutdown to the TextLineDetector (no-op for sync detector)."""
        if self.line_detector is not None:
            self.line_detector.shutdown()
