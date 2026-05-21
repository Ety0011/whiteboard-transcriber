import numpy as np

from .grouper import AnchorGrouper
from .text_line_detector import TextLineDetector
from .base import BaseLayoutDetector


class TextBlockDetector(BaseLayoutDetector):
    """
    Composes TextLineDetector (PP-OCRv5_server_det) with any AnchorGrouper.
    Bridges the anchor → EntityGroup pipeline into the BaseLayoutDetector frame → list[dict] contract.
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

    def detect(self, frame: np.ndarray) -> list[dict]:
        result = self.anchor_detector.detect(frame)
        anchors = result.anchors

        if not anchors:
            return []

        groups = self.strategy.group(anchors)

        regions = []
        for g_idx, group in enumerate(groups):
            x1, y1, x2, y2 = group.bbox.tolist()
            poly_pts = np.array(
                [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32
            )
            regions.append(
                {
                    "text": f"Block {g_idx} ({len(group.anchors)} lines)",
                    "poly": poly_pts,
                    "label": "TEXT",
                    "color": (255, 0, 255),
                }
            )

        return sorted(regions, key=lambda r: r["poly"][:, 1].min())
