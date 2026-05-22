import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TextLine:
    bbox: np.ndarray  # (4,) int32: x1, y1, x2, y2
    confidence: float


def _extract_polys(raw_results: list) -> list[tuple[list, float]]:
    """Extract (polygon, score) pairs from raw TextDetection output."""
    if not raw_results:
        return []
    results = []
    for result in raw_results:
        polys = result.get("dt_polys", [])
        scores = result.get("dt_scores", [1.0] * len(polys))
        for poly, score in zip(polys, scores):
            results.append(
                ([[float(pt[0]), float(pt[1])] for pt in poly], float(score))
            )
    return results


def _polygon_to_bbox(polygon: list, img_h: int, img_w: int) -> np.ndarray:
    """Convert polygon to axis-aligned bbox clamped to image bounds."""
    pts = np.array(polygon, dtype=np.float32)
    x1 = int(np.clip(pts[:, 0].min(), 0, img_w))
    y1 = int(np.clip(pts[:, 1].min(), 0, img_h))
    x2 = int(np.clip(pts[:, 0].max(), 0, img_w))
    y2 = int(np.clip(pts[:, 1].max(), 0, img_h))
    return np.array([x1, y1, x2, y2], dtype=np.int32)


def _raw_to_lines(raw: list, h: int, w: int) -> list[TextLine]:
    lines = []
    for poly, score in _extract_polys(raw):
        bbox = _polygon_to_bbox(poly, h, w)
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            continue
        lines.append(TextLine(bbox=bbox, confidence=score))
    return lines


class TextLineDetector:
    """Synchronous PaddleOCR text line detector.

    Runs PP-OCRv5_server_det directly in the calling process. Intended to be
    called from inside the stage5-layout subprocess, which provides async
    isolation relative to the main process.
    """

    def __init__(self, box_thresh: float = 0.6, unclip_ratio: float = 1.2) -> None:
        self._box_thresh = box_thresh
        self._unclip_ratio = unclip_ratio
        self._detector = None

    def load(self) -> None:
        from paddleocr import TextDetection

        self._detector = TextDetection(
            model_name="PP-OCRv5_server_det",
            box_thresh=self._box_thresh,
            unclip_ratio=self._unclip_ratio,
        )
        logger.info("TextLineDetector: PP-OCRv5_server_det ready")

    def detect(self, composite: np.ndarray) -> list[TextLine]:
        h, w = composite.shape[:2]
        try:
            raw = self._detector.predict(composite)
            lines = _raw_to_lines(raw, h, w)
            logger.debug("TextLineDetector: %d text lines", len(lines))
            return lines
        except Exception:
            logger.exception("PaddleOCR detection failed")
            return []

    def shutdown(self) -> None:
        pass


class UnionFind:
    """Disjoint-Set Forest for clustering."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i: int, j: int) -> bool:
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j
            return True
        return False
