import enum
import logging
import multiprocessing as mp
from dataclasses import dataclass, field

import numpy as np


class AnchorType(enum.Enum):
    TEXT_LINE = "TEXT_LINE"


@dataclass
class Anchor:
    bbox: np.ndarray  # (4,) int32: x1, y1, x2, y2
    confidence: float
    anchor_type: AnchorType


@dataclass
class DetectorResult:
    anchors: list[Anchor] = field(default_factory=list)


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


def _raw_to_anchors(raw: list, h: int, w: int) -> list[Anchor]:
    anchors = []
    for poly, score in _extract_polys(raw):
        bbox = _polygon_to_bbox(poly, h, w)
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            continue
        anchors.append(
            Anchor(bbox=bbox, confidence=score, anchor_type=AnchorType.TEXT_LINE)
        )
    return anchors


def _worker_main(
    in_q: mp.Queue,
    out_q: mp.Queue,
    box_thresh: float,
    unclip_ratio: float,
) -> None:
    """PaddleOCR text detection loop — runs in a dedicated child process."""
    import logging as _log

    _log.basicConfig(level=logging.DEBUG)
    log = _log.getLogger(__name__)

    from paddleocr import TextDetection

    detector = TextDetection(
        model_name="PP-OCRv5_server_det",
        box_thresh=box_thresh,
        unclip_ratio=unclip_ratio,
    )
    log.info("AnchorDetector: PP-OCRv5_server_det ready")

    while True:
        composite = in_q.get()  # block until work arrives
        if composite is None:  # shutdown sentinel
            break

        anchors: list[Anchor] = []
        try:
            h, w = composite.shape[:2]
            raw = detector.predict(composite)
            anchors = _raw_to_anchors(raw, h, w)
            log.debug("AnchorDetector: %d TEXT_LINE anchors", len(anchors))
        except Exception:
            log.exception("PaddleOCR detection failed")

        result = DetectorResult(anchors=anchors)
        try:
            out_q.get_nowait()
        except Exception:
            pass
        try:
            out_q.put_nowait(result)
        except Exception:
            pass


class AnchorDetector:
    """Non-blocking PaddleOCR anchor detector."""

    def __init__(
        self,
        box_thresh: float = 0.6,
        unclip_ratio: float = 1.2,
    ) -> None:
        self._cached = DetectorResult()
        self._in_q: mp.Queue = mp.Queue(maxsize=1)
        self._out_q: mp.Queue = mp.Queue(maxsize=1)
        self._worker = mp.Process(
            target=_worker_main,
            args=(self._in_q, self._out_q, box_thresh, unclip_ratio),
            daemon=True,
            name="paddle-detect",
        )
        self._worker.start()
        print(f"AnchorDetector worker started (pid={self._worker.pid})")

    def detect(self, composite: np.ndarray) -> DetectorResult:
        """Submit composite for async detection; return latest cached result."""
        try:
            self._in_q.put_nowait(composite)
        except Exception:
            pass

        try:
            self._cached = self._out_q.get_nowait()
        except Exception:
            pass

        return self._cached

    def shutdown(self) -> None:
        """Signal the worker to stop and wait for clean exit."""
        try:
            self._in_q.put_nowait(None)
        except Exception:
            pass
        self._worker.join(timeout=5)
        if self._worker.is_alive():
            self._worker.terminate()
        print("AnchorDetector worker stopped")


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
