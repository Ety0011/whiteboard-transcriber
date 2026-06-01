"""Stage 6 — Text Line Detection (PaddleOCR PP-OCRv5_server_det).

TextLineDetector runs synchronously inside the layout-detector subprocess.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np


@dataclass
class TextLine:
    """A single detected text-line anchor from PaddleOCR.

    Attributes:
        bbox: Axis-aligned bounding box, shape (4,) int32: x1, y1, x2, y2
            in rectified 1920×1080 coordinate space.
        confidence: Detection confidence score from PP-OCRv5_server_det.
    """

    bbox: np.ndarray
    confidence: float

logger = logging.getLogger(__name__)


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
    """Convert raw PaddleOCR output to a list of TextLine objects.

    Args:
        raw: Raw list returned by TextDetection.predict().
        h: Image height in pixels (for bbox clamping).
        w: Image width in pixels (for bbox clamping).

    Returns:
        List of TextLine objects with valid (non-degenerate) bboxes.
    """
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
    called from inside the layout-detector subprocess, which provides async
    isolation relative to the main process.
    """

    def __init__(
        self,
        box_thresh: float = 0.6,
        unclip_ratio: float = 1.2,
    ) -> None:
        """Configure the detector.

        Args:
            box_thresh: Minimum average pixel score inside a polygon for it to
                be reported as a text line.
            unclip_ratio: Vatti clipping expansion factor applied to each
                detected polygon. Higher values grow bboxes outward, reducing
                inter-fragment gaps seen by the clusterer.
            thresh: Pixel-level binarization threshold on the probability map.
                Lower values produce larger connected text regions and fewer
                fragments before expansion.
        """
        self._box_thresh = box_thresh
        self._unclip_ratio = unclip_ratio
        self._detector = None

    def load(self) -> None:
        """Load PP-OCRv5_server_det inside the worker subprocess.

        Suppresses C++-level stdout/stderr during model initialisation to
        prevent glog/TFLite noise from polluting the process output.
        """
        from logging_config import devnull_fds

        with devnull_fds(1, 2):
            from paddleocr import TextDetection

            kwargs: dict = dict(
                model_name="PP-OCRv5_server_det",
                box_thresh=self._box_thresh,
                unclip_ratio=self._unclip_ratio,
            )
            self._detector = TextDetection(**kwargs)
        logger.info("PP-OCRv5_server_det ready")

    def detect(self, composite: np.ndarray) -> list[TextLine]:
        """Run text-line detection on *composite* and return detected lines.

        Args:
            composite: BGR uint8 clean board composite from Stage 5.

        Returns:
            List of TextLine objects, or empty list on detection failure.
        """
        if self._detector is None:
            logger.error("Detect called before load completed")
            return []

        h, w = composite.shape[:2]
        try:
            raw = self._detector.predict(composite)
            return _raw_to_lines(raw, h, w)
        except Exception:
            logger.exception("PaddleOCR detection failed")
            return []

    def shutdown(self) -> None:
        """No-op — model is released when the worker process exits."""
