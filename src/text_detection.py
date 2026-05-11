"""Stage 5 — Text Line Detection.

Takes a list of Region objects from Stage 4 and returns the same regions
enriched with detected text line bounding boxes (in region-crop coordinates).

Model: PaddleOCR PP-OCRv5_server_det via TextDetection.
Load once at startup via init() or the first call to process() — never per frame.

Detection runs synchronously. Region crops are small, so per-crop latency is
acceptable within the pipeline cycle. No child process needed here.

Non-text regions (figure, table) are skipped — their lines list is empty.
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np
from paddleocr import TextDetection

from layout import Region

log = logging.getLogger(__name__)

_SKIP_LABELS: frozenset[str] = frozenset({"figure", "table"})


@dataclasses.dataclass
class TextLine:
    """A detected text line within a layout region."""

    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 in region-crop coordinates
    crop: np.ndarray  # BGR uint8


@dataclasses.dataclass
class RegionWithLines(Region):
    """A layout region enriched with detected text line bounding boxes."""

    lines: list[TextLine] = dataclasses.field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _polygon_to_bbox(
    polygon: list,
    img_h: int,
    img_w: int,
) -> tuple[int, int, int, int]:
    """Convert a polygon to an axis-aligned bbox clamped to image bounds.

    Args:
        polygon: List of (x, y) points (any length ≥ 1).
        img_h: Image height in pixels.
        img_w: Image width in pixels.

    Returns:
        Clamped (x1, y1, x2, y2) tuple.
    """
    xs = [pt[0] for pt in polygon]
    ys = [pt[1] for pt in polygon]
    x1 = max(0, int(min(xs)))
    y1 = max(0, int(min(ys)))
    x2 = min(img_w, int(max(xs)))
    y2 = min(img_h, int(max(ys)))
    return x1, y1, x2, y2


def _parse_lines(raw_results: list, crop: np.ndarray) -> list[TextLine]:
    """Parse TextDetection.predict() output into TextLine objects.

    Expected format: list of dicts with key "dt_polys" containing a list of
    polygon point arrays (each polygon is a list of [x, y] pairs).

    Args:
        raw_results: Raw output from TextDetection.predict().
        crop: The region crop the detection was run on (used for bounds + slicing).

    Returns:
        List of TextLine objects in crop coordinates.
    """
    if not raw_results:
        return []
    h, w = crop.shape[:2]
    lines: list[TextLine] = []
    for result in raw_results:
        polys = result.get("dt_polys", [])
        for poly in polys:
            x1, y1, x2, y2 = _polygon_to_bbox(poly, h, w)
            if x2 <= x1 or y2 <= y1:
                log.debug(
                    "Skipping degenerate text line bbox (%d,%d,%d,%d)", x1, y1, x2, y2
                )
                continue
            line_crop = crop[y1:y2, x1:x2].copy()
            lines.append(TextLine(bbox=(x1, y1, x2, y2), crop=line_crop))
    return lines


# ---------------------------------------------------------------------------
# TextDetector
# ---------------------------------------------------------------------------


class TextDetector:
    """Runs PP-OCRv5_det_server on region crops from Stage 4.

    Loaded once at startup. process() is synchronous — crops are small and
    per-crop latency fits comfortably in the pipeline cycle.
    """

    def __init__(self) -> None:
        """Load PP-OCRv5_server_det. This takes a few seconds on first call."""
        self._engine = TextDetection(model_name="PP-OCRv5_server_det")
        log.info("TextDetector initialised (model=PP-OCRv5_server_det)")

    def process(self, regions: list[Region]) -> list[RegionWithLines]:
        """Detect text lines within each region crop.

        Skips figure and table regions (returns empty lines list for those).

        Args:
            regions: Layout regions from Stage 4.

        Returns:
            Same regions as RegionWithLines, each with a lines list populated.
        """
        results: list[RegionWithLines] = []
        for region in regions:
            if region.label in _SKIP_LABELS:
                lines: list[TextLine] = []
                log.debug("Skipping text detection for label=%r", region.label)
            else:
                lines = self._run_detection(region.crop)
                log.debug(
                    "Detected %d line(s) in region label=%r bbox=%s",
                    len(lines),
                    region.label,
                    region.bbox,
                )
            results.append(
                RegionWithLines(
                    bbox=region.bbox,
                    label=region.label,
                    confidence=region.confidence,
                    crop=region.crop,
                    lines=lines,
                )
            )
        return results

    def _run_detection(self, crop: np.ndarray) -> list[TextLine]:
        """Run text line detection synchronously on a single region crop.

        Used directly by unit tests (monkeypatch TextDetection before calling).

        Args:
            crop: BGR uint8 numpy array — a single region crop from Stage 4.

        Returns:
            List of detected TextLine objects in crop coordinates.
        """
        raw = self._engine.predict(crop)
        return _parse_lines(raw, crop)


# ---------------------------------------------------------------------------
# Module-level singleton + convenience functions
# ---------------------------------------------------------------------------

_global_detector: TextDetector | None = None


def init() -> None:
    """Load PP-OCRv5_server_det. Call once at startup before process().

    Subsequent calls are no-ops only if the singleton already exists; calling
    again replaces it (useful for testing with different model configs).
    """
    global _global_detector
    _global_detector = TextDetector()


def process(regions: list[Region]) -> list[RegionWithLines]:
    """Detect text lines using the module-level singleton detector.

    Lazily initialises with default settings if init() was not called first.

    Args:
        regions: Layout regions from Stage 4.

    Returns:
        Same regions as RegionWithLines with lines populated.
    """
    global _global_detector
    if _global_detector is None:
        log.warning("text_detection.process() called before init() — using defaults")
        _global_detector = TextDetector()
    return _global_detector.process(regions)
