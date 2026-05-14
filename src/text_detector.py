"""Stage 5 — Text Line Detection.

Takes a list of Region objects from Stage 4 and returns the same regions
enriched with detected text line bounding boxes (in region-crop coordinates).

Model: PaddleOCR PP-OCRv5_server_det via TextDetection.
Load once at startup via init() or the first call to process() — never per frame.

Inference runs in a dedicated child process (multiprocessing.Process), not a
thread. PaddlePaddle holds the Python GIL during inference, which would block
cv2.waitKey on the main thread if run in a thread. The child process has its
own GIL and never contends with the main process.

process() is non-blocking: it submits the region list, polls the result queue,
and immediately returns the last cached result.

Non-text regions (figure, table) are skipped — their lines list is empty.
"""

from __future__ import annotations

import dataclasses
import logging
import multiprocessing as mp
from multiprocessing.queues import Empty

import numpy as np
from paddleocr import TextDetection

from layout import LayoutRegion

log = logging.getLogger(__name__)

_SKIP_LABELS: frozenset[str] = frozenset({"figure", "table"})


@dataclasses.dataclass
class TextLine:
    """A detected text line within a layout region."""

    bbox: np.ndarray  # shape (4,) int32: x1, y1, x2, y2 in region-crop coordinates
    confidence: float  # Added: Actual detection score
    crop: np.ndarray  # BGR uint8


@dataclasses.dataclass
class RegionWithLines(LayoutRegion):
    """A layout region enriched with detected text line bounding boxes."""

    lines: list[TextLine] = dataclasses.field(default_factory=list)


# ---------------------------------------------------------------------------
# Child-process helpers (module-level so they are picklable)
# ---------------------------------------------------------------------------


def _inference_worker(
    regions_queue: mp.Queue,
    result_queue: mp.Queue,
    box_thresh: float,
    unclip_ratio: float,
) -> None:
    """Entry point for the child process.

    Owns TextDetection entirely — the main process never touches it.
    Reads region lists from regions_queue, runs inference on each crop,
    puts serialisable polygon lists into result_queue. Runs until it
    receives None as a sentinel.
    """
    engine = TextDetection(
        model_name="PP-OCRv5_server_det",
        box_thresh=box_thresh,
        unclip_ratio=unclip_ratio,
    )
    while True:
        items = regions_queue.get()
        if items is None:
            break
        try:
            poly_lists = _detect_in_worker(engine, items)
        except Exception:
            poly_lists = [[] for _ in items]
        result_queue.put(poly_lists)


def _detect_in_worker(engine, items: list[dict]) -> list[list]:
    """Run text detection on each region crop. Called inside the child process.

    Args:
        engine: TextDetection instance owned by the child process.
        items: List of {"label": str, "crop": np.ndarray} dicts, one per region.

    Returns:
        List (one per region) of polygon lists. Each polygon is a list of
        [x, y] pairs as plain Python floats. Skipped or failed regions get [].
    """
    results = []
    for item in items:
        if item["label"] in _SKIP_LABELS:
            results.append([])
            continue
        try:
            raw = engine.predict(item["crop"])
            polys = _extract_polys(raw)
        except Exception:
            polys = []
        results.append(polys)
    return results


def _extract_polys(raw_results: list) -> list[list]:
    """Extract polygon point lists from raw TextDetection output.

    Converts to plain Python floats so results are safely picklable
    across the process boundary (numpy arrays also pickle but are heavier).

    Args:
        raw_results: Raw output from TextDetection.predict().

    Returns:
        List of polygons; each polygon is a list of [x, y] float pairs.
    """
    if not raw_results:
        return []
    results = []
    for result in raw_results:
        polys = result.get("dt_polys", [])
        scores = result.get("dt_scores", [])
        for poly, score in zip(polys, scores):
            results.append(
                ([[float(pt[0]), float(pt[1])] for pt in poly], float(score))
            )
    return results


def _build_regions_with_lines(
    poly_lists: list[list],
    regions: list[LayoutRegion],
) -> list[RegionWithLines]:
    """Reconstruct RegionWithLines objects in the parent process.

    Combines polygon data returned by the child process with the original
    Region crops (which live in the parent process) to produce TextLine crops.

    Args:
        poly_lists: List of polygon lists per region (from child process).
        regions: Original Region objects from Stage 4, ordered identically.

    Returns:
        Regions enriched with TextLine objects in crop coordinates.
    """
    results = []
    for region, detections in zip(regions, poly_lists):
        h, w = region.crop.shape[:2]
        lines = []
        for poly, score in detections:
            bbox = _polygon_to_bbox(poly, h, w)
            x1, y1, x2, y2 = bbox
            if x2 <= x1 or y2 <= y1:
                continue
            lines.append(
                TextLine(
                    bbox=bbox,
                    confidence=score,
                    crop=region.crop[y1:y2, x1:x2].copy(),
                )
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _polygon_to_bbox(
    polygon: list,
    img_h: int,
    img_w: int,
) -> np.ndarray:
    """Convert a polygon to an axis-aligned bbox clamped to image bounds.

    Args:
        polygon: List of (x, y) points (any length ≥ 1).
        img_h: Image height in pixels.
        img_w: Image width in pixels.

    Returns:
        Clamped int32 array [x1, y1, x2, y2].
    """
    pts = np.array(polygon, dtype=np.float32)
    x1 = int(np.clip(pts[:, 0].min(), 0, img_w))
    y1 = int(np.clip(pts[:, 1].min(), 0, img_h))
    x2 = int(np.clip(pts[:, 0].max(), 0, img_w))
    y2 = int(np.clip(pts[:, 1].max(), 0, img_h))
    return np.array([x1, y1, x2, y2], dtype=np.int32)


def _parse_lines(raw_results: list, crop: np.ndarray) -> list[TextLine]:
    """Parse TextDetection.predict() output into TextLine objects.

    Used by _run_detection() on the synchronous test path. Converts polygons
    to axis-aligned bboxes, skips degenerate results, crops the image.

    Args:
        raw_results: Raw output from TextDetection.predict().
        crop: The region crop the detection was run on.

    Returns:
        List of TextLine objects in crop coordinates.
    """
    detections = _extract_polys(raw_results)
    h, w = crop.shape[:2]
    lines = []
    for poly, score in detections:
        bbox = _polygon_to_bbox(poly, h, w)
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            continue
        lines.append(
            TextLine(bbox=bbox, confidence=score, crop=crop[y1:y2, x1:x2].copy())
        )
    return lines


# ---------------------------------------------------------------------------
# TextDetector
# ---------------------------------------------------------------------------


# TODO: decommission layout, rely on text detector directly
class TextDetector:
    """Runs PP-OCRv5_server_det in a child process so the main GIL is never blocked.

    Same external pattern as Stage 4 LayoutDetector:
    - process() never blocks — returns cached regions immediately.
    - A new detection is submitted whenever the child is idle.

    Args:
        box_thresh: Minimum pixel-level score threshold for a region to be
            included as a text box candidate. Lower values recall more (noisier)
            boxes; higher values are more conservative. Default 0.6.
        unclip_ratio: Controls how much detected text polygons are expanded
            before converting to bounding boxes. Higher values include more context
            around text; lower values hug the text tighter. Default 1.5.
    """

    def __init__(
        self,
        box_thresh: float = 0.6,
        unclip_ratio: float = 1.2,
    ) -> None:
        """Start the child process and load PP-OCRv5_server_det inside it."""
        self._box_thresh = box_thresh
        self._unclip_ratio = unclip_ratio

        self._cached_results: list[RegionWithLines] = []
        self._detecting = False
        self._pending_regions: list[LayoutRegion] | None = None

        self._regions_queue: mp.Queue = mp.Queue(maxsize=1)
        self._result_queue: mp.Queue = mp.Queue(maxsize=1)
        self._worker = mp.Process(
            target=_inference_worker,
            args=(
                self._regions_queue,
                self._result_queue,
                self._box_thresh,
                self._unclip_ratio,
            ),
            daemon=True,
            name="text-detect",
        )
        self._worker.start()
        log.info("TextDetector initialised (model=PP-OCRv5_server_det)")

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def process(self, regions: list[LayoutRegion]) -> list[RegionWithLines]:
        """Submit *regions* to the child process and return the last cached result.

        Never blocks. Polls the result queue on every call; submits a new
        region list when the child is idle.

        Args:
            regions: Layout regions from Stage 4.

        Returns:
            Last cached list of RegionWithLines (empty until first detection).
        """
        # Poll for completed result from child process (non-blocking).
        try:
            poly_lists = self._result_queue.get_nowait()
            if self._pending_regions is not None:
                self._cached_results = _build_regions_with_lines(
                    poly_lists, self._pending_regions
                )
            self._detecting = False
            log.debug("Text detection result: %d regions", len(self._cached_results))
        except Empty:
            pass

        # Submit new work if child is idle.
        if not self._detecting:
            try:
                items = [{"label": r.label, "crop": r.crop} for r in regions]
                self._regions_queue.put_nowait(items)
                self._pending_regions = list(regions)
                self._detecting = True
            except Exception:
                pass  # queue still full — skip this cycle

        return list(self._cached_results)

    # ------------------------------------------------------------------
    # Synchronous path — for tests only, not called in production
    # ------------------------------------------------------------------

    def _run_detection(self, crop: np.ndarray) -> list[TextLine]:
        """Run inference synchronously in the calling process.

        Used by unit tests to verify filtering logic without going through
        the child process. Monkeypatching TextDetection affects this path.

        Args:
            crop: BGR uint8 numpy array — a single region crop from Stage 4.

        Returns:
            List of detected TextLine objects in crop coordinates.
        """
        engine = TextDetection(model_name="PP-OCRv5_server_det")
        raw = engine.predict(crop)
        return _parse_lines(raw, crop)
