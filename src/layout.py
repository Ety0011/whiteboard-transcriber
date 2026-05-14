"""Stage 4 — Layout Detection.

Takes a clean board composite (BGR uint8 numpy array from Stage 3) and returns
a list of detected regions with bounding boxes, class labels, confidence scores,
and cropped image arrays.

Model: PaddleOCR PP-DocLayout_plus-L via LayoutDetection.
Load once at startup via init() or the first call to process() — never per frame.

Inference runs in a dedicated child process (multiprocessing.Process), not a
thread. This is necessary because PaddlePaddle holds the Python GIL during
inference, which blocks cv2.waitKey on the main thread even when using threading.
The child process has its own GIL and never contends with the main process.

process() is non-blocking: it submits the image at most every recompute_interval
seconds, polls the result queue, and immediately returns the last cached result.
"""

from __future__ import annotations

import dataclasses
import logging
import multiprocessing as mp
import time
from multiprocessing.queues import Empty

import numpy as np
from paddleocr import LayoutDetection

log = logging.getLogger(__name__)

_WHITEBOARD_LABELS = {
    "text",
    "paragraph_title",
    "table",
    "image",
    "figure",
    "figure_table_title",
    "formula",
    "formula_number",
    "algorithm",
    "chart",
}


@dataclasses.dataclass
class LayoutRegion:
    """A detected layout region on the whiteboard."""

    bbox: np.ndarray  # shape (4,) int32: x1, y1, x2, y2
    label: str
    confidence: float
    crop: np.ndarray  # BGR uint8, same dtype as input


# ---------------------------------------------------------------------------
# Child-process helpers (module-level so they are picklable)
# ---------------------------------------------------------------------------


def _inference_worker(
    img_queue: mp.Queue,
    result_queue: mp.Queue,
    confidence_threshold: float,
) -> None:
    """Entry point for the child process.

    Owns PaddlePaddle entirely — the main process never touches it.
    Reads images from img_queue, runs inference, puts serialisable box dicts
    into result_queue. Runs until it receives None as a sentinel.
    """
    engine = LayoutDetection(
        model_name="PP-DocLayout_plus-L",
        engine="transformers",
        layout_nms=True,
        layout_merge_bboxes_mode="large",
        layout_unclip_ratio=1.05,
        threshold=confidence_threshold,
    )
    while True:
        image: np.ndarray = img_queue.get()
        if image is None:
            break
        try:
            raw = engine.predict(image, layout_nms=True)
            boxes = _filter_boxes(raw, confidence_threshold, image.shape)
        except Exception:
            boxes = []
        result_queue.put(boxes)


def _filter_boxes(
    raw_results: list,
    confidence_threshold: float,
    img_shape: tuple[int, ...],
) -> list[dict]:
    """Filter and normalise raw PP-DocLayout output into plain serialisable dicts."""
    if not raw_results:
        return []
    h, w = img_shape[:2]
    out: list[dict] = []
    for box in raw_results[0].get("boxes", []):
        label: str = box.get("label", "")
        confidence: float = float(box.get("score", 1.0))
        coord = box.get("coordinate", [])

        if label not in _WHITEBOARD_LABELS:
            continue
        if confidence < confidence_threshold:
            continue
        if len(coord) < 4:
            continue

        x1 = max(0, int(coord[0]))
        y1 = max(0, int(coord[1]))
        x2 = min(w, int(coord[2]))
        y2 = min(h, int(coord[3]))
        out.append({"label": label, "confidence": confidence, "bbox": (x1, y1, x2, y2)})
    return out


def _build_regions(box_dicts: list[dict], image: np.ndarray) -> list[LayoutRegion]:
    """Reconstruct LayoutRegion objects (with crops) from serialisable box dicts."""
    regions: list[LayoutRegion] = []
    for d in box_dicts:
        x1, y1, x2, y2 = d["bbox"]
        crop = image[y1:y2, x1:x2].copy()
        regions.append(
            LayoutRegion(
                bbox=np.array([x1, y1, x2, y2], dtype=np.int32),
                label=d["label"],
                confidence=d["confidence"],
                crop=crop,
            )
        )
    return regions


# ---------------------------------------------------------------------------
# LayoutDetector
# ---------------------------------------------------------------------------


class LayoutDetector:
    """Runs PP-DocLayout in a child process so the main GIL is never blocked.

    Same external pattern as Stage 1 Registrar / SAM3:
    - process() never blocks — returns cached regions immediately.
    - A new detection is submitted at most every recompute_interval seconds.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.4,
        recompute_interval: float = 2.0,
    ) -> None:
        """Start the child process and load PP-DocLayout inside it.

        Args:
            confidence_threshold: Detections below this score are discarded.
            recompute_interval: Minimum seconds between successive detections.
        """
        self._confidence_threshold = confidence_threshold
        self._recompute_interval = recompute_interval
        self._last_detect_time: float = 0.0

        self._cached_regions: list[LayoutRegion] = []
        self._detecting = False
        self._pending_image: np.ndarray | None = None

        self._img_queue: mp.Queue = mp.Queue(maxsize=1)
        self._result_queue: mp.Queue = mp.Queue(maxsize=1)
        self._worker = mp.Process(
            target=_inference_worker,
            args=(self._img_queue, self._result_queue, confidence_threshold),
            daemon=True,
            name="layout-detect",
        )
        self._worker.start()
        log.info(
            "LayoutDetector initialised (threshold=%.2f, interval=%.1fs)",
            confidence_threshold,
            recompute_interval,
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def process(self, image: np.ndarray) -> list[LayoutRegion]:
        """Submit *image* to the child process and return the last cached result.

        Never blocks. Polls the result queue on every call; submits a new image
        when the child is idle and recompute_interval seconds have elapsed.

        Args:
            image: BGR uint8 numpy array (H×W×3) from Stage 3.

        Returns:
            Last detected list of LayoutRegion objects (empty until first detection).
        """
        # Poll for completed result from child process (non-blocking).
        try:
            box_dicts = self._result_queue.get_nowait()
            if self._pending_image is not None:
                self._cached_regions = _build_regions(box_dicts, self._pending_image)
            self._detecting = False
            log.debug("Layout result received: %d regions", len(self._cached_regions))
        except Empty:
            pass

        # Submit a new image if the child is idle and interval has elapsed.
        now = time.monotonic()
        if (
            not self._detecting
            and (now - self._last_detect_time) >= self._recompute_interval
        ):
            try:
                img_copy = image.copy()
                self._img_queue.put_nowait(img_copy)
                self._pending_image = img_copy
                self._detecting = True
                self._last_detect_time = now
            except Exception:
                pass  # queue still full — skip this frame

        return list(self._cached_regions)

    # ------------------------------------------------------------------
    # Synchronous path — for tests only, not called in production
    # ------------------------------------------------------------------

    def _run_detection(self, image: np.ndarray) -> list[LayoutRegion]:
        """Run inference synchronously in the calling process.

        Used by unit tests to verify filtering logic without going through
        the child process. Monkeypatching LayoutDetection affects this path.
        """
        engine = LayoutDetection(
            model_name="PP-DocLayout_plus-L",
            engine="transformers",
            layout_nms=True,
            layout_merge_bboxes_mode="large",
            layout_unclip_ratio=1.05,
            threshold=self._confidence_threshold,
        )
        raw = engine.predict(image, layout_nms=True)
        box_dicts = _filter_boxes(raw, self._confidence_threshold, image.shape)
        return _build_regions(box_dicts, image)
