"""Stage 1+2 — Board Tracking & Body Matting.

Replaces board_detector.py (BoardDetector) and person_masker.py (PersonMasker).

A single SAM 3.1 multiprocessing worker handles both tasks per frame:
  - Board corners via text prompt "whiteboard"
  - Body + shadow mask via text prompts "person", "arm", "hand", "shadow"

process() is always non-blocking. The main loop submits the latest frame via a
maxsize=1 queue (old frames are dropped when the worker is busy) and reads the
latest TrackerResult from the output queue, falling back to the cached result
when no new result is ready yet.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "sam3.1_multiplex.pt"

_BODY_PROMPTS = ["person", "arm", "hand", "shadow"]


@dataclass
class TrackerResult:
    corners: np.ndarray | None = None
    """(4, 2) float32, ordered TL / TR / BR / BL. None until first detection."""
    body_mask: np.ndarray | None = None
    """uint8 H×W, value 1 = occluder or shadow, 0 = board. None until first frame."""
    frame: np.ndarray | None = None
    """The exact frame SAM analyzed. Non-None only on fresh results (not cached repeats).
    Use this — not the current camera frame — as input to Stage 4 so mask and frame match."""


# ---------------------------------------------------------------------------
# Worker process
# ---------------------------------------------------------------------------


def _worker_main(
    in_q: mp.Queue,
    out_q: mp.Queue,
    model_path: str,
    dilation_px: int,
) -> None:
    """SAM 3.1 inference loop — runs in a dedicated child process."""
    import logging as _log

    _log.basicConfig(level=logging.WARNING)

    from ultralytics.models.sam import SAM3SemanticPredictor

    sam = SAM3SemanticPredictor(
        overrides=dict(
            model=model_path,
            task="segment",
            mode="predict",
            imgsz=644,
            save=False,
            verbose=False,
        )
    )

    kernel: np.ndarray | None = None
    if dilation_px > 0:
        ksize = 2 * dilation_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

    cached_corners: np.ndarray | None = None

    while True:
        frame = in_q.get()  # block until a frame arrives
        if frame is None:   # shutdown sentinel
            break

        h, w = frame.shape[:2]
        body_mask = np.zeros((h, w), dtype=np.uint8)

        # --- Board corners ---
        try:
            board_res = sam(frame, text=["whiteboard"])
            if board_res and board_res[0].masks is not None:
                masks = board_res[0].masks.data.cpu().numpy()
                if masks.shape[0] > 0:
                    areas = masks.sum(axis=(1, 2))
                    best = (masks[areas.argmax()] > 0.5).astype(np.uint8)
                    new_c = _mask_to_corners(best)
                    if new_c is not None:
                        sorted_c = _sort_corners(new_c)
                        if cached_corners is None or _are_corners_shifted(sorted_c, cached_corners):
                            cached_corners = sorted_c
        except Exception:
            logging.getLogger(__name__).exception("SAM board detection failed")

        # --- Body + shadow mask ---
        try:
            body_res = sam(frame, text=_BODY_PROMPTS)
            if body_res and body_res[0].masks is not None:
                masks = body_res[0].masks.data.cpu().numpy()
                if masks.shape[0] > 0:
                    union = np.zeros((h, w), dtype=np.uint8)
                    for m in masks:
                        union |= (m > 0.5).astype(np.uint8)
                    if kernel is not None:
                        union = cv2.dilate(union, kernel, iterations=1)
                    body_mask = union
        except Exception:
            logging.getLogger(__name__).exception("SAM body detection failed")

        result = TrackerResult(corners=cached_corners, body_mask=body_mask, frame=frame)

        # Replace any unconsumed result so the consumer always gets the freshest data
        try:
            out_q.get_nowait()
        except Exception:
            pass
        try:
            out_q.put_nowait(result)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Geometry helpers (ported from board_detector.py)
# ---------------------------------------------------------------------------


def _mask_to_corners(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    peri = cv2.arcLength(hull, True)
    for eps in (0.02, 0.04, 0.06, 0.08, 0.10):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
    return None


def _sort_corners(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1).ravel()
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def _are_corners_shifted(
    new: np.ndarray,
    cached: np.ndarray,
    threshold: float = 50.0,
) -> bool:
    def _area(p: np.ndarray) -> float:
        return 0.5 * abs(
            np.dot(p[:, 0], np.roll(p[:, 1], 1))
            - np.dot(p[:, 1], np.roll(p[:, 0], 1))
        )

    def _ratio(p: np.ndarray) -> float:
        return np.linalg.norm(p[0] - p[2]) / (np.linalg.norm(p[1] - p[3]) + 1e-6)

    if _area(new) < _area(cached) * 0.98:
        return False

    dists = np.linalg.norm(new - cached, axis=1)
    count = int(np.count_nonzero(dists > threshold))
    if count == 1:
        return abs(_ratio(new) - 1.0) < abs(_ratio(cached) - 1.0)
    return count >= 2


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class BoardTracker:
    """Non-blocking SAM 3.1 board tracker and body masker.

    Spawns a single child process that runs SAM 3.1 for both board corner
    detection and body/shadow segmentation. process() always returns immediately
    with the latest available TrackerResult.
    """

    def __init__(
        self,
        model_path: Path = _MODEL_PATH,
        dilation_px: int = 5,
    ) -> None:
        self._cached = TrackerResult()
        self._in_q: mp.Queue = mp.Queue(maxsize=1)
        self._out_q: mp.Queue = mp.Queue(maxsize=1)
        self._worker = mp.Process(
            target=_worker_main,
            args=(self._in_q, self._out_q, str(model_path), dilation_px),
            daemon=True,
            name="sam3-tracker",
        )
        self._worker.start()
        logger.info("BoardTracker worker started (pid=%d)", self._worker.pid)

    def process(self, frame: np.ndarray) -> TrackerResult:
        """Submit *frame* for async SAM inference; return latest result.

        Non-blocking. ``result.frame`` is non-None only when SAM just produced a
        fresh result — use it (not the current camera frame) to drive Stage 4 so
        the body mask and the frame it describes are always matched.
        """
        try:
            self._in_q.put_nowait(frame)
        except Exception:
            pass  # worker busy — drop this frame

        fresh = None
        try:
            fresh = self._out_q.get_nowait()
        except Exception:
            pass  # no new result yet

        if fresh is not None:
            self._cached = fresh
            return fresh  # frame field is set

        # Return cached result but clear frame so caller knows this is not fresh
        return TrackerResult(
            corners=self._cached.corners,
            body_mask=self._cached.body_mask,
            frame=None,
        )

    def shutdown(self) -> None:
        """Signal the worker to stop and wait for it to exit."""
        try:
            self._in_q.put_nowait(None)
        except Exception:
            pass
        self._worker.join(timeout=5)
        if self._worker.is_alive():
            self._worker.terminate()
        logger.info("BoardTracker worker stopped")
