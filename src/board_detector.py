"""Stage 1 — Board Detection.

Locates the whiteboard in the raw camera frame using SAM 3 (Segment Anything
Model 3) with a text prompt ("whiteboard"), extracts the board boundary from
the resulting binary mask, and maintains a cache of the four corner points.

SAM 3 re-runs every ``recompute_interval`` seconds on a background thread so
the main processing loop never blocks on inference. Corner updates are filtered
by a geometric guard: a new detection only replaces the cache when the area
is not shrinking (occlusion guard), the displacement passes the threshold, and
single-corner shifts are only accepted if they improve the board's symmetry.

Models:
  ``sam3.1_multiplex.pt``   — whiteboard detection (auto-downloaded by Ultralytics).

Typical usage::

    detector = BoardDetector()
    # each frame:
    detector.submit_frame(frame)
    corners = detector.get_corners()   # non-blocking, returns cached value
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics.models.sam import SAM3SemanticPredictor

logger = logging.getLogger(__name__)


class BoardDetector:
    """Stateful whiteboard-localization stage backed by SAM 3.

    SAM 3 fires asynchronously every ``recompute_interval`` seconds.
    ``get_corners()`` never blocks — it returns the most recently accepted
    four-corner array, or ``None`` before any successful detection.
    """

    def __init__(
        self,
        recompute_interval: float = 5.0,
        cache_threshold: float = 50.0,
        sam_model: str = "sam3.1_multiplex.pt",
    ) -> None:
        """
        Args:
            recompute_interval: Seconds between SAM 3 re-detection runs.
            cache_threshold: Corner displacement in pixels used by the
                geometric update filter. All four corners must exceed this
                (or a single-corner improvement check must pass) to commit
                a new detection.
            sam_model: Ultralytics model filename for SAM 3 detection.
        """
        self._recompute_interval = recompute_interval
        self._cache_threshold = cache_threshold

        _model = Path(__file__).parent.parent / "models" / sam_model
        self._sam: SAM3SemanticPredictor = SAM3SemanticPredictor(
            overrides=dict(
                model=str(_model),
                task="segment",
                mode="predict",
                imgsz=644,  # nearest multiple of SAM 3's stride-14 above default 640
                save=False,
                verbose=False,
            )
        )

        # Set to 0 so the very first submit_frame() call fires SAM 3 immediately.
        self._last_detect_time: float = 0.0

        self._cached_corners: np.ndarray | None = None  # (4, 2) float32, TL/TR/BR/BL
        self._lock = threading.Lock()

        self._detecting: bool = False
        self._pending_frame: np.ndarray | None = None
        self._detect_event = threading.Event()
        self._detect_thread = threading.Thread(
            target=self._detection_loop,
            daemon=True,
            name="sam3-detect",
        )
        self._detect_thread.start()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def submit_frame(self, frame: np.ndarray) -> None:
        """Trigger an async SAM 3 detection if the recompute interval has elapsed.

        Safe to call every frame — detection only fires at most once per
        ``recompute_interval`` seconds and does not block the caller.

        Args:
            frame: BGR uint8 raw camera frame.
        """
        now = time.monotonic()
        if not self._detecting and (now - self._last_detect_time) >= self._recompute_interval:
            self._last_detect_time = now
            self._pending_frame = frame
            self._detecting = True
            self._detect_event.set()

    def get_corners(self) -> np.ndarray | None:
        """Return the most recently accepted board corners without blocking.

        Returns:
            Float32 array of shape ``(4, 2)`` ordered TL/TR/BR/BL, or
            ``None`` if no successful detection has occurred yet.
        """
        with self._lock:
            return self._cached_corners

    # ------------------------------------------------------------------
    # Background detection thread
    # ------------------------------------------------------------------

    def _detection_loop(self) -> None:
        """Daemon thread: runs SAM 3 periodically and updates the corner cache."""
        while True:
            self._detect_event.wait()
            self._detect_event.clear()

            frame = self._pending_frame
            if frame is None:
                self._detecting = False
                continue

            try:
                corners = self._detect_corners(frame)
                if corners is not None:
                    sorted_corners = self._sort_corners(corners)
                    with self._lock:
                        if self._cached_corners is None or self._are_corners_shifted(
                            sorted_corners
                        ):
                            self._cached_corners = sorted_corners
                            logger.debug("Board corners updated: %s", sorted_corners)
            except Exception:
                logger.exception("SAM 3 detection failed")
            finally:
                self._detecting = False

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _detect_corners(self, frame: np.ndarray) -> np.ndarray | None:
        """Run SAM 3 with a text prompt and return the board quad.

        Args:
            frame: BGR uint8 raw camera frame.

        Returns:
            Corner array of shape ``(4, 2)`` as float32, or ``None``.
        """
        results = self._sam(frame, text=["whiteboard"])

        if not results or results[0].masks is None:
            logger.debug("SAM 3 returned no masks for 'whiteboard'")
            return None

        masks = results[0].masks.data.cpu().numpy()
        if masks.shape[0] == 0:
            logger.debug("SAM 3 returned empty mask tensor")
            return None

        areas = masks.sum(axis=(1, 2))
        best = (masks[areas.argmax()] > 0.5).astype(np.uint8)
        return self._mask_to_corners(best)

    @staticmethod
    def _mask_to_corners(mask: np.ndarray) -> np.ndarray | None:
        """Extract a 4-corner quadrilateral from a binary segmentation mask.

        Takes the convex hull of the largest contour in *mask*, then sweeps
        the ``approxPolyDP`` epsilon until the polygon collapses to 4 vertices.

        Args:
            mask: Binary uint8 mask, shape ``(H, W)``, values 0 or 1.

        Returns:
            Corner array of shape ``(4, 2)`` as float32, or ``None`` if no
            suitable quadrilateral can be found.
        """
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

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sort_corners(pts: np.ndarray) -> np.ndarray:
        """Return corners ordered as TL, TR, BR, BL."""
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        d = np.diff(pts, axis=1).ravel()
        rect[1] = pts[np.argmin(d)]
        rect[3] = pts[np.argmax(d)]
        return rect

    def _are_corners_shifted(self, sorted_corners: np.ndarray) -> bool:
        """Return True if corners have shifted enough to warrant a cache update.

        Prevents spurious updates during occlusion (area guard) and accepts
        single-corner shifts only when they improve the board's diagonal symmetry.
        Called from inside ``_lock`` — reads ``self._cached_corners`` directly.
        """
        if self._cached_corners is None:
            return True

        def get_area(p: np.ndarray) -> float:
            return 0.5 * abs(
                np.dot(p[:, 0], np.roll(p[:, 1], 1))
                - np.dot(p[:, 1], np.roll(p[:, 0], 1))
            )

        def get_ratio(p: np.ndarray) -> float:
            return np.linalg.norm(p[0] - p[2]) / (np.linalg.norm(p[1] - p[3]) + 1e-6)

        new_a, old_a = get_area(sorted_corners), get_area(self._cached_corners)
        new_r, old_r = get_ratio(sorted_corners), get_ratio(self._cached_corners)

        # Occlusion guard: reject if detected area shrank (professor covering corners)
        if new_a < old_a * 0.98:
            return False

        dists = np.linalg.norm(sorted_corners - self._cached_corners, axis=1)
        count = np.count_nonzero(dists > self._cache_threshold)

        # Single-corner shift: only accept if it makes the board more symmetric
        if count == 1:
            return abs(new_r - 1.0) < abs(old_r - 1.0)

        # Multiple corners moved → accept; zero corners moved → reject (noise)
        return count >= 2
