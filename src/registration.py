"""Stage 1 — Spatial Registration.

Detects the whiteboard quadrilateral using SAM 3 (Segment Anything Model 3)
with a text prompt ("whiteboard"), extracts the board boundary from the
resulting binary mask, and warps every frame to a canonical 1280×720 output.

SAM 3 re-runs every ``recompute_every`` frames. A re-detection only updates
the homography when **all four corners** have moved beyond ``cache_threshold``
pixels from their previously cached positions — single-corner drift is treated
as noise and ignored.

Models:
  ``sam3-l.pt``   — whiteboard detection (auto-downloaded by Ultralytics).

Typical usage::

    registrar = Registrar()
    warped = registrar.process(frame)
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


class Registrar:
    """Stateful perspective-correction stage backed by SAM 3 board detection.

    SAM 3 re-runs every ``recompute_interval`` seconds. Between runs the
    cached homography is reused, keeping per-frame cost minimal. A new
    homography is committed only when all four detected corners have shifted
    beyond ``cache_threshold`` pixels — partial drift (one or two corners) is
    discarded as measurement noise.
    """

    def __init__(
        self,
        output_size: tuple[int, int] = (1920, 1080),
        cache_threshold: float = 50.0,
        recompute_interval: float = 5.0,
        sam_model: str = "sam3.1_multiplex.pt",
    ) -> None:
        """
        Args:
            output_size: (width, height) of the warped output image.
            cache_threshold: Corner displacement in pixels. All four corners
                must exceed this to trigger a homography update.
            recompute_interval: Seconds between SAM 3 re-detection runs.
            sam_model: Ultralytics model filename for SAM 3 detection.
        """
        self._output_size = output_size
        self._cache_threshold = cache_threshold
        self._recompute_interval = recompute_interval

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

        # Set to 0 so the very first process() call fires SAM 3 immediately.
        self._last_detect_time: float = 0.0

        self._homography: np.ndarray | None = None
        self._cached_corners: np.ndarray | None = None  # (4, 2) float32, TL/TR/BR/BL
        self._lock = threading.Lock()  # guards _homography and _cached_corners

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

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Warp *frame* to remove perspective distortion.

        SAM 3 detection is fired asynchronously on a background thread every
        ``recompute_interval`` seconds; this method never blocks on inference.

        Args:
            frame: BGR uint8 image captured from the camera.

        Returns:
            Perspective-corrected BGR uint8 image at ``output_size``.
            Falls back to a centred resize if no board has ever been detected.
        """
        now = time.monotonic()
        elapsed = now - self._last_detect_time
        if not self._detecting and elapsed >= self._recompute_interval:
            self._last_detect_time = now
            self._pending_frame = frame
            self._detecting = True
            self._detect_event.set()

        with self._lock:
            homography = self._homography

        if homography is None:
            logger.debug("No board detected yet — returning resized frame")
            return cv2.resize(frame, self._output_size)

        return cv2.warpPerspective(frame, homography, self._output_size)

    # ------------------------------------------------------------------
    # Background detection thread
    # ------------------------------------------------------------------

    def _detection_loop(self) -> None:
        """Daemon thread: runs SAM 3 periodically and updates the homography."""
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
                            self._homography = self._compute_homography(sorted_corners)
                            self._cached_corners = sorted_corners
                            logger.debug(
                                "Homography updated — corners: %s", sorted_corners
                            )
            except Exception:
                logger.exception("SAM 3 detection failed")
            finally:
                self._detecting = False

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _detect_corners(self, frame: np.ndarray) -> np.ndarray | None:
        """Run SAM 3 with a text prompt and return the board quad.

        Uses SAM3SemanticPredictor with the concept prompt "whiteboard" so
        detection is location-independent — no point prompts needed.

        Args:
            frame: BGR uint8 camera frame.

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

    def _compute_homography(self, corners: np.ndarray) -> np.ndarray:
        """Compute perspective transform from *corners* to the output rectangle."""
        w, h = self._output_size
        dst = np.array(
            [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
            dtype=np.float32,
        )
        return cv2.getPerspectiveTransform(corners, dst)

    def _are_corners_shifted(self, sorted_corners: np.ndarray) -> bool:
        """Return True if 2-3 corners shift, or 1 corner shifts and improves ratio.

        Prevents updates during occlusion via area guard and ensures the board
        converges toward a rectangular shape using an absolute diagonal ratio.
        """
        if self._cached_corners is None:
            return True

        def get_area(p):
            return 0.5 * np.abs(
                np.dot(p[:, 0], np.roll(p[:, 1], 1))
                - np.dot(p[:, 1], np.roll(p[:, 0], 1))
            )

        def get_ratio(p):
            return np.linalg.norm(p[0] - p[2]) / (np.linalg.norm(p[1] - p[3]) + 1e-6)

        # 1. Geometry Metrics
        new_a, old_a = get_area(sorted_corners), get_area(self._cached_corners)
        new_r, old_r = get_ratio(sorted_corners), get_ratio(self._cached_corners)

        # Area Guard: Ignore if the professor occludes corners (shrinking)
        if new_a < old_a * 0.98:
            return False

        # 2. Movement Logic
        dists = np.linalg.norm(sorted_corners - self._cached_corners, axis=1)
        count = np.count_nonzero(dists > self._cache_threshold)

        # If exactly one corner shifts, only accept if it makes the board more symmetric
        if count == 1:
            return abs(new_r - 1.0) < abs(old_r - 1.0)

        # Accept multiple corners change
        return True
