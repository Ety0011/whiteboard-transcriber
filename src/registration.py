"""Stage 1 — Spatial Registration.

Detects the whiteboard quadrilateral using SAM2 (Segment Anything Model 2)
with a centre-of-frame point prompt, extracts the board boundary from the
resulting binary mask, and warps every frame to a canonical 1280×720 output.

SAM2 segments by semantic boundary rather than by edge contrast, so it is
robust to board content (markers, diagrams) and lighting variation that would
fool a Canny-based detector.

The homography is cached and SAM2 is re-run every ``recompute_every`` calls
to ``process()``, keeping the per-frame cost near zero between re-detections.

Model: ``sam2.1_t.pt`` (~38 MB, auto-downloaded by Ultralytics on first run).

Typical usage::

    registrar = Registrar()
    warped = registrar.process(frame)
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from ultralytics import SAM

logger = logging.getLogger(__name__)

OUTPUT_SIZE: tuple[int, int] = (1280, 720)


class Registrar:
    """Stateful perspective-correction stage backed by SAM2 board detection.

    SAM2 runs only once every ``recompute_every`` pipeline cycles; between
    runs the cached homography is reused, keeping per-frame cost minimal.
    """

    def __init__(
        self,
        output_size: tuple[int, int] = OUTPUT_SIZE,
        cache_threshold: float = 20.0,
        recompute_every: int = 30,
        debug: bool = False,
    ) -> None:
        """
        Args:
            output_size: (width, height) of the warped output image.
            cache_threshold: Maximum corner displacement in pixels before the
                homography is recomputed even within a recompute cycle.
            recompute_every: Run SAM2 detection every this many ``process()``
                calls. Default 30 ≈ 30–60 s at pipeline speed.
            debug: When True, draw the detected quad and corners onto the
                raw input frame (shown in the debug window of debug_view.py).
        """
        self._output_size = output_size
        self._cache_threshold = cache_threshold
        self._recompute_every = recompute_every
        self.debug = debug

        self._sam = SAM("sam2.1_t.pt")
        # Start counter at recompute_every so detection fires on the first call.
        self._calls_since_detect: int = recompute_every

        self._homography: np.ndarray | None = None
        self._cached_corners: np.ndarray | None = None  # (4, 2) float32, TL/TR/BR/BL

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Warp *frame* to remove perspective distortion.

        Args:
            frame: BGR uint8 image captured from the camera.

        Returns:
            Perspective-corrected BGR uint8 image at ``output_size``.
            Falls back to a centred resize if no board has ever been detected.
        """
        self._calls_since_detect += 1
        should_detect = (
            self._homography is None
            or self._calls_since_detect >= self._recompute_every
        )

        if should_detect:
            self._calls_since_detect = 0
            corners = self._detect_corners(frame)
            if corners is not None:
                sorted_corners = self._sort_corners(corners)
                if self._cached_corners is None or self._corners_shifted(sorted_corners):
                    self._homography = self._compute_homography(sorted_corners)
                    self._cached_corners = sorted_corners
                    logger.debug("Homography updated — corners: %s", sorted_corners)

        if self.debug and self._cached_corners is not None:
            frame = self._draw_debug(frame.copy(), self._cached_corners)

        if self._homography is None:
            logger.debug("No board detected yet — returning resized frame")
            return cv2.resize(frame, self._output_size)

        return cv2.warpPerspective(frame, self._homography, self._output_size)

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _detect_corners(self, frame: np.ndarray) -> np.ndarray | None:
        """Run SAM2 with a centre-point prompt and return the board quad.

        Args:
            frame: BGR uint8 camera frame.

        Returns:
            Corner array of shape ``(4, 2)`` as float32, or ``None``.
        """
        h, w = frame.shape[:2]
        results = self._sam(
            frame,
            points=[[w // 2, h // 2]],
            labels=[1],  # 1 = foreground point
            verbose=False,
        )

        if not results or results[0].masks is None:
            logger.debug("SAM2 returned no masks")
            return None

        # SAM2 may return multiple candidate masks — take the largest area.
        masks = results[0].masks.data.cpu().numpy()  # (N, H, W) float
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
    # Geometry helpers (unchanged)
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

    def _corners_shifted(self, sorted_corners: np.ndarray) -> bool:
        """Return True if any corner moved more than ``cache_threshold`` pixels."""
        if self._cached_corners is None:
            return True
        dists = np.linalg.norm(sorted_corners - self._cached_corners, axis=1)
        return bool(np.any(dists > self._cache_threshold))

    # ------------------------------------------------------------------
    # Debug overlay (unchanged)
    # ------------------------------------------------------------------

    def _draw_debug(self, frame: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """Draw corner markers and quad outline on *frame*."""
        pts = corners.astype(np.int32)
        cv2.polylines(
            frame, [pts.reshape(-1, 1, 2)], isClosed=True, color=(0, 0, 220), thickness=2
        )
        for i, (x, y) in enumerate(pts):
            cv2.circle(frame, (int(x), int(y)), 8, (0, 200, 0), -1)
            cv2.putText(
                frame, str(i), (int(x) + 10, int(y) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            )
        return frame


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_global_registrar: Registrar | None = None


def process(frame: np.ndarray) -> np.ndarray:
    """Warp *frame* using a module-global :class:`Registrar`."""
    global _global_registrar
    if _global_registrar is None:
        _global_registrar = Registrar()
    return _global_registrar.process(frame)
