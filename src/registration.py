"""Stage 1 — Spatial Registration.

Detects the whiteboard quadrilateral via Canny edge detection and contour
analysis, computes a perspective homography, and warps every frame to a
canonical 1280×720 output. The homography is cached and only recomputed
when detected corners shift by more than ``cache_threshold`` pixels.

Typical usage::

    registrar = Registrar(debug=True)
    warped = registrar.process(frame)
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

OUTPUT_SIZE: tuple[int, int] = (1280, 720)


class Registrar:
    """Stateful perspective-correction stage.

    Keeps a cached homography so re-detection only runs when the board
    shifts significantly, keeping the per-frame cost low after lock-on.
    """

    def __init__(
        self,
        output_size: tuple[int, int] = OUTPUT_SIZE,
        cache_threshold: float = 20.0,
        debug: bool = False,
    ) -> None:
        """
        Args:
            output_size: (width, height) of the warped output image.
            cache_threshold: Maximum corner displacement in pixels before the
                homography is recomputed.
            debug: When True, draw the detected quad and corners onto the
                input frame before warping.
        """
        self._output_size = output_size
        self._cache_threshold = cache_threshold
        self.debug = debug

        self._homography: np.ndarray | None = None
        self._cached_corners: np.ndarray | None = None  # shape (4, 2) float32

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Warp *frame* to remove perspective distortion.

        Args:
            frame: BGR uint8 image captured from the camera.

        Returns:
            Perspective-corrected BGR uint8 image at ``output_size``.
            Falls back to a centred resize if no board can be detected and
            no cached homography is available.
        """
        corners = self._detect_corners(frame)

        if corners is not None:
            sorted_corners = self._sort_corners(corners)
            if self._cached_corners is None or self._corners_shifted(sorted_corners):
                self._homography = self._compute_homography(sorted_corners)
                self._cached_corners = sorted_corners
                logger.debug("Homography updated — new corners: %s", sorted_corners)

        if self.debug and corners is not None:
            frame = self._draw_debug(frame.copy(), corners)

        if self._homography is None:
            logger.debug("No board detected and no cached homography — resizing")
            return cv2.resize(frame, self._output_size)

        return cv2.warpPerspective(frame, self._homography, self._output_size)

    # ------------------------------------------------------------------
    # Detection helpers
    # ------------------------------------------------------------------

    def _detect_corners(self, frame: np.ndarray) -> np.ndarray | None:
        """Find the largest convex quadrilateral in *frame*.

        Returns:
            Corner array of shape ``(4, 2)`` as float32, or ``None`` if no
            suitable quadrilateral was found.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        frame_area = frame.shape[0] * frame.shape[1]
        min_area = frame_area * 0.10

        for contour in contours[:5]:
            if cv2.contourArea(contour) < min_area:
                break

            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            if len(approx) == 4 and cv2.isContourConvex(approx):
                return approx.reshape(4, 2).astype(np.float32)

        return None

    @staticmethod
    def _sort_corners(pts: np.ndarray) -> np.ndarray:
        """Return corners ordered as TL, TR, BR, BL.

        Args:
            pts: Array of shape ``(4, 2)`` (unsorted corners).

        Returns:
            Array of shape ``(4, 2)`` with corners in TL, TR, BR, BL order.
        """
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]   # TL: smallest x+y
        rect[2] = pts[np.argmax(s)]   # BR: largest x+y
        d = np.diff(pts, axis=1).ravel()
        rect[1] = pts[np.argmin(d)]   # TR: smallest y-x
        rect[3] = pts[np.argmax(d)]   # BL: largest y-x
        return rect

    def _compute_homography(self, corners: np.ndarray) -> np.ndarray:
        """Compute perspective transform from *corners* to the output rectangle.

        Args:
            corners: Sorted corner array (TL, TR, BR, BL), shape ``(4, 2)``.

        Returns:
            3×3 homography matrix as float64.
        """
        w, h = self._output_size
        dst = np.array(
            [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
            dtype=np.float32,
        )
        return cv2.getPerspectiveTransform(corners, dst)

    def _corners_shifted(self, sorted_corners: np.ndarray) -> bool:
        """Return True if any corner moved more than ``cache_threshold`` pixels.

        Args:
            sorted_corners: Newly detected corners already in TL, TR, BR, BL
                order so that the element-wise comparison against the cached
                sorted corners is meaningful.
        """
        if self._cached_corners is None:
            return True
        dists = np.linalg.norm(sorted_corners - self._cached_corners, axis=1)
        return bool(np.any(dists > self._cache_threshold))

    # ------------------------------------------------------------------
    # Debug overlay
    # ------------------------------------------------------------------

    def _draw_debug(self, frame: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """Draw corner markers and quad outline on *frame* in-place.

        Args:
            frame: BGR image to annotate (already a copy).
            corners: Detected corners, shape ``(4, 2)``.

        Returns:
            Annotated frame.
        """
        pts = corners.astype(np.int32)
        # Quad outline
        cv2.polylines(frame, [pts.reshape(-1, 1, 2)], isClosed=True, color=(0, 0, 220), thickness=2)
        # Corner markers + labels
        for i, (x, y) in enumerate(pts):
            cv2.circle(frame, (int(x), int(y)), 8, (0, 200, 0), -1)
            cv2.putText(
                frame,
                str(i),
                (int(x) + 10, int(y) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
        return frame


# ---------------------------------------------------------------------------
# Module-level convenience — delegates to a lazily-created global instance
# ---------------------------------------------------------------------------

_global_registrar: Registrar | None = None


def process(frame: np.ndarray) -> np.ndarray:
    """Warp *frame* using a module-global :class:`Registrar`.

    Args:
        frame: BGR uint8 image captured from the camera.

    Returns:
        Perspective-corrected BGR uint8 image at the canonical board resolution.
    """
    global _global_registrar
    if _global_registrar is None:
        _global_registrar = Registrar()
    return _global_registrar.process(frame)
