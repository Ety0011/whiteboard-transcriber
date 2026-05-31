"""Stage 4 — Perspective Correction.

Warps the raw camera frame and person mask to a canonical 1920×1080 view.
Accepts the latest board mask from BoardSegmenter; derives corners from it,
computes and caches the homography, then warps both inputs every frame.
Falls back to a simple resize before the first SAM result arrives.

All geometry (corner extraction, homography) lives here — upstream stages
produce masks only.
"""

from __future__ import annotations

import cv2
import numpy as np

from stage import InlineStage


# ---------------------------------------------------------------------------
# Geometry helpers (operate on raw-space masks and corner arrays)
# ---------------------------------------------------------------------------


def _mask_to_corners(mask: np.ndarray) -> np.ndarray | None:
    """Extract four corners from a binary board mask, or return None."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    perimeter = cv2.arcLength(hull, True)
    for eps in (0.02, 0.04, 0.06, 0.08, 0.10):
        approx = cv2.approxPolyDP(hull, eps * perimeter, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
    return None


def _sort_corners(pts: np.ndarray) -> np.ndarray:
    """Reorder four corner points to TL / TR / BR / BL."""
    sorted_corners = np.zeros((4, 2), dtype=np.float32)
    coord_sum = pts.sum(axis=1)
    sorted_corners[0] = pts[np.argmin(coord_sum)]
    sorted_corners[2] = pts[np.argmax(coord_sum)]
    coord_diff = np.diff(pts, axis=1).ravel()
    sorted_corners[1] = pts[np.argmin(coord_diff)]
    sorted_corners[3] = pts[np.argmax(coord_diff)]
    return sorted_corners


def _quad_area(p: np.ndarray) -> float:
    """Return the area of a quad via the shoelace formula."""
    return 0.5 * abs(
        np.dot(p[:, 0], np.roll(p[:, 1], 1)) - np.dot(p[:, 1], np.roll(p[:, 0], 1))
    )


def _quad_ratio(p: np.ndarray) -> float:
    """Return the diagonal aspect ratio (TL-BR length / TR-BL length) of a quad."""
    return np.linalg.norm(p[0] - p[2]) / (np.linalg.norm(p[1] - p[3]) + 1e-6)


def _are_corners_shifted(
    new: np.ndarray,
    cached: np.ndarray,
    threshold: float = 50.0,
) -> bool:
    """Return True if *new* corners represent a meaningful shift from *cached*."""
    if _quad_area(new) < _quad_area(cached) * 0.98:
        return False

    dists = np.linalg.norm(new - cached, axis=1)
    count = int(np.count_nonzero(dists > threshold))
    if count == 1:
        return abs(_quad_ratio(new) - 1.0) < abs(_quad_ratio(cached) - 1.0)
    return count >= 2


def _compute_homography(
    corners: np.ndarray,
    output_size: tuple[int, int],
) -> np.ndarray:
    """Compute perspective transform from *corners* to the output rectangle.

    Args:
        corners:     (4, 2) float32 source corners in TL/TR/BR/BL order.
        output_size: (width, height) of the destination rectangle.

    Returns:
        3×3 float64 perspective transform matrix.
    """
    w, h = output_size
    dst = np.array(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(corners, dst)


# ---------------------------------------------------------------------------
# Rectifier
# ---------------------------------------------------------------------------


class Rectifier(InlineStage):
    """Stateful perspective-rectification stage.

    Each time a new board mask arrives, corners are extracted and the
    homography is recomputed and cached. Every frame the cached homography
    is applied to warp the raw frame and person mask to the canonical output
    size. Falls back to a centred resize until the first board mask arrives.

    Output buffers are pre-allocated to avoid per-frame heap allocation.

    Args:
        output_size: (width, height) of the rectified output images.
    """

    def __init__(self, output_size: tuple[int, int] = (1920, 1080)) -> None:
        super().__init__(interval_s=0.0)
        self._output_size = output_size
        self._homography: np.ndarray | None = None
        self._cached_corners: np.ndarray | None = None  # (4, 2) float32, TL/TR/BR/BL

        w, h = output_size
        self._rect_frame = np.empty((h, w, 3), dtype=np.uint8)
        self._rect_mask = np.empty((h, w), dtype=np.uint8)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def cached_corners(self) -> np.ndarray | None:
        """Latest derived corners in raw-frame space (debug overlay use only)."""
        return self._cached_corners

    def rectify(
        self,
        frame: np.ndarray,
        board_mask: np.ndarray | None,
        person_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Warp *frame* and *person_mask* to remove perspective distortion.

        Args:
            frame: BGR uint8 raw camera frame.
            board_mask: uint8 H×W board segmentation from BoardSegmenter, or None
                when SAM has not produced a fresh result this cycle. When
                non-None, corners are re-derived and the homography updated.
            person_mask: uint8 H×W person mask from PersonSegmenter (always fresh).

        Returns:
            Tuple ``(rect_frame, rect_mask)`` both at ``output_size``.
            Falls back to a simple resize when no homography is cached yet.
        """
        if board_mask is not None:
            self._update_homography(board_mask)

        w, h = self._output_size

        if self._homography is None:
            cv2.resize(frame, (w, h), dst=self._rect_frame)
            cv2.resize(person_mask, (w, h), dst=self._rect_mask,
                       interpolation=cv2.INTER_NEAREST)
        else:
            cv2.warpPerspective(frame, self._homography, (w, h),
                                dst=self._rect_frame)
            cv2.warpPerspective(person_mask, self._homography, (w, h),
                                dst=self._rect_mask, flags=cv2.INTER_NEAREST)

        return self._rect_frame, self._rect_mask

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _update_homography(self, board_mask: np.ndarray) -> None:
        """Derive corners from *board_mask* and recompute homography if shifted."""
        corners = _mask_to_corners(board_mask)
        if corners is None:
            return
        sorted_c = _sort_corners(corners)
        is_first = self._cached_corners is None
        if is_first or _are_corners_shifted(sorted_c, self._cached_corners):  # type: ignore[arg-type]
            self._homography = _compute_homography(sorted_c, self._output_size)
            self._cached_corners = sorted_c
            if is_first:
                self._log.info("Homography established")
            else:
                self._log.debug("Homography recomputed — board corners shifted")
