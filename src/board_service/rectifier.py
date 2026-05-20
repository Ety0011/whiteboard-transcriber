"""Stage 3 — Perspective Rectification.

Warps the raw camera frame and person mask to a canonical 1920×1080 view.
Accepts the latest board mask from BoardMasker; derives corners from it,
computes and caches the homography, then warps both inputs every frame.
Falls back to a simple resize before the first SAM result arrives.

All geometry (corner extraction, homography) lives here — upstream stages
produce masks only.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


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


def _are_corners_shifted(
    new: np.ndarray,
    cached: np.ndarray,
    threshold: float = 50.0,
) -> bool:
    """Return True if *new* corners represent a meaningful shift from *cached*."""

    def _area(p: np.ndarray) -> float:
        return 0.5 * abs(
            np.dot(p[:, 0], np.roll(p[:, 1], 1)) - np.dot(p[:, 1], np.roll(p[:, 0], 1))
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
# Rectifier
# ---------------------------------------------------------------------------


class Rectifier:
    """Stateful perspective-rectification stage.

    Each time a new board mask arrives, corners are extracted and the
    homography is recomputed and cached. Every frame the cached homography
    is applied to warp the raw frame and person mask to the canonical output
    size. Falls back to a centred resize until the first board mask arrives.

    Args:
        output_size: (width, height) of the rectified output images.
    """

    def __init__(self, output_size: tuple[int, int] = (1920, 1080)) -> None:
        self._output_size = output_size
        self._homography: np.ndarray | None = None
        self._cached_corners: np.ndarray | None = None  # (4, 2) float32, TL/TR/BR/BL

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
            board_mask: uint8 H×W board segmentation from BoardMasker, or None
                when SAM has not produced a fresh result this cycle. When
                non-None, corners are re-derived and the homography updated.
            person_mask: uint8 H×W person mask from PersonMasker (always fresh).

        Returns:
            Tuple ``(rect_frame, rect_mask)`` both at ``output_size``.
            Falls back to a simple resize when no homography is cached yet.
        """
        if board_mask is not None:
            self._update_homography(board_mask)

        w, h = self._output_size

        if self._homography is None:
            return (
                cv2.resize(frame, (w, h)),
                cv2.resize(person_mask, (w, h), interpolation=cv2.INTER_NEAREST),
            )

        rect_frame = cv2.warpPerspective(frame, self._homography, (w, h))
        rect_mask = cv2.warpPerspective(
            person_mask, self._homography, (w, h), flags=cv2.INTER_NEAREST
        )
        return rect_frame, rect_mask

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _update_homography(self, board_mask: np.ndarray) -> None:
        """Derive corners from *board_mask* and recompute homography if shifted."""
        corners = _mask_to_corners(board_mask)
        if corners is None:
            return
        sorted_c = _sort_corners(corners)
        if self._cached_corners is None or _are_corners_shifted(
            sorted_c, self._cached_corners
        ):
            self._homography = self._compute_homography(sorted_c)
            self._cached_corners = sorted_c
            logger.debug("Homography updated — corners: %s", sorted_c)

    def _compute_homography(self, corners: np.ndarray) -> np.ndarray:
        """Compute perspective transform from *corners* to the output rectangle."""
        w, h = self._output_size
        dst = np.array(
            [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
            dtype=np.float32,
        )
        return cv2.getPerspectiveTransform(corners, dst)
