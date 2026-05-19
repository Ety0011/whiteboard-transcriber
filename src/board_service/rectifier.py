"""Stage 3 — Anchor-Refined Perspective Rectification.

Warps the raw camera frame and body mask to a canonical 1920×1080 view.
Homography is cached and recomputed only when corners change. Falls back
to a simple resize before the first SAM result arrives.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Rectifier:
    """Stateful perspective-rectification stage.

    Caches the homography and only recomputes it when new corners arrive.
    Falls back to a centred resize when no corners have been detected yet.
    """

    def __init__(self, output_size: tuple[int, int] = (1920, 1080)) -> None:
        """
        Args:
            output_size: (width, height) of the rectified output images.
        """
        self._output_size = output_size
        self._homography: np.ndarray | None = None
        self._cached_corners: np.ndarray | None = None  # (4, 2) float32, TL/TR/BR/BL

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def process(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        corners: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Warp *frame* and *mask* to remove perspective distortion.

        Args:
            frame: BGR uint8 raw camera frame.
            mask: Binary uint8 person mask from Stage 2, shape ``(H, W)``.
            corners: Float32 ``(4, 2)`` board corners from BoardDetector,
                ordered TL/TR/BR/BL, or ``None`` if not yet detected.

        Returns:
            Tuple ``(rectified_frame, rectified_mask)`` at ``output_size``.
            Falls back to a simple resize when no corners are available.
        """
        if corners is not None and not np.array_equal(corners, self._cached_corners):
            self._homography = self._compute_homography(corners)
            self._cached_corners = corners
            logger.debug("Homography updated — corners: %s", corners)

        w, h = self._output_size

        if self._homography is None:
            logger.debug("No board detected yet — returning resized frame and mask")
            return (
                cv2.resize(frame, (w, h)),
                cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST),
            )

        rect_frame = cv2.warpPerspective(frame, self._homography, (w, h))
        rect_mask = cv2.warpPerspective(
            mask, self._homography, (w, h), flags=cv2.INTER_NEAREST
        )
        return rect_frame, rect_mask

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def _compute_homography(self, corners: np.ndarray) -> np.ndarray:
        """Compute perspective transform from *corners* to the output rectangle."""
        w, h = self._output_size
        dst = np.array(
            [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
            dtype=np.float32,
        )
        return cv2.getPerspectiveTransform(corners, dst)
