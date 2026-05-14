"""Stage 3 — Perspective Rectification.

Applies a perspective warp to both the raw camera frame and the person mask,
mapping the whiteboard surface to a canonical fronto-parallel view at a fixed
output resolution.

The homography is derived from the four board corners produced by Stage 1
(BoardDetector) and cached between updates — it is only recomputed when a new
corners array arrives. Both outputs are spatially aligned because they use the
same homography.

Frame interpolation uses INTER_LINEAR; mask interpolation uses INTER_NEAREST
to preserve binary values (no blurred edges between 0 and 1).

Falls back to a simple resize when no corners have been detected yet, so
downstream stages always receive output at the canonical size.

Typical usage::

    rectifier = Rectifier()
    rect_frame, rect_mask = rectifier.process(frame, mask, corners)
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
