"""Stage 4 — Board Reconstruction.

Maintains a persistent, clean model of the whiteboard surface by accumulating
evidence over time using an Exponential Moving Average (EMA) with a spatially
varying learning rate.

The learning rate scales with the distance from the person mask: pixels far
from any person update quickly toward the current frame, while pixels near
or under the person are frozen at their last known value. This way the
reconstructed board "remembers" what was written in regions that are currently
occluded, and converges cleanly once the person moves away.

The learning rate formula per pixel is::

    lr(x) = max_lr * (dist(x) / falloff_distance) ^ power

where ``dist(x)`` is the Euclidean distance to the nearest person pixel.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# TODO: fix for some reason arm is not masked if very near to border
class BoardReconstructor:
    """Stateful board-reconstruction stage backed by distance-weighted EMA.

    Maintains a running model of the whiteboard surface. The learning rate
    scales from 0 at the person's edge to max_lr at a set falloff distance.
    This prevents stationary people and their nearby shadows from corrupting
    the board model.
    """

    def __init__(
        self,
        max_lr: float = 1.0,
        falloff_distance: float = 100.0,
        power: float = 2.0,
    ) -> None:
        """
        Args:
            max_lr: The maximum learning rate for distant pixels.
            falloff_distance: Distance in pixels from the mask where learning
                reaches max_lr.
            power: The exponent for the falloff curve (e.g., 2.0 for squared).
        """
        self._max_lr = max_lr
        self._falloff_distance = falloff_distance
        self._power = power
        self._background_float: np.ndarray | None = None

        logger.info(
            "BoardReconstructor ready (Exp-Distance-EMA: max_lr=%.4f, falloff=%.1f, p=%.1f)",
            max_lr,
            falloff_distance,
            power,
        )

    def process(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Update the board model and return the reconstructed board surface.

        Args:
            frame: BGR uint8 rectified frame from Stage 3.
            mask:  Binary person mask from Stage 3 (uint8, 0=board, 1=person),
                   already warped to rectified coordinates.

        Returns:
            BGR uint8 image representing the clean board surface. Pixels where
            people stood recently retain the last known board content rather
            than the current (occluded) frame values.
        """
        frame_float = frame.astype(np.float32)

        if self._background_float is None:
            self._background_float = frame_float.copy()
            return frame.copy()

        # distanceTransform computes distance to the nearest zero pixel.
        # Board pixels (mask==0) are set to 1, so person pixels (mask==1) become 0.
        visible_mask = (mask == 0).astype(np.uint8)
        dist_map = cv2.distanceTransform(visible_mask, cv2.DIST_L2, 5)

        # lr = max_lr * (dist / falloff)^power — zero at person boundary, max_lr far away
        normalized_dist = np.clip(dist_map / self._falloff_distance, 0, 1)
        pixel_wise_lr = (np.power(normalized_dist, self._power) * self._max_lr)[
            ..., np.newaxis
        ]

        self._background_float = (
            1 - pixel_wise_lr
        ) * self._background_float + pixel_wise_lr * frame_float

        return self._background_float.astype(np.uint8)
