"""Stage 3 — Surface Reconstruction.

Maintains a clean, unobstructed view of the whiteboard surface using
an Exponential Moving Average (EMA) with a spatially varying learning rate.
The learning rate is proportional to the distance from the person mask,
minimizing shadow leakage near the professor while allowing fast updates
for clear board regions.

The composite output is: background_image where mask==1, current_frame
where mask==0.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# TODO: fix for some reason arm is not masked if very near to border
class BackgroundReconstructor:
    """Stateful surface-reconstruction stage backed by distance-weighted EMA.

    Maintains a background model of the whiteboard surface. The learning rate
    scales from 0 at the person's edge to max_lr at a set falloff distance.
    This prevents stationary people and their nearby shadows from corrupting
    the background.
    """

    def __init__(
        self,
        max_lr: float = 0.1,
        falloff_distance: float = 500.0,
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
            "BackgroundReconstructor ready (Exp-Distance-EMA: max_lr=%.4f, falloff=%.1f, p=%.1f)",
            max_lr,
            falloff_distance,
            power,
        )

    def process(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Update the background model and return the clean board composite.

        Args:
            frame: BGR uint8 warped frame from Stage 1.
            mask:  Binary person mask from Stage 2 (uint8, 0=board, 1=person).

        Returns:
            BGR uint8 composite image showing the board surface without people.
        """
        frame_float = frame.astype(np.float32)

        if self._background_float is None:
            self._background_float = frame_float.copy()
            return frame.copy()

        # 1. Calculate distance from the person mask
        # distanceTransform calculates distance to the nearest 0 pixel.
        # We set board pixels (mask==0) to 1 so the person (mask==1) becomes 0.
        visible_mask = (mask == 0).astype(np.uint8)
        dist_map = cv2.distanceTransform(visible_mask, cv2.DIST_L2, 5)

        # 2. Exponential Learning Rate calculation
        # Formula: LR = max_lr * (dist / falloff)^power
        # This creates a "dead zone" of zero-learning where the professor stands.
        normalized_dist = np.clip(dist_map / self._falloff_distance, 0, 1)
        exponential_weight = np.power(normalized_dist, self._power)
        pixel_wise_lr = (exponential_weight * self._max_lr)[..., np.newaxis]

        # 3. EMA Update: BG = (1 - LR) * BG + LR * Frame
        self._background_float = (
            1 - pixel_wise_lr
        ) * self._background_float + pixel_wise_lr * frame_float

        return self._background_float.astype(np.uint8)
