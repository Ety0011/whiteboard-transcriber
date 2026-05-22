"""Stage 4 — Specular-Free Board Reconstruction.

Maintains a clean composite of the whiteboard surface using a distance-weighted
Exponential Moving Average (EMA).

Person/shadow removal:
  lr(x) = max_lr * (dist(x) / falloff_distance) ^ power
  Pixels under/near the body mask are frozen at their last known value,
  preserving written content under occluding figures.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class BoardReconstructor:
    """Stateful board-reconstruction stage backed by distance-weighted EMA."""

    def __init__(
        self,
        max_lr: float = 0.2,
        falloff_distance: float = 200.0,
        power: float = 2.0,
    ) -> None:
        self._max_lr = max_lr
        self._falloff_distance = falloff_distance
        self._power = power
        self._composite: np.ndarray | None = None  # float32 BGR

        logger.info(
            "BoardReconstructor initialised (max_lr=%.4f, falloff=%.1f, p=%.1f)",
            max_lr,
            falloff_distance,
            power,
        )

    def update(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Update the board model and return the clean composite.

        Args:
            frame: BGR uint8 rectified frame from Stage 3.
            mask:  Binary body mask, uint8 H×W (1=occluder, 0=board).

        Returns:
            BGR uint8 clean board composite.
        """
        frame_float = frame.astype(np.float32)

        if self._composite is None:
            self._composite = frame_float.copy()
        elif not mask.any():
            # No person: uniform EMA — skip O(H×W) distanceTransform
            self._composite += self._max_lr * (frame_float - self._composite)
        else:
            visible = (mask == 0).astype(np.uint8)
            dist_map = cv2.distanceTransform(visible, cv2.DIST_L2, 5)
            norm_dist = np.clip(dist_map / self._falloff_distance, 0.0, 1.0)
            lr = (np.power(norm_dist, self._power) * self._max_lr)[..., np.newaxis]
            self._composite = (1.0 - lr) * self._composite + lr * frame_float

        return self._composite.astype(np.uint8)
