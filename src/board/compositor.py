"""Stage 5 — Board Compositing.

Maintains a clean composite of the whiteboard surface using a distance-weighted
Exponential Moving Average (EMA).

Person/shadow removal:
  lr(x) = max_lr * (dist(x) / falloff_distance) ^ power
  Pixels under/near the body mask are frozen at their last known value,
  preserving written content under occluding figures.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np

from stage import InlineStage


# ---------------------------------------------------------------------------
# Compositor ABC
# ---------------------------------------------------------------------------


class Compositor(ABC):
    """Abstract base for board compositing stages.

    Concrete implementations: BoardCompositor (EMA-based) and
    NullBoardCompositor (pass-through for demo mode).
    """

    @abstractmethod
    def composite(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Update the board model and return the clean composite.

        Args:
            frame: BGR uint8 rectified frame from Stage 4.
            mask:  uint8 H×W binary occluder mask (1=person, 0=board).

        Returns:
            BGR uint8 clean board composite.
        """
        ...


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------


class BoardCompositor(InlineStage, Compositor):
    """Stateful board-composition stage backed by distance-weighted EMA.

    Args:
        max_lr:            Maximum per-pixel learning rate (0.0–1.0).
        falloff_distance:  Distance in pixels at which lr reaches max_lr.
        power:             Exponent for the distance-to-lr curve (higher = sharper falloff).
    """

    def __init__(
        self,
        max_lr: float = 0.2,
        falloff_distance: float = 200.0,
        power: float = 2.0,
    ) -> None:
        super().__init__(interval_s=0.0)
        self._max_lr = max_lr
        self._falloff_distance = falloff_distance
        self._power = power
        self._composite: np.ndarray | None = None  # float32 BGR
        self._diff_buf: np.ndarray | None = None   # pre-allocated diff scratch buffer

        self._log.debug(
            "BoardCompositor initialised (max_lr=%.4f, falloff=%.1f, p=%.1f)",
            max_lr,
            falloff_distance,
            power,
        )

    def composite(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Update the board model and return the clean composite.

        Args:
            frame: BGR uint8 rectified frame from Stage 4.
            mask:  Binary body mask, uint8 H×W (1=occluder, 0=board).

        Returns:
            BGR uint8 clean board composite.
        """
        frame_float = frame.astype(np.float32)

        if self._composite is None:
            self._composite = frame_float.copy()
            self._diff_buf = np.empty_like(self._composite)
        elif not mask.any():
            # No person: uniform EMA — skip O(H×W) distanceTransform
            self._composite += self._max_lr * (frame_float - self._composite)
        else:
            visible = (mask == 0).astype(np.uint8)
            dist_map = cv2.distanceTransform(visible, cv2.DIST_L2, 5)
            norm_dist = np.clip(dist_map / self._falloff_distance, 0.0, 1.0)
            lr = (np.power(norm_dist, self._power) * self._max_lr)[..., np.newaxis]
            # composite += lr * (frame - composite) — avoids (1-lr)*composite broadcast
            np.subtract(frame_float, self._composite, out=self._diff_buf)
            self._diff_buf *= lr
            self._composite += self._diff_buf

        return self._composite.astype(np.uint8)


class NullBoardCompositor(Compositor):
    """Drop-in for BoardCompositor that passes the frame through unchanged.

    The canvas is already clean, so EMA would ghost erased strokes.
    """

    def composite(self, frame: np.ndarray, _mask: np.ndarray) -> np.ndarray:
        """Return *frame* unchanged."""
        return frame
