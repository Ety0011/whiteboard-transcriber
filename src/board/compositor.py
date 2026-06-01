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

    All internal scratch buffers are allocated once on the first call and
    reused across frames. The no-person path uses cv2.accumulateWeighted
    (uint8 source accepted, no float32 conversion needed). The person path
    computes per-pixel learning rates fully in-place.

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
        self._use_square = power == 2.0  # np.square is faster than np.power(..., 2.0)
        self._inv_falloff = 1.0 / falloff_distance

        # All scratch buffers are None until first composite() call.
        self._composite: np.ndarray | None = None   # float32 BGR accumulator
        self._diff_buf: np.ndarray | None = None    # float32 BGR, scratch for person path
        self._frame_float: np.ndarray | None = None # float32 BGR, holds converted input
        self._visible_buf: np.ndarray | None = None # uint8 H×W, inverted person mask
        self._dist_buf: np.ndarray | None = None    # float32 H×W, distance / lr map

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
            BGR uint8 clean board composite. Each call returns a freshly
            allocated array; the caller may hold the reference across frames.
        """
        if self._composite is None:
            # First call: seed the accumulator and pre-allocate all scratch buffers.
            h, w = frame.shape[:2]
            self._composite = frame.astype(np.float32)
            self._diff_buf = np.empty_like(self._composite)
            self._frame_float = np.empty((h, w, 3), dtype=np.float32)
            self._visible_buf = np.empty((h, w), dtype=np.uint8)
            self._dist_buf = np.empty((h, w), dtype=np.float32)
        elif not mask.any():
            # No person: uniform EMA — cv2.accumulateWeighted accepts uint8 src
            # directly, avoiding the float32 conversion entirely.
            cv2.accumulateWeighted(frame, self._composite, self._max_lr)
        else:
            # Person present: distance-weighted per-pixel LR.
            # All operations are in-place to avoid intermediate allocations.
            np.copyto(self._frame_float, frame, casting="unsafe")

            # visible_buf: 1 where no person, 0 where person (inverts binary mask).
            np.bitwise_xor(mask, 1, out=self._visible_buf)

            # dist_buf: distance of each board pixel from the nearest person pixel.
            cv2.distanceTransform(self._visible_buf, cv2.DIST_L2, 5, dst=self._dist_buf)

            # Normalize → clamp → raise to power → scale to [0, max_lr], all in-place.
            self._dist_buf *= self._inv_falloff
            np.clip(self._dist_buf, 0.0, 1.0, out=self._dist_buf)
            if self._use_square:
                np.square(self._dist_buf, out=self._dist_buf)
            else:
                np.power(self._dist_buf, self._power, out=self._dist_buf)
            self._dist_buf *= self._max_lr

            lr = self._dist_buf[..., np.newaxis]  # (H, W, 1) view — no alloc
            np.subtract(self._frame_float, self._composite, out=self._diff_buf)
            self._diff_buf *= lr
            self._composite += self._diff_buf

        return self._composite.astype(np.uint8)
