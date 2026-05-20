"""Stage 4 — Specular-Free Board Reconstruction.

Maintains a clean composite of the whiteboard surface using a distance-weighted
Exponential Moving Average (EMA), then instantly inpainted any detected glare
regions using lightweight classical OpenCV inpainting.

Person/shadow removal (EMA layer):
  lr(x) = max_lr * (dist(x) / falloff_distance) ^ power
  Pixels under/near the body mask are frozen at their last known value.
  This allows the reconstructed board to "remember" what was written under
  occluding human figures in real-time.

Glare suppression (spatial detection + classical inpainting):
  Glare = pixels that are simultaneously very bright (near saturation) AND
  spatially smooth (low Laplacian response). These are excluded from the EMA
  update and then inpainted instantly on the CPU using cv2.inpaint (Telea).
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Glare detection thresholds
_GLARE_BRIGHTNESS: int = 248  # grayscale ≥ this → candidate glare pixel
_GLARE_EDGE_MAX: float = 15.0  # |Laplacian| < this → spatially smooth (not ink)


class BoardReconstructor:
    """Stateful board-reconstruction stage backed by distance-weighted EMA.

    Maintains a running, specular-free model of the whiteboard surface.
    Inpaints glare regions using instant, zero-VRAM classical inpainting.
    """

    def __init__(
        self,
        max_lr: float = 0.2,
        falloff_distance: float = 200.0,
        power: float = 2.0,
    ) -> None:
        """
        Args:
            max_lr: The maximum learning rate for distant pixels.
            falloff_distance: Distance in pixels from the mask where learning
                reaches max_lr.
            power: The exponent for the falloff curve.
        """
        self._max_lr = max_lr
        self._falloff_distance = falloff_distance
        self._power = power
        self._composite: np.ndarray | None = None  # float32 BGR

        logger.info(
            "BoardReconstructor initialised (max_lr=%.4f, falloff=%.1f, p=%.1f, inpaint=classical)",
            max_lr,
            falloff_distance,
            power,
        )

    def process(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Update the board model and return the clean, specular-free composite.

        Args:
            frame: BGR uint8 rectified frame from Stage 3.
            mask:  Binary body mask, uint8 H×W (1=occluder/shadow, 0=board).

        Returns:
            BGR uint8 clean board image with glare inpainted.
        """
        frame_float = frame.astype(np.float32)
        glare_mask = _detect_glare(frame)

        if self._composite is None:
            self._composite = frame_float.copy()
        else:
            # Combine body mask and glare mask
            norm_glare = (glare_mask > 0).astype(np.uint8)
            occlusion = np.clip(mask.astype(np.uint8) + norm_glare, 0, 1)

            # Distance transform to scale learning rate near the occlusion boundaries
            visible = (occlusion == 0).astype(np.uint8)
            dist_map = cv2.distanceTransform(visible, cv2.DIST_L2, 5)

            # Scale learning rate (lr is zero at the occlusion boundary and scales up)
            norm_dist = np.clip(dist_map / self._falloff_distance, 0.0, 1.0)
            lr = (np.power(norm_dist, self._power) * self._max_lr)[..., np.newaxis]

            self._composite = (1.0 - lr) * self._composite + lr * frame_float

        out = self._composite.astype(np.uint8)

        # Inpaint glare regions instantly on CPU using classical Telea propagation
        if glare_mask.any():
            out = cv2.inpaint(out, glare_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

        return out


# ---------------------------------------------------------------------------
# Glare detection
# ---------------------------------------------------------------------------


def _detect_glare(frame: np.ndarray) -> np.ndarray:
    """Return binary mask of specular glare: bright AND smooth pixels."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bright = (gray >= _GLARE_BRIGHTNESS).astype(np.uint8)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    smooth = (np.abs(lap) < _GLARE_EDGE_MAX).astype(np.uint8)

    # Return as a 0 or 255 mask expected by cv2.inpaint
    return (bright & smooth).astype(np.uint8) * 255
