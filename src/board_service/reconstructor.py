"""Stage 4 — Specular-Free Board Reconstruction.

Maintains a clean composite of the whiteboard surface using distance-weighted
EMA, then inpaints any detected glare regions before returning the result.

Person/shadow removal (EMA layer):
  lr(x) = max_lr * (dist(x) / falloff_distance) ^ power
  Pixels under/near the body mask are frozen at their last known value.
  With max_lr=1.0 and the SAM-gated update cadence this is effectively a
  single-frame inpaint: unoccluded pixels take the current frame directly,
  occluded pixels retain the previous composite.

Glare suppression (spatial detection + inpainting):
  Glare = pixels that are simultaneously very bright (near saturation) AND
  spatially smooth (low Laplacian response). These are excluded from the EMA
  update and then inpainted in the output.

Two inpainting backends (switchable via neural_inpaint constructor flag):
  Neural  — LaMa (large-mask inpainting) loaded as TorchScript from
            HuggingFace Hub (smartywu/big-lama, ~200 MB, downloaded once).
            Better reconstruction; recommended.
  Classical — cv2.inpaint (Telea). Zero dependencies, instant. Use as
              fallback when the neural model is unavailable.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Glare detection thresholds
_GLARE_BRIGHTNESS: int = 248   # grayscale ≥ this → candidate glare pixel
_GLARE_EDGE_MAX: float = 15.0  # |Laplacian| < this → spatially smooth (not ink)


# ---------------------------------------------------------------------------
# Inpainting backends
# ---------------------------------------------------------------------------

class _ClassicalInpainter:
    """cv2 Telea inpainting — no model, instant, Option A."""

    def inpaint(self, bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return cv2.inpaint(bgr, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)


class _LaMaInpainter:
    """LaMa neural inpainter loaded as TorchScript from HuggingFace Hub, Option B."""

    def __init__(self) -> None:
        import torch
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download("smartywu/big-lama", "big-lama.pt")
        device = "mps" if _mps_available() else "cpu"
        self._model = torch.jit.load(model_path, map_location=device)
        self._model.eval()
        self._device = device
        logger.info("LaMa inpainter loaded on %s", device)

    def inpaint(self, bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        import torch

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img_t = (
            torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        ).to(self._device)
        msk_t = (
            torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
        ).to(self._device)

        with torch.no_grad():
            out = self._model(img_t, msk_t)

        out_np = (
            out.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0).cpu().numpy() * 255
        ).astype(np.uint8)
        return cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)


def _mps_available() -> bool:
    try:
        import torch
        return torch.backends.mps.is_available()
    except Exception:
        return False


def _load_inpainter(neural: bool):
    if neural:
        try:
            return _LaMaInpainter()
        except Exception:
            logger.warning(
                "LaMa load failed — falling back to classical inpainting", exc_info=True
            )
    return _ClassicalInpainter()


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class BoardReconstructor:
    """Stage 4: distance-weighted EMA + glare suppression with inpainting.

    Args:
        neural_inpaint: True = LaMa neural inpainter (Option B, recommended).
                        False = cv2.inpaint Telea (Option A, fallback).
    """

    def __init__(
        self,
        max_lr: float = 1.0,
        falloff_distance: float = 100.0,
        power: float = 2.0,
        neural_inpaint: bool = True,
    ) -> None:
        self._max_lr = max_lr
        self._falloff_distance = falloff_distance
        self._power = power
        self._neural = neural_inpaint
        self._composite: np.ndarray | None = None  # float32 BGR
        self._inpainter = None  # lazy-loaded on first glare hit

        logger.info(
            "BoardReconstructor ready (max_lr=%.1f, falloff=%.0fpx, inpaint=%s)",
            max_lr,
            falloff_distance,
            "neural" if neural_inpaint else "classical",
        )

    def process(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Update the board model and return the clean composite.

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
            occlusion = np.clip(mask.astype(np.uint8) | glare_mask, 0, 1)
            visible = (occlusion == 0).astype(np.uint8)
            dist_map = cv2.distanceTransform(visible, cv2.DIST_L2, 5)
            norm_dist = np.clip(dist_map / self._falloff_distance, 0.0, 1.0)
            lr = (np.power(norm_dist, self._power) * self._max_lr)[..., np.newaxis]
            self._composite = (1.0 - lr) * self._composite + lr * frame_float

        out = self._composite.astype(np.uint8)

        if glare_mask.any():
            if self._inpainter is None:
                self._inpainter = _load_inpainter(self._neural)
            out = self._inpainter.inpaint(out, glare_mask)

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
    return bright & smooth
