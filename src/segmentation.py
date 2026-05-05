"""Stage 2 — Person Segmentation.

Produces a binary mask that marks pixels belonging to people (arms,
torso, hands holding markers) so they can be excluded from the
background model in Stage 3.

Library: MediaPipe Image Segmenter (Tasks API, ≥ 0.10).

The selfie-segmenter landscape TFLite model (~450 KB) is downloaded
automatically to ``models/selfie_segmenter.tflite`` on first run.

WARNING: MediaPipe requires RGB input — always convert from BGR before
calling the segmenter, or the mask will be silently incorrect.

The output mask is uint8 with values 0 (board visible) and 1 (person).
The mask is dilated by ``dilation_px`` pixels to create a safety margin
around detected person boundaries, covering arms and hands that the model
tends to under-segment.

Typical usage::

    segmenter = Segmenter()
    mask = segmenter.process(warped_frame)  # warped_frame from Stage 1
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)

# MediaPipe selfie-segmenter landscape model (2 classes: background=0, person=1)
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "image_segmenter/selfie_segmenter_landscape/float16/latest/"
    "selfie_segmenter_landscape.tflite"
)
_DEFAULT_MODEL_PATH = Path("models") / "selfie_segmenter.tflite"


def _ensure_model(path: Path = _DEFAULT_MODEL_PATH) -> Path:
    """Download the selfie-segmenter model if it is not already on disk.

    Args:
        path: Destination file path for the TFLite model.

    Returns:
        The resolved path to the model file.
    """
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading selfie segmenter model to %s …", path)
        urllib.request.urlretrieve(_MODEL_URL, path)
        logger.info("Model downloaded (%.0f KB)", path.stat().st_size / 1024)
    return path


class Segmenter:
    """Stateful person-segmentation stage backed by MediaPipe Image Segmenter.

    The MediaPipe model is initialised eagerly in ``__init__`` so the
    one-time startup cost is paid at pipeline construction, not mid-stream.
    """

    def __init__(
        self,
        model_path: Path | None = None,
        threshold: float = 0.5,
        dilation_px: int = 12,
    ) -> None:
        """
        Args:
            model_path: Path to the TFLite model file. Downloads automatically
                if ``None`` and the default path does not exist.
            threshold: Confidence cutoff on the float segmentation mask.
                Pixels with person-confidence > threshold are classified as person.
            dilation_px: Elliptical dilation radius applied to the binary mask
                to create a safety margin around detected person boundaries.
                Set to 0 to disable dilation.
        """
        self._threshold = threshold
        self._dilation_px = dilation_px

        resolved = _ensure_model(model_path or _DEFAULT_MODEL_PATH)
        base_options = mp.tasks.BaseOptions(model_asset_path=str(resolved))
        options = mp.tasks.vision.ImageSegmenterOptions(
            base_options=base_options,
            output_confidence_masks=True,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
        )
        self._segmenter = mp.tasks.vision.ImageSegmenter.create_from_options(options)

        if dilation_px > 0:
            ksize = 2 * dilation_px + 1
            self._kernel: np.ndarray | None = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (ksize, ksize)
            )
        else:
            self._kernel = None

        logger.info(
            "Segmenter ready (threshold=%.2f, dilation=%dpx)",
            threshold,
            dilation_px,
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Compute a binary person mask for *frame*.

        Args:
            frame: BGR uint8 image (perspective-corrected output from Stage 1).

        Returns:
            Binary mask as uint8 ndarray with shape ``(H, W)``.
            Value 1 means "person present"; 0 means "board visible".
        """
        # MediaPipe requires RGB — BGR input silently produces garbage masks
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        float_mask = self._run_inference(rgb)

        mask = (float_mask > self._threshold).astype(np.uint8)

        if self._kernel is not None:
            mask = cv2.dilate(mask, self._kernel, iterations=1)

        return mask

    # ------------------------------------------------------------------
    # Internal — isolated for easy mocking in tests
    # ------------------------------------------------------------------

    def _run_inference(self, rgb: np.ndarray) -> np.ndarray:
        """Run MediaPipe inference and return a float32 person-confidence mask.

        Args:
            rgb: RGB uint8 image of shape ``(H, W, 3)``.

        Returns:
            Float32 array of shape ``(H, W)`` with per-pixel person confidence
            in [0, 1].
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._segmenter.segment(mp_image)
        masks = result.confidence_masks
        if len(masks) >= 2:
            # Landscape selfie segmenter: index 0 = background, 1 = person
            return np.array(masks[1].numpy_view())
        # Fallback for single-output models: invert the background confidence
        return 1.0 - np.array(masks[0].numpy_view())


# ---------------------------------------------------------------------------
# Module-level convenience — delegates to a lazily-created global instance
# ---------------------------------------------------------------------------

_global_segmenter: Segmenter | None = None


def process(frame: np.ndarray) -> np.ndarray:
    """Compute a binary person mask using a module-global :class:`Segmenter`.

    Args:
        frame: BGR uint8 image (perspective-corrected, from Stage 1).

    Returns:
        Binary mask as uint8 ndarray with shape ``(H, W)``.
        Pixel value 1 means "person present"; 0 means "board visible".
    """
    global _global_segmenter
    if _global_segmenter is None:
        _global_segmenter = Segmenter()
    return _global_segmenter.process(frame)
