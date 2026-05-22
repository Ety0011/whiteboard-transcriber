"""Stage 2 — Person Masker (MediaPipe, sync).

Runs MediaPipe selfie segmentation synchronously every frame to produce a
person/shadow mask in the raw camera frame's coordinate space. Always returns
a fresh mask for the current frame — no async gating required.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from logging_config import devnull_fds

logger = logging.getLogger(__name__)

_MP_MODEL_PATH = (
    Path(__file__).parent.parent.parent / "models" / "selfie_segmenter.tflite"
)


class PersonMasker:
    """MediaPipe selfie segmenter — synchronous per-frame person mask.

    Returns a uint8 H×W mask (1=person, 0=board) for every frame submitted.
    No background process — runs in the main process at ~5ms per frame.

    Args:
        model_path: Path to the MediaPipe selfie segmenter TFLite model.
        threshold: Confidence threshold above which a pixel is classified as person.
        dilation_px: Dilation radius applied to the raw mask to cover arm edges
            and close small gaps.
    """

    def __init__(
        self,
        model_path: Path = _MP_MODEL_PATH,
        threshold: float = 0.5,
        dilation_px: int = 5,
    ) -> None:
        self._threshold = threshold

        with devnull_fds(2):
            import mediapipe as mp_lib

            base_options = mp_lib.tasks.BaseOptions(model_asset_path=str(model_path))
            options = mp_lib.tasks.vision.ImageSegmenterOptions(
                base_options=base_options,
                output_confidence_masks=True,
                running_mode=mp_lib.tasks.vision.RunningMode.IMAGE,
            )
            self._segmenter = mp_lib.tasks.vision.ImageSegmenter.create_from_options(
                options
            )

        self._kernel: np.ndarray | None = None
        if dilation_px > 0:
            ksize = 2 * dilation_px + 1
            self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

        logger.info(
            "PersonMasker ready (MediaPipe threshold=%.2f, dilation=%dpx)",
            threshold,
            dilation_px,
        )

    def segment(self, frame: np.ndarray) -> np.ndarray:
        """Return a fresh uint8 H×W person mask for *frame*.

        Args:
            frame: BGR uint8 camera frame.

        Returns:
            uint8 mask, same spatial size as *frame*: 1=person, 0=board.
        """
        import mediapipe as mp_lib

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp_lib.Image(image_format=mp_lib.ImageFormat.SRGB, data=rgb)
        result = self._segmenter.segment(mp_image)
        float_mask = np.array(result.confidence_masks[0].numpy_view()).squeeze()
        mask = (float_mask > self._threshold).astype(np.uint8)
        if self._kernel is not None:
            mask = cv2.dilate(mask, self._kernel, iterations=1)
        return mask
