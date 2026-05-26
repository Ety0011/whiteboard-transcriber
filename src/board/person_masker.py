"""Stage 3 — Person Segmentation (MediaPipe, sync, self-throttled).

Runs MediaPipe selfie segmentation in the main loop. Throttles via InlineStage._due()
to limit CPU cost — returns the cached mask between runs.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from logging_config import devnull_fds
from stage import InlineStage

_MP_MODEL_PATH = (
    Path(__file__).parent.parent.parent / "models" / "selfie_segmenter.tflite"
)


class PersonMasker(InlineStage):
    """MediaPipe selfie segmenter — self-throttled person mask.

    Returns a uint8 H×W mask (1=person, 0=board). Runs in the main loop but
    throttles via InlineStage._due() to limit CPU cost — returns the cached
    mask when the interval has not elapsed.

    Args:
        model_path: Path to the MediaPipe selfie segmenter TFLite model.
        threshold: Confidence threshold above which a pixel is classified as person.
        dilation_px: Dilation radius applied to the raw mask to cover arm edges
            and close small gaps.
        interval_s: Minimum seconds between segmentation runs. Staleness is
            covered by dilation_px — tune together. Default 0.1s ≈ 10 Hz.
    """

    def __init__(
        self,
        model_path: Path = _MP_MODEL_PATH,
        threshold: float = 0.5,
        dilation_px: int = 5,
        interval_s: float = 0.1,
    ) -> None:
        super().__init__(interval_s=interval_s)
        self._threshold = threshold
        self._cached_mask: np.ndarray | None = None

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

        self._log.info(
            "ready (threshold=%.2f, dilation=%dpx, interval=%.2fs)",
            threshold,
            dilation_px,
            interval_s,
        )

    def segment(self, frame: np.ndarray) -> np.ndarray:
        """Return a person mask for *frame*, refreshed at most every interval_s seconds.

        Args:
            frame: BGR uint8 camera frame.

        Returns:
            uint8 mask, same spatial size as *frame*: 1=person, 0=board.
        """
        if self._cached_mask is None or self._due():
            self._cached_mask = self._segment_frame(frame)
        return self._cached_mask

    def _segment_frame(self, frame: np.ndarray) -> np.ndarray:
        import mediapipe as mp_lib

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp_lib.Image(image_format=mp_lib.ImageFormat.SRGB, data=rgb)
        result = self._segmenter.segment(mp_image)
        float_mask = np.array(result.confidence_masks[0].numpy_view()).squeeze()
        mask = (float_mask > self._threshold).astype(np.uint8)
        if self._kernel is not None:
            mask = cv2.dilate(mask, self._kernel, iterations=1)
        return mask
