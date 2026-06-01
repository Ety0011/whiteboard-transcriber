"""Stage 3 — Person Segmentation (MediaPipe, synchronous InlineStage).

Runs MediaPipe selfie segmentation inline in the orchestrator thread.
segment() is throttled to ~10 Hz via InlineStage._due() and returns None
on sub-interval ticks. The orchestrator caches the last known mask.

MediaPipe selfie segmenter runs at ~5ms per frame on Apple Silicon M4,
making subprocess IPC overhead unnecessary and keeping the architecture simpler.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from stage import InlineStage

from .segmenter import Segmenter

_MP_MODEL_PATH = (
    Path(__file__).parent.parent.parent / "models" / "selfie_segmenter.tflite"
)


class PersonSegmenter(InlineStage, Segmenter):
    """Synchronous MediaPipe selfie segmentation, throttled to ~10 Hz.

    Call load() once before the pipeline starts to initialize the MediaPipe
    model in the orchestrator thread. segment() is non-blocking and returns
    None on sub-interval ticks; the orchestrator caches the last known mask.

    Args:
        model_path: Path to the MediaPipe TFLite selfie segmenter model.
        threshold: Confidence threshold above which a pixel is classified as person.
        dilation_px: Morphological dilation radius applied to the raw mask (pixels).
        recompute_interval: Minimum seconds between inference runs (~10 Hz default).

    Raises:
        FileNotFoundError: If *model_path* does not exist (checked at construction time).
    """

    def __init__(
        self,
        model_path: Path = _MP_MODEL_PATH,
        threshold: float = 0.5,
        dilation_px: int = 5,
        recompute_interval: float = 0.1,
    ) -> None:
        if not Path(model_path).is_file():
            raise FileNotFoundError(
                f"MediaPipe model not found: {model_path!r}. "
                "Download selfie_segmenter.tflite and place it in models/."
            )
        super().__init__(interval_s=recompute_interval)
        self._model_path = str(model_path)
        self._threshold = threshold
        self._dilation_px = dilation_px
        self._segmenter = None  # loaded in load()
        self._kernel: np.ndarray | None = None  # built in load()
        self._mp = None  # mediapipe module ref, cached in load()
        self._cached_mask: np.ndarray | None = None  # returned on sub-interval ticks

    def load(self) -> None:
        """Load the MediaPipe model. Call once before the pipeline starts."""
        from logging_config import devnull_fds

        with devnull_fds(2):
            import mediapipe as mp_lib

            base_options = mp_lib.tasks.BaseOptions(model_asset_path=self._model_path)
            options = mp_lib.tasks.vision.ImageSegmenterOptions(
                base_options=base_options,
                output_confidence_masks=True,
                running_mode=mp_lib.tasks.vision.RunningMode.IMAGE,
            )
            self._segmenter = mp_lib.tasks.vision.ImageSegmenter.create_from_options(
                options
            )
        self._mp = mp_lib

        if self._dilation_px > 0:
            ksize = 2 * self._dilation_px + 1
            self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

        self._log.info(
            "PersonSegmenter ready (threshold=%.2f, dilation=%dpx, interval=%.2fs)",
            self._threshold,
            self._dilation_px,
            self._interval_s,
        )

    @property
    def cached_mask(self) -> np.ndarray | None:
        """Latest computed person mask, or None before first inference."""
        return self._cached_mask

    def wait_ready(self, timeout: float | None = None) -> bool:
        """Return True immediately — PersonSegmenter loads synchronously via load()."""
        return self._segmenter is not None

    def segment(self, frame: np.ndarray) -> np.ndarray | None:
        """Run MediaPipe inference if due; return latest person mask.

        Non-blocking. Throttled by _interval_s; returns the cached result on
        sub-interval ticks so callers never need to manage their own cache.
        Returns None only before the first inference has run.

        Args:
            frame: BGR uint8 camera frame.

        Returns:
            uint8 H×W mask (1=person, 0=board), or None before first run.
        """
        if not self._due():
            return self._cached_mask

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._segmenter.segment(mp_image)
        float_mask = np.array(result.confidence_masks[0].numpy_view()).squeeze()
        mask = (float_mask > self._threshold).astype(np.uint8)
        if self._kernel is not None:
            mask = cv2.dilate(mask, self._kernel, iterations=1)
        self._cached_mask = mask
        return mask

    def shutdown(self) -> None:
        """Close the MediaPipe segmenter and release model resources."""
        if self._segmenter is not None:
            self._segmenter.close()
            self._segmenter = None


class NullPersonSegmenter(Segmenter):
    """Drop-in for PersonSegmenter that always returns None (demo mode).

    The orchestrator falls back to a zero mask when None is received, achieving
    the same effect as having no person detected.
    """

    def segment(self, frame: np.ndarray) -> np.ndarray | None:
        """Return None — no person present in demo mode."""
        return None

    def wait_ready(self, timeout: float | None = None) -> bool:
        """Immediately ready — no model to load."""
        return True
