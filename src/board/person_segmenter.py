"""Stage 3 — Person Segmentation (MediaPipe, async WorkerStage subprocess).

Runs MediaPipe selfie segmentation in a dedicated subprocess. segment() returns
the latest mask when a fresh result is available, or None between inference cycles.
The orchestrator caches the last known mask and falls back to it on None returns.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from stage import WorkerStage

_MP_MODEL_PATH = (
    Path(__file__).parent.parent.parent / "models" / "selfie_segmenter.tflite"
)


class PersonSegmenterWorker(WorkerStage):
    """Non-blocking MediaPipe selfie segmentation subprocess.

    Spawns a child process running MediaPipe to segment the person region.
    segment() returns a fresh uint8 H×W mask (1=person, 0=board) when the
    subprocess produces a new result, or None between cycles. The caller should
    cache and reuse the last known mask on None returns.

    Args:
        model_path: Path to the MediaPipe selfie segmenter TFLite model.
        threshold: Confidence threshold above which a pixel is person.
        dilation_px: Dilation radius applied to the raw mask.
        recompute_interval: Minimum seconds between inference runs (~10 Hz default).
    """

    _process_name = "mediapipe-person-segmenter"
    _daemon = True
    _in_queue_size = 1
    _out_queue_size = 1
    _drop_old = True

    def __init__(
        self,
        model_path: Path = _MP_MODEL_PATH,
        threshold: float = 0.5,
        dilation_px: int = 5,
        recompute_interval: float = 0.1,
    ) -> None:
        self._model_path = str(model_path)
        self._threshold = threshold
        self._dilation_px = dilation_px
        self._recompute_interval = recompute_interval
        self._segmenter = None  # loaded in load()
        self._kernel: np.ndarray | None = None  # built in load()
        super().__init__()

    def load(self) -> None:
        """Load MediaPipe model inside the subprocess."""
        from logging_config import devnull_fds

        with devnull_fds(2):
            import mediapipe as mp_lib

            base_options = mp_lib.tasks.BaseOptions(
                model_asset_path=self._model_path
            )
            options = mp_lib.tasks.vision.ImageSegmenterOptions(
                base_options=base_options,
                output_confidence_masks=True,
                running_mode=mp_lib.tasks.vision.RunningMode.IMAGE,
            )
            self._segmenter = mp_lib.tasks.vision.ImageSegmenter.create_from_options(
                options
            )

        if self._dilation_px > 0:
            ksize = 2 * self._dilation_px + 1
            self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

        self._log.info(
            "ready (threshold=%.2f, dilation=%dpx, interval=%.2fs)",
            self._threshold,
            self._dilation_px,
            self._recompute_interval,
        )

    def _process_item(self, frame: np.ndarray) -> np.ndarray:
        """Segment one frame — runs inside the subprocess."""
        import mediapipe as mp_lib

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp_lib.Image(image_format=mp_lib.ImageFormat.SRGB, data=rgb)
        result = self._segmenter.segment(mp_image)
        float_mask = np.array(result.confidence_masks[0].numpy_view()).squeeze()
        mask = (float_mask > self._threshold).astype(np.uint8)
        if self._kernel is not None:
            mask = cv2.dilate(mask, self._kernel, iterations=1)
        return mask

    def segment(self, frame: np.ndarray) -> np.ndarray | None:
        """Submit frame for async inference; return fresh mask or None.

        Non-blocking. Returns a uint8 H×W mask (1=person, 0=board) when the
        subprocess produces a new result, otherwise None. The caller should
        cache and reuse the last mask on None returns.

        Args:
            frame: BGR uint8 camera frame.

        Returns:
            Fresh person mask, or None if no new result this cycle.
        """
        result = self._poll()
        if result is not None:
            return result

        if not self._is_busy:
            self._submit_if_due(frame, self._recompute_interval)

        return None


class NullPersonSegmenter:
    """Drop-in for PersonSegmenterWorker that always returns an empty mask (demo mode).

    With no person detected, BoardCompositor skips distanceTransform and
    the renderer shows no mask overlay.
    """

    def segment(self, frame: np.ndarray) -> np.ndarray:
        """Return an all-zeros mask — no person present."""
        return np.zeros(frame.shape[:2], dtype=np.uint8)

    def wait_ready(self, timeout: float | None = None) -> bool:
        """Immediately ready — no subprocess to wait for."""
        return True

    def shutdown(self) -> None:
        """No-op."""
