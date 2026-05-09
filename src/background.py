"""Stage 3 — Surface Reconstruction.

Maintains a clean, unobstructed view of the whiteboard surface using
OpenCV's MOG2 Gaussian Mixture background subtractor. Person-region
pixels are masked out before feeding frames to MOG2 so that people
standing still do not corrupt the background model.

The composite output is: background_image where mask==1, current_frame
where mask==0.

Key OpenCV API: cv2.createBackgroundSubtractorMOG2,
bg_subtractor.apply(), bg_subtractor.getBackgroundImage().

Typical usage::

    reconstructor = BackgroundReconstructor()
    composite = reconstructor.process(warped_frame, person_mask)
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class BackgroundReconstructor:
    """Stateful surface-reconstruction stage backed by OpenCV MOG2.

    Maintains a background model of the whiteboard surface. Person pixels
    (mask == 1) are replaced with white before each MOG2 update so that
    stationary people do not corrupt the learned background.

    The composite output takes background pixels where people were detected
    and keeps the current frame pixels everywhere the board is visible.
    """

    def __init__(
        self,
        history: int = 500,
        var_threshold: float = 16.0,
        learning_rate: float = 0.005,
    ) -> None:
        """
        Args:
            history: Number of frames used to build the background model.
            var_threshold: Mahalanobis distance threshold for background/foreground
                classification. Lower values make the subtractor more sensitive.
            learning_rate: How fast the background model adapts (0–1). 0.005
                converges in ~200 frames (~7 s at 30 fps) while remaining stable.
        """
        self._learning_rate = learning_rate
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=False,
        )
        logger.info(
            "BackgroundReconstructor ready (history=%d, var_threshold=%.1f, lr=%.4f)",
            history,
            var_threshold,
            learning_rate,
        )

    def process(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Update the background model and return the clean board composite.

        Args:
            frame: BGR uint8 warped frame from Stage 1.
            mask:  Binary person mask from Stage 2 (uint8, 0=board, 1=person).

        Returns:
            BGR uint8 composite image showing the board surface without people.
        """
        masked_frame = frame.copy()
        masked_frame[mask == 1] = (255, 255, 255)

        self._subtractor.apply(masked_frame, learningRate=self._learning_rate)

        bg = self._subtractor.getBackgroundImage()
        if bg is None:
            return frame.copy()

        composite = frame.copy()
        composite[mask == 1] = bg[mask == 1]
        return composite


class _ProgressiveBackgroundReconstructor:
    """Alternative to MOG2: simple progressive last-seen buffer.

    Every frame, board-visible pixels (mask==0) overwrite the buffer;
    person-occluded pixels (mask==1) are left unchanged, preserving the
    last known board content beneath them. Simpler and adapts instantly
    to new writing, but takes raw noisy pixels with no temporal averaging.
    """

    def __init__(self) -> None:
        self._background: np.ndarray | None = None

    def process(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if self._background is None:
            self._background = frame.copy()
        self._background[mask == 0] = frame[mask == 0]
        composite = frame.copy()
        composite[mask == 1] = self._background[mask == 1]
        return composite


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_global_reconstructor: BackgroundReconstructor | None = None


def process(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Update the background model and return the clean board composite.

    Args:
        frame: BGR uint8 warped frame from Stage 1.
        mask:  Binary person mask from Stage 2 (uint8, 0=board, 1=person).

    Returns:
        BGR uint8 composite image showing the board surface without people.
    """
    global _global_reconstructor
    if _global_reconstructor is None:
        _global_reconstructor = BackgroundReconstructor()
    return _global_reconstructor.process(frame, mask)
