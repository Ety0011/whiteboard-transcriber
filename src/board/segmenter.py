"""Segmenter — abstract base class for board and person segmenters."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Segmenter(ABC):
    """Abstract base for synchronous and asynchronous segmentation workers.

    All implementations must be safe to call from the orchestrator thread.
    segment() must return immediately and never block.
    """

    @abstractmethod
    def segment(self, frame: np.ndarray) -> np.ndarray | None:
        """Submit *frame* and return the latest segmentation mask or None.

        Args:
            frame: BGR uint8 camera frame.

        Returns:
            uint8 H×W binary mask (1=region, 0=background), or None when no
            fresh result is available this tick. Callers should cache and
            reuse the last known mask on None returns.
        """
        ...

    @abstractmethod
    def wait_ready(self, timeout: float | None = None) -> bool:
        """Block until the segmenter is ready to process frames.

        Args:
            timeout: Maximum seconds to wait. None means wait indefinitely.

        Returns:
            True if ready within timeout, False on timeout expiry.
        """
        ...

    @abstractmethod
    def shutdown(self) -> None:
        """Release model resources."""
        ...
