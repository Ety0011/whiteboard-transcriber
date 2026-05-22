"""MockTranscriber — zero-model drop-in for development."""

from __future__ import annotations

import numpy as np

from .base import BaseTranscriber


class MockTranscriber(BaseTranscriber):
    """Returns placeholder text without loading any model."""

    def load(self) -> None:
        pass

    def transcribe(self, crop: np.ndarray) -> str:
        return "[mock OCR]"
