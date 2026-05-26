"""BaseTranscriber — abstract interface for OCR backends.

__init__ must stay lightweight (store config only, no model loading).
Transcriber pickles the factory and ships it to a subprocess;
model weights are not picklable. load() is called by the worker AFTER
unpickling, so models are created inside the subprocess.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class TranscriptionResult:
    """OCR result returned by the transcription worker subprocess.

    Attributes:
        entity_id: Registry ID of the entity whose crop was transcribed.
        text: Recognised text (and/or LaTeX) returned by the VLM backend.
    """

    entity_id: int
    text: str


class BaseTranscriber(ABC):
    """Abstract OCR backend.

    Implement load() to initialise the model and transcribe() to run inference.
    """

    @abstractmethod
    def load(self) -> None:
        """Load model inside the worker subprocess. Never call from main process."""

    @abstractmethod
    def transcribe(self, crop: np.ndarray) -> str:
        """Run OCR on a BGR uint8 crop and return the recognised text."""

    def shutdown(self) -> None:
        """Release any resources held by the transcriber. No-op by default."""
