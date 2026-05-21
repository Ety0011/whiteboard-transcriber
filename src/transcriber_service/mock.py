"""MockTranscriber — zero-RAM drop-in for development.

No subprocess, no model. Returns a placeholder immediately.
Duck-typed to match TranscriptionWorker's submit/get_results/shutdown API.
"""

from __future__ import annotations

import logging

import numpy as np

from .base import TranscriptionResult

log = logging.getLogger(__name__)


class MockTranscriber:
    """Drop-in transcriber that returns placeholder text without loading any model."""

    def __init__(self) -> None:
        self._pending: list[TranscriptionResult] = []
        log.info("MockTranscriber active — no VLM loaded")

    def submit(self, entity_id: int, crop: np.ndarray) -> None:
        self._pending.append(TranscriptionResult(entity_id=entity_id, text="[mock OCR]"))

    def get_results(self) -> list[TranscriptionResult]:
        results, self._pending = self._pending, []
        return results

    def shutdown(self) -> None:
        pass
