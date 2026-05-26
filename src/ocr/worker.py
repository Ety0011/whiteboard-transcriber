"""TranscriptionWorker — non-blocking WorkerStage subprocess for any BaseTranscriber.

Factory is pickled and shipped to the worker process; model loading happens
inside the subprocess after unpickling. Main process submits crops via submit()
and drains results via get_results() — both non-blocking.

Queue design:
  in_q  (maxsize=10): (entity_id, crop) — accepts multiple newly-stable
        regions per frame without dropping.
  out_q (maxsize=30): TranscriptionResult — drained each frame.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from stage import WorkerStage

from .base import BaseTranscriber, TranscriptionResult


class TranscriptionWorker(WorkerStage):
    """Non-blocking transcription worker running in a dedicated subprocess.

    submit() enqueues a crop. get_results() drains completed transcriptions.
    Both are non-blocking; the subprocess handles inference independently.

    Args:
        factory: Zero-argument callable that constructs the BaseTranscriber
            inside the subprocess after unpickling.
    """

    _process_name = "transcription-worker"
    _in_queue_size = 10
    _out_queue_size = 30
    _drop_old = False
    _daemon = True
    _shutdown_timeout = 10.0

    def __init__(self, factory: Callable[[], BaseTranscriber]) -> None:
        self._factory = factory
        super().__init__()

    def _process_item(self, item: tuple[int, np.ndarray]) -> TranscriptionResult:
        assert self._model is not None
        entity_id, crop = item
        text = ""
        try:
            text = self._model.transcribe(crop)
            self._log.debug("entity %d → %d chars: %r", entity_id, len(text), text[:60])
        except Exception:
            self._log.exception("inference failed for entity %d", entity_id)
        return TranscriptionResult(entity_id=entity_id, text=text)

    def submit(self, entity_id: int, crop: np.ndarray) -> None:
        """Enqueue *crop* for transcription. Non-blocking; logs if queue full."""
        self._submit((entity_id, crop))

    def get_results(self) -> list[TranscriptionResult]:
        """Drain all completed transcriptions available right now. Non-blocking."""
        results: list[TranscriptionResult] = []
        while True:
            result = self._poll()
            if result is None:
                break
            results.append(result)
        return results
