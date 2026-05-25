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

import logging
from typing import Callable

import numpy as np

from stage import WorkerStage

from .base import BaseTranscriber, TranscriptionResult

log = logging.getLogger(__name__)


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
        self._transcriber: BaseTranscriber | None = None  # loaded in load()
        super().__init__()

    def load(self) -> None:
        """Instantiate and load the transcriber inside the subprocess."""
        self._transcriber = self._factory()
        self._transcriber.load()
        log.info("%s ready", type(self._transcriber).__name__)

    def _process_item(self, item: tuple[int, np.ndarray]) -> TranscriptionResult:
        assert self._transcriber is not None
        entity_id, crop = item
        text = ""
        try:
            text = self._transcriber.transcribe(crop)
            log.debug("entity %d → %d chars: %r", entity_id, len(text), text[:60])
        except Exception:
            log.exception("inference failed for entity %d", entity_id)
        return TranscriptionResult(entity_id=entity_id, text=text)

    def _put_result(self, result: TranscriptionResult) -> None:  # type: ignore[override]
        try:
            self._out_q.put_nowait(result)
        except Exception:
            log.warning("output queue full — entity %d dropped", result.entity_id)

    def _on_input_full(self, item: tuple[int, np.ndarray]) -> None:
        entity_id, _ = item
        log.warning("input queue full — entity %d dropped", entity_id)

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
