"""TranscriptionWorker — non-blocking WorkerStage subprocess for any BaseTranscriber.

Factory is pickled and shipped to the worker process; model loading happens
inside the subprocess after unpickling. Main process submits notes via transcribe()
and drains completed TranscriptionResult objects each call — both paths non-blocking.

Queue design:
  in_q  (maxsize=10): (note_id, crop) — accepts multiple newly-stable
        regions per frame without dropping.
  out_q (maxsize=30): TranscriptionResult — drained each frame.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from stage import WorkerStage
from tracker import Note

from .base import BaseTranscriber, TranscriptionResult


class TranscriptionWorker(WorkerStage):
    """Non-blocking transcription worker running in a dedicated subprocess.

    transcribe() submits notes for OCR and drains completed TranscriptionResult
    objects. Callers apply results to the NoteTracker separately.

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
        note_id, crop = item
        text = ""
        try:
            text = self._model.transcribe(crop)
            self._log.debug("note %d → %d chars: %r", note_id, len(text), text[:60])
        except Exception:
            self._log.exception("inference failed for note %d", note_id)
        return TranscriptionResult(note_id=note_id, text=text)

    def submit(self, notes: list[Note]) -> None:
        """Enqueue *notes* for OCR — non-blocking.

        Args:
            notes: Newly-INFERRING notes whose crop fields are populated.
        """
        for note in notes:
            self._submit((note.id, note.crop))

    def collect(self) -> list[TranscriptionResult]:
        """Drain all completed OCR results since the last call — non-blocking.

        Returns:
            All TranscriptionResult objects available in the output queue.
        """
        results = []
        while True:
            result = self._poll()
            if result is None:
                break
            results.append(result)
        return results
