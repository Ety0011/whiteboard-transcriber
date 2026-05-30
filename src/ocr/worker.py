"""TranscriptionWorker — non-blocking WorkerStage subprocess for PaddleVLTranscriber.

Model loading happens inside the subprocess after unpickling. Main process submits
notes via submit() and drains completed TranscriptionResult objects via collect()
— both paths non-blocking.

Queue design:
  in_q  (maxsize=10): (note_id, crop) — accepts multiple newly-stable
        regions per frame without dropping.
  out_q (maxsize=30): TranscriptionResult — drained each frame.
"""

from __future__ import annotations

import numpy as np

from stage import WorkerStage
from tracker import Note, TranscriptionResult

from .paddle_vl import PaddleVLTranscriber


class TranscriptionWorker(WorkerStage):
    """Non-blocking transcription worker running in a dedicated subprocess.

    submit() enqueues notes for OCR; collect() drains completed
    TranscriptionResult objects. Callers apply results to NoteTracker separately.
    """

    _process_name = "transcription-worker"
    _in_queue_size = 10
    _out_queue_size = 30
    _drop_old = False
    _daemon = True
    _shutdown_timeout = 10.0

    def __init__(self) -> None:
        self._transcriber: PaddleVLTranscriber | None = None
        super().__init__()

    def load(self) -> None:
        """Instantiate and load PaddleVLTranscriber inside the subprocess."""
        self._transcriber = PaddleVLTranscriber()
        self._transcriber.load()
        self._log.info("PaddleVLTranscriber ready")

    def _process_item(self, item: tuple[int, np.ndarray]) -> TranscriptionResult:
        assert self._transcriber is not None
        note_id, crop = item
        text = ""
        try:
            text = self._transcriber.transcribe(crop)
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
