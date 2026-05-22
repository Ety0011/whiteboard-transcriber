"""Transcriber — subprocess manager for any BaseTranscriber backend.

Mirrors Discovery: factory is pickled and shipped to the worker process;
model loading happens inside the subprocess after unpickling. Main process
submits crops via submit() and drains results via get_results() — both
non-blocking.

Queue design:
  in_q  (maxsize=10): (entity_id, crop) — accepts multiple newly-stable
        regions per frame without dropping.
  out_q (maxsize=30): TranscriptionResult — drained each frame.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from typing import Callable

import numpy as np

from transcriber_service.base import BaseTranscriber, TranscriptionResult

log = logging.getLogger(__name__)


def _worker_main(
    factory: Callable[[], BaseTranscriber],
    in_q: mp.Queue,
    out_q: mp.Queue,
) -> None:
    """Transcription inference loop — runs in a dedicated child process."""
    import logging as _log
    import os

    from logging_config import suppress_worker_noise

    _level = logging.DEBUG if os.getenv("LOG_LEVEL") == "DEBUG" else logging.INFO
    _log.basicConfig(level=_level, format="%(levelname)s %(name)s: %(message)s")
    suppress_worker_noise()
    _log = _log.getLogger(__name__)

    transcriber = factory()
    transcriber.load()
    _log.info("%s ready", type(transcriber).__name__)

    while True:
        item = in_q.get()  # block until work arrives
        if item is None:  # shutdown sentinel
            break

        entity_id, crop = item
        text = ""
        try:
            text = transcriber.transcribe(crop)
            _log.debug("entity %d → %d chars: %r", entity_id, len(text), text[:60])
        except Exception:
            _log.exception("inference failed for entity %d", entity_id)

        try:
            out_q.put_nowait(TranscriptionResult(entity_id=entity_id, text=text))
        except Exception:
            _log.warning("output queue full — entity %d dropped", entity_id)


class Transcriber:
    """Non-blocking transcription worker running in a dedicated subprocess.

    submit() enqueues a crop. get_results() drains completed transcriptions.
    Both are non-blocking; the subprocess handles inference independently.
    """

    def __init__(self, factory: Callable[[], BaseTranscriber]) -> None:
        self._in_q: mp.Queue = mp.Queue(maxsize=10)
        self._out_q: mp.Queue = mp.Queue(maxsize=30)
        self._worker = mp.Process(
            target=_worker_main,
            args=(factory, self._in_q, self._out_q),
            daemon=True,
            name="transcription-worker",
        )
        self._worker.start()
        log.info("Transcriber started (pid=%d)", self._worker.pid)

    def submit(self, entity_id: int, crop: np.ndarray) -> None:
        """Enqueue *crop* for transcription. Non-blocking; logs if queue full."""
        try:
            self._in_q.put_nowait((entity_id, crop))
        except Exception:
            log.warning("input queue full — entity %d dropped", entity_id)

    def get_results(self) -> list[TranscriptionResult]:
        """Drain all completed transcriptions available right now. Non-blocking."""
        results = []
        while True:
            try:
                results.append(self._out_q.get_nowait())
            except Exception:
                break
        return results

    def shutdown(self) -> None:
        """Signal the worker to stop and wait for clean exit."""
        try:
            self._in_q.put_nowait(None)
        except Exception:
            pass
        self._worker.join(timeout=10)
        if self._worker.is_alive():
            self._worker.terminate()
        log.info("Transcriber stopped")
