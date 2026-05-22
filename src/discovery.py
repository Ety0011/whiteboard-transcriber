"""Stage 5 — non-blocking layout detector running in a dedicated subprocess.

Discovery wraps any BaseLayoutDetector behind a single input/output queue
pair.  detect() is non-blocking: it submits the frame and immediately returns
the most recently completed result.  The worker process handles model loading,
inference, and shutdown independently of the main loop.
"""

import logging
import multiprocessing as mp
from typing import Callable

import numpy as np

from layout_service.base import BaseLayoutDetector
from layout_service.grouper import Block

log = logging.getLogger(__name__)


def _worker_main(
    factory: Callable[[], BaseLayoutDetector],
    in_q: mp.Queue,
    out_q: mp.Queue,
) -> None:
    """Layout detector loop — runs in a dedicated child process."""
    import os

    from logging_config import suppress_worker_noise

    _level = logging.DEBUG if os.getenv("LOG_LEVEL") == "DEBUG" else logging.INFO
    logging.basicConfig(level=_level, format="%(levelname)s %(name)s: %(message)s")
    suppress_worker_noise()
    detector = factory()
    detector.load()
    log.info("%s ready", type(detector).__name__)

    while True:
        frame = in_q.get()  # block until work arrives
        if frame is None:  # shutdown sentinel
            detector.shutdown()
            break

        blocks: list[Block] = []
        try:
            blocks = detector.detect(frame)
            log.debug("%d blocks detected", len(blocks))
        except Exception:
            log.exception("detect failed")

        try:
            out_q.get_nowait()
        except Exception:
            pass
        try:
            out_q.put_nowait(blocks)
        except Exception:
            pass


class Discovery:
    """Non-blocking layout detector running in a dedicated subprocess.

    Call detect(frame) every pipeline tick — it submits the frame to the
    worker and returns the latest cached blocks immediately.
    """

    def __init__(self, factory: Callable[[], BaseLayoutDetector]) -> None:
        self._cached: list[Block] = []
        self._is_busy = False
        self._in_q: mp.Queue = mp.Queue(maxsize=1)
        self._out_q: mp.Queue = mp.Queue(maxsize=1)
        self._worker = mp.Process(
            target=_worker_main,
            args=(factory, self._in_q, self._out_q),
            daemon=False,
            name="stage5-layout",
        )
        self._worker.start()
        log.info("worker started (pid=%d)", self._worker.pid)

    @property
    def is_busy(self) -> bool:
        return self._is_busy

    def detect(self, frame: np.ndarray) -> list[Block]:
        """Submit frame and return latest cached blocks — non-blocking."""
        try:
            self._in_q.put_nowait(frame)
            self._is_busy = True
        except Exception:
            pass  # queue full — worker still processing previous frame

        try:
            self._cached = self._out_q.get_nowait()
            self._is_busy = False
        except Exception:
            pass

        return self._cached

    def shutdown(self) -> None:
        """Send the shutdown sentinel and wait for the worker to exit cleanly."""
        try:
            self._in_q.put_nowait(None)
        except Exception:
            pass
        self._worker.join(timeout=5)
        if self._worker.is_alive():
            self._worker.terminate()
        log.info("worker stopped")
