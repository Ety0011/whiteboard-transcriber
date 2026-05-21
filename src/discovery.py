import logging
import multiprocessing as mp
import time
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
    logging.basicConfig(level=logging.WARNING)
    detector = factory()
    detector.load()
    log.info("Stage5: %s ready", type(detector).__name__)

    while True:
        frame = in_q.get()  # block until work arrives
        if frame is None:  # shutdown sentinel
            break

        blocks: list[Block] = []
        latency = 0.0
        try:
            t0 = time.monotonic()
            blocks = detector.detect(frame)
            latency = time.monotonic() - t0
            log.debug("Stage5: %d blocks (%.1fms)", len(blocks), latency * 1000)
        except Exception:
            log.exception("Stage5 detect failed")

        try:
            out_q.get_nowait()
        except Exception:
            pass
        try:
            out_q.put_nowait((blocks, latency))
        except Exception:
            pass


class Discovery:
    """Non-blocking layout detector running in a dedicated subprocess.

    Call detect(frame) every pipeline tick — it submits the frame to the
    worker and returns the latest cached (blocks, latency) pair immediately.
    """

    def __init__(self, factory: Callable[[], BaseLayoutDetector]) -> None:
        self._cached: tuple[list[Block], float] = ([], 0.0)
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
        log.info("Discovery worker started (pid=%d)", self._worker.pid)

    @property
    def is_busy(self) -> bool:
        return self._is_busy

    def poll(self) -> tuple[list[Block], float]:
        """Return latest cached result without submitting a new frame."""
        try:
            self._cached = self._out_q.get_nowait()
            self._is_busy = False
        except Exception:
            pass
        return self._cached

    def detect(self, frame: np.ndarray) -> tuple[list[Block], float]:
        """Submit frame and return latest (blocks, latency) — non-blocking."""
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
        try:
            self._in_q.put_nowait(None)
        except Exception:
            pass
        self._worker.join(timeout=5)
        if self._worker.is_alive():
            self._worker.terminate()
        log.info("Discovery worker stopped")
