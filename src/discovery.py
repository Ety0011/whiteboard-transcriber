import logging
import multiprocessing as mp
import time
from typing import Callable

import numpy as np

from layout_service.base import BaseLayoutDetector

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

        regions: list[dict] = []
        latency = 0.0
        try:
            t0 = time.monotonic()
            regions = detector.detect(frame)
            latency = time.monotonic() - t0
            log.debug("Stage5: %d regions (%.1fms)", len(regions), latency * 1000)
        except Exception:
            log.exception("Stage5 detect failed")

        try:
            out_q.get_nowait()
        except Exception:
            pass
        try:
            out_q.put_nowait((regions, latency))
        except Exception:
            pass


class Discovery:
    """Non-blocking layout detector running in a dedicated subprocess.

    Call detect(frame) every pipeline tick — it submits the frame to the
    worker and returns the latest cached (regions, latency) pair immediately,
    matching the AnchorDetector non-blocking contract.
    """

    def __init__(
        self,
        factory: Callable[[], BaseLayoutDetector],
        target_w: int,
        target_h: int,
    ) -> None:
        self.target_w = target_w
        self.target_h = target_h
        self._cached: tuple[list[dict], float] = ([], 0.0)
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
        print(f"Discovery worker started (pid={self._worker.pid})")

    @property
    def is_busy(self) -> bool:
        return self._is_busy

    def poll(self) -> tuple[list[dict], float]:
        """Return latest cached result without submitting a new frame."""
        try:
            self._cached = self._out_q.get_nowait()
            self._is_busy = False
        except Exception:
            pass
        return self._cached

    def detect(self, frame: np.ndarray) -> tuple[list[dict], float]:
        """Submit frame and return latest (regions, latency) — non-blocking."""
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
        print("Discovery worker stopped")
