"""Stage 6 — Text Line Detection (non-blocking WorkerStage subprocess).

LayoutWorker wraps any BaseLayoutDetector behind a WorkerStage subprocess.
detect() is non-blocking: it submits the composite frame (throttled) and
immediately returns the most recently completed result.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from stage import WorkerStage

from .base import BaseLayoutDetector
from .block import Block


class LayoutWorker(WorkerStage):
    """Non-blocking layout detector running in a dedicated subprocess.

    Call detect(frame) every pipeline tick — it submits the frame to the
    worker (at most once per submit_interval_s) and returns the latest
    cached blocks immediately.

    Args:
        factory: Zero-argument callable that constructs the BaseLayoutDetector
            inside the subprocess after unpickling.
        submit_interval_s: Minimum seconds between frame submissions.
    """

    _process_name = "layout-detector"

    def __init__(
        self,
        factory: Callable[[], BaseLayoutDetector],
        submit_interval_s: float = 0.5,
    ) -> None:
        self._factory = factory
        self._submit_interval_s = submit_interval_s
        self._cached: list[Block] = []
        super().__init__()

    def _process_item(self, frame: np.ndarray) -> list[Block]:
        assert self._model is not None
        blocks = self._model.detect(frame)
        self._log.debug("%d blocks detected", len(blocks))
        return blocks

    def detect(self, frame: np.ndarray) -> list[Block]:
        """Submit frame (throttled) and return latest cached blocks — non-blocking."""
        self._submit_if_due(frame, self._submit_interval_s)
        result = self._poll()
        if result is not None:
            self._cached = result
        return self._cached
