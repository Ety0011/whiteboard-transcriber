"""Stage 6+7 — Text Line Detection + Clustering (non-blocking WorkerStage subprocess).

LayoutWorker directly owns TextLineDetector and SingleLinkageClusterer.
detect() is non-blocking: it submits the composite frame (throttled) and
immediately returns the most recently completed result.
"""

from __future__ import annotations

import numpy as np

from stage import WorkerStage

from .clusterer import Block, SingleLinkageClusterer
from .detector import TextLineDetector


class LayoutWorker(WorkerStage):
    """Non-blocking layout detector running in a dedicated subprocess.

    Call detect(frame) every pipeline tick — it submits the frame to the
    worker (at most once per submit_interval_s) and returns the latest
    cached blocks immediately.

    Args:
        submit_interval_s: Minimum seconds between frame submissions.
    """

    _process_name = "layout-detector"

    def __init__(self, submit_interval_s: float = 0.5) -> None:
        self._submit_interval_s = submit_interval_s
        self._cached: list[Block] = []
        self._detector: TextLineDetector | None = None
        self._clusterer: SingleLinkageClusterer | None = None
        super().__init__()

    def load(self) -> None:
        """Instantiate and load models inside the subprocess."""
        self._detector = TextLineDetector()
        self._detector.load()
        self._clusterer = SingleLinkageClusterer()
        self._log.info("TextLineDetector + SingleLinkageClusterer ready")

    def _on_shutdown(self) -> None:
        if self._detector is not None:
            self._detector.shutdown()

    def _process_item(self, frame: np.ndarray) -> list[Block]:
        if self._detector is None or self._clusterer is None:
            self._log.error("_process_item called before load() completed; skipping frame")
            return []
        lines = self._detector.detect(frame)
        blocks = self._clusterer.cluster(lines)
        self._log.debug("%d blocks detected", len(blocks))
        return blocks

    def detect(self, frame: np.ndarray) -> list[Block]:
        """Submit frame (throttled) and return latest cached blocks — non-blocking."""
        self._submit_if_due(frame, self._submit_interval_s)
        result = self._poll()
        if result is not None:
            self._cached = result
        return self._cached
