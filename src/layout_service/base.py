from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .grouper import Block


class BaseLayoutDetector(ABC):
    """Abstract interface to decouple model architectures from pipeline execution.

    __init__ must stay lightweight (store config only, no model loading).
    Discovery pickles the factory and ships it to a subprocess; model weights
    are not picklable. load() is called by the worker AFTER unpickling, so
    models are created inside the subprocess where they will actually run.
    """

    @abstractmethod
    def load(self) -> None:
        """Load models inside the worker subprocess. Never call from main process."""
        pass

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Block]:
        """Run inference and return detected layout blocks.

        Each Block carries: bbox, confidence, and lines (list[TextLine]).
        """
        pass

    def shutdown(self) -> None:
        """Release any resources held by the detector. No-op by default."""
        pass
