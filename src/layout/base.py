"""Abstract base for layout detectors that run inside LayoutWorker's worker subprocess.

The load/detect split enforces the subprocess contract: __init__ must be
lightweight and picklable (no model weights), while load() runs inside the
worker after unpickling where models can be safely allocated on the target device.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .block import Block


class BaseLayoutDetector(ABC):
    """Abstract interface to decouple model architectures from pipeline execution.

    __init__ must stay lightweight (store config only, no model loading).
    LayoutWorker pickles the factory and ships it to a subprocess; model weights
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

        Args:
            frame: BGR uint8 clean board composite from Stage 5.

        Returns:
            List of Blocks, each carrying bbox, confidence, and constituent lines.
        """
        pass

    def shutdown(self) -> None:
        """Release any resources held by the detector. No-op by default."""
        pass
