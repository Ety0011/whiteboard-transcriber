from abc import ABC, abstractmethod

import numpy as np


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
    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Inference loop execution.
        Must return a list of dictionaries structured as:
            {
                "text": str,          # Text labels/confidence to overlay
                "poly": np.ndarray,   # Boundary coordinates, shape (N, 2), dtype=int32
                "label": str,         # Simplified taxonomy ("MATH", "TABLE", "DIAGRAM", "TEXT")
                "color": tuple        # BGR coordinate color
            }
        """
        pass
