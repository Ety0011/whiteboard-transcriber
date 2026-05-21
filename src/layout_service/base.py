from abc import ABC, abstractmethod

import numpy as np


class BaseLayoutDetector(ABC):
    """Abstract interface to decouple model architectures from pipeline execution."""

    @abstractmethod
    def load(self) -> None:
        """Initialize models, configure devices (MPS/CPU), and load weights."""
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
