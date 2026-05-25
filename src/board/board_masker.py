"""Stage 2 — Board Segmentation (SAM 3.1, async WorkerStage).

Runs SAM 3.1 in a background process to segment the whiteboard region.
Returns a raw uint8 board mask each time SAM fires (~5s cadence);
returns None between cycles so the caller can reuse the cached homography.
Corner extraction and homography computation are the rectifier's responsibility.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from stage import WorkerStage

logger = logging.getLogger(__name__)

_MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "sam3.1_multiplex.pt"


class BoardMasker(WorkerStage):
    """Non-blocking SAM 3.1 board segmentation.

    Spawns a child process running SAM 3.1 to segment the whiteboard region.
    segment() returns the latest board mask when SAM produces a fresh result,
    or None between cycles. The rectifier caches homography across None returns.

    Args:
        model_path: Path to the SAM 3.1 model weights.
        recompute_interval: Minimum seconds between SAM inference runs.
    """

    _process_name = "sam3-board-masker"
    _daemon = True

    def __init__(
        self,
        model_path: Path = _MODEL_PATH,
        recompute_interval: float = 5.0,
    ) -> None:
        self._model_path = str(model_path)
        self._recompute_interval = recompute_interval
        self._sam: Any = None  # loaded in load()
        super().__init__()

    def load(self) -> None:
        """Load SAM 3.1 inside the subprocess."""
        from logging_config import devnull_fds

        with devnull_fds(1, 2):
            from ultralytics.models.sam import SAM3SemanticPredictor

            self._sam = SAM3SemanticPredictor(
                overrides=dict(
                    model=self._model_path,
                    task="segment",
                    mode="predict",
                    imgsz=644,
                    save=False,
                    verbose=False,
                )
            )
        logging.getLogger(type(self).__name__).info("SAM worker ready")

    def _process_item(self, frame: np.ndarray) -> np.ndarray:
        log = logging.getLogger(type(self).__name__)
        board_mask: np.ndarray | None = None
        try:
            board_res = self._sam(frame, text=["whiteboard"])
            if board_res and board_res[0].masks is not None:
                masks = board_res[0].masks.data.cpu().numpy()
                if masks.shape[0] > 0:
                    areas = masks.sum(axis=(1, 2))
                    board_mask = (masks[areas.argmax()] > 0.5).astype(np.uint8)
        except Exception:
            log.exception("SAM board segmentation failed")

        if board_mask is None:
            h, w = frame.shape[:2]
            board_mask = np.zeros((h, w), dtype=np.uint8)

        return board_mask

    def segment(self, frame: np.ndarray) -> np.ndarray | None:
        """Submit *frame* for async SAM inference; return fresh board mask or None.

        Non-blocking. Returns a uint8 H×W mask (1=board, 0=background) when SAM
        produces a new result, otherwise None. The rectifier should use its cached
        homography when None is returned.
        """
        result = self._poll()
        if result is not None:
            return result

        if not self._is_busy:
            self._submit_if_due(frame, self._recompute_interval)

        return None
