"""Stage 1 — Board Masker (SAM 3.1, async).

Runs SAM 3.1 in a background process to segment the whiteboard region.
Returns a raw uint8 board mask each time SAM fires (~10s cadence);
returns None between cycles so the caller can reuse the cached homography.
Corner extraction and homography computation are the rectifier's responsibility.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "sam3.1_multiplex.pt"


# ---------------------------------------------------------------------------
# Worker process
# ---------------------------------------------------------------------------


def _worker_main(
    in_q: mp.Queue,
    out_q: mp.Queue,
    model_path: str,
) -> None:
    """SAM 3.1 board-segmentation worker — runs in a dedicated child process."""
    import logging as _log

    _log.basicConfig(level=logging.WARNING)

    from ultralytics.models.sam import SAM3SemanticPredictor

    sam = SAM3SemanticPredictor(
        overrides=dict(
            model=model_path,
            task="segment",
            mode="predict",
            imgsz=644,
            save=False,
            verbose=False,
        )
    )

    while True:
        frame = in_q.get()
        if frame is None:
            break

        board_mask: np.ndarray | None = None
        try:
            board_res = sam(frame, text=["whiteboard"])
            if board_res and board_res[0].masks is not None:
                masks = board_res[0].masks.data.cpu().numpy()
                if masks.shape[0] > 0:
                    areas = masks.sum(axis=(1, 2))
                    board_mask = (masks[areas.argmax()] > 0.5).astype(np.uint8)
        except Exception:
            logging.getLogger(__name__).exception("SAM board segmentation failed")

        if board_mask is None:
            h, w = frame.shape[:2]
            board_mask = np.zeros((h, w), dtype=np.uint8)

        try:
            out_q.get_nowait()
        except Exception:
            pass
        try:
            out_q.put_nowait(board_mask)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class BoardMasker:
    """Non-blocking SAM 3.1 board segmentation.

    Spawns a child process running SAM 3.1 to segment the whiteboard region.
    segment() returns the latest board mask when SAM produces a fresh result,
    or None between cycles. The rectifier caches homography across None returns.

    Args:
        model_path: Path to the SAM 3.1 model weights.
    """

    def __init__(self, model_path: Path = _MODEL_PATH) -> None:
        self._cached: np.ndarray | None = None
        self._in_q: mp.Queue = mp.Queue(maxsize=1)
        self._out_q: mp.Queue = mp.Queue(maxsize=1)
        self._worker = mp.Process(
            target=_worker_main,
            args=(self._in_q, self._out_q, str(model_path)),
            daemon=True,
            name="sam3-board-masker",
        )
        self._worker.start()
        logger.info("BoardMasker worker started (pid=%d)", self._worker.pid)

    def segment(self, frame: np.ndarray) -> np.ndarray | None:
        """Submit *frame* for async SAM inference; return fresh board mask or None.

        Non-blocking. Returns a uint8 H×W mask (1=board, 0=background) when SAM
        produces a new result, otherwise None. The rectifier should use its cached
        homography when None is returned.
        """
        try:
            self._in_q.put_nowait(frame)
        except Exception:
            pass

        try:
            new_mask = self._out_q.get_nowait()
            self._cached = new_mask
            return new_mask
        except Exception:
            return None

    def shutdown(self) -> None:
        """Signal the worker to stop and wait for it to exit."""
        try:
            self._in_q.put_nowait(None)
        except Exception:
            pass
        self._worker.join(timeout=5)
        if self._worker.is_alive():
            self._worker.terminate()
        logger.info("BoardMasker worker stopped")
