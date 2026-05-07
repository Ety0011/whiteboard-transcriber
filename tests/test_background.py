"""Tests for Stage 3 — Surface Reconstruction (src/background.py).

No mocking required: BackgroundReconstructor uses only pure OpenCV and NumPy,
with no model downloads or GPU dependencies.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

import src.background as bg_module
from src.background import BackgroundReconstructor
from src.background import process as bg_process


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _warm_up(rec: BackgroundReconstructor, frame: np.ndarray, mask: np.ndarray, n: int = 5) -> None:
    """Feed *n* frames through *rec* to build a stable background model."""
    for _ in range(n):
        rec.process(frame, mask)


# ---------------------------------------------------------------------------
# Output shape and dtype
# ---------------------------------------------------------------------------


def test_output_shape(blank_board: np.ndarray) -> None:
    """process() output shape must equal the input frame shape."""
    rec = BackgroundReconstructor()
    mask = np.zeros(blank_board.shape[:2], dtype=np.uint8)
    result = rec.process(blank_board, mask)
    assert result.shape == blank_board.shape


def test_output_dtype(blank_board: np.ndarray) -> None:
    """process() must return a uint8 array."""
    rec = BackgroundReconstructor()
    mask = np.zeros(blank_board.shape[:2], dtype=np.uint8)
    result = rec.process(blank_board, mask)
    assert result.dtype == np.uint8


# ---------------------------------------------------------------------------
# Compositing correctness
# ---------------------------------------------------------------------------


def test_zero_mask_returns_frame(blank_board: np.ndarray) -> None:
    """All-zero mask (no person) — composite must equal the input frame exactly."""
    rec = BackgroundReconstructor(history=2)
    mask = np.zeros(blank_board.shape[:2], dtype=np.uint8)
    _warm_up(rec, blank_board, mask)
    result = rec.process(blank_board, mask)
    np.testing.assert_array_equal(result, blank_board)


def test_partial_mask_compositing(blank_board: np.ndarray) -> None:
    """Board pixels (mask=0) come from frame; person pixels (mask=1) come from bg."""
    h, w = blank_board.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[:, w // 2 :] = 1  # right half is "person"

    rec = BackgroundReconstructor(history=2)
    _warm_up(rec, blank_board, mask)

    test_frame = np.zeros_like(blank_board)
    test_frame[:, : w // 2] = (255, 255, 255)  # left: white (board side)
    test_frame[:, w // 2 :] = (0, 0, 200)      # right: blue (person side)

    result = rec.process(test_frame, mask)

    # Board-side pixels pass through unchanged
    np.testing.assert_array_equal(result[:, : w // 2], test_frame[:, : w // 2])
    # Person-side pixels come from the background model, not the current frame
    assert not np.array_equal(result[:, w // 2 :], test_frame[:, w // 2 :])


def test_all_person_mask_differs_from_frame(blank_board: np.ndarray) -> None:
    """All-one mask — composite must differ from the current frame (sourced from bg)."""
    rec = BackgroundReconstructor(history=2)
    mask = np.ones(blank_board.shape[:2], dtype=np.uint8)
    _warm_up(rec, blank_board, mask)

    # Feed a visually distinct frame after warmup
    blue_frame = np.zeros_like(blank_board)
    blue_frame[:] = (200, 0, 0)

    result = rec.process(blue_frame, mask)
    # If compositing worked, the result is sourced from the bg model, not blue_frame
    assert not np.array_equal(result, blue_frame)


# ---------------------------------------------------------------------------
# Cold-start guard
# ---------------------------------------------------------------------------


def test_cold_start_returns_frame_unchanged(blank_board: np.ndarray) -> None:
    """When getBackgroundImage() returns None, process() returns the frame as-is."""
    from unittest.mock import MagicMock

    rec = BackgroundReconstructor()
    mask = np.zeros(blank_board.shape[:2], dtype=np.uint8)

    # cv2 C-extension methods are read-only; replace the whole subtractor object
    mock_sub = MagicMock()
    mock_sub.getBackgroundImage.return_value = None
    rec._subtractor = mock_sub

    result = rec.process(blank_board, mask)

    assert result.shape == blank_board.shape
    assert result.dtype == np.uint8
    np.testing.assert_array_equal(result, blank_board)


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------


def test_module_level_process_creates_global_and_works(blank_board: np.ndarray) -> None:
    """bg_process() must create a global BackgroundReconstructor and return valid output."""
    bg_module._global_reconstructor = None  # reset for test isolation

    mask = np.zeros(blank_board.shape[:2], dtype=np.uint8)
    result = bg_process(blank_board, mask)

    assert bg_module._global_reconstructor is not None
    assert result.shape == blank_board.shape
    assert result.dtype == np.uint8
