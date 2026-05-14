"""Tests for Stage 4 — Board Reconstruction (src/board_reconstructor.py).

No mocking required: BoardReconstructor uses only pure OpenCV and NumPy,
with no model downloads or GPU dependencies.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.board_reconstructor import BoardReconstructor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _warm_up(
    rec: BoardReconstructor, frame: np.ndarray, mask: np.ndarray, n: int = 5
) -> None:
    """Feed *n* identical frames through *rec* to build a stable board model."""
    for _ in range(n):
        rec.process(frame, mask)


# ---------------------------------------------------------------------------
# Output shape and dtype
# ---------------------------------------------------------------------------


def test_output_shape(blank_board: np.ndarray) -> None:
    """process() output shape must equal the input frame shape."""
    rec = BoardReconstructor()
    mask = np.zeros(blank_board.shape[:2], dtype=np.uint8)
    result = rec.process(blank_board, mask)
    assert result.shape == blank_board.shape


def test_output_dtype(blank_board: np.ndarray) -> None:
    """process() must return a uint8 array."""
    rec = BoardReconstructor()
    mask = np.zeros(blank_board.shape[:2], dtype=np.uint8)
    result = rec.process(blank_board, mask)
    assert result.dtype == np.uint8


# ---------------------------------------------------------------------------
# Cold-start guard
# ---------------------------------------------------------------------------


def test_cold_start_returns_frame_unchanged(blank_board: np.ndarray) -> None:
    """First call initialises the model and returns the frame unchanged."""
    rec = BoardReconstructor()
    mask = np.zeros(blank_board.shape[:2], dtype=np.uint8)
    result = rec.process(blank_board, mask)
    np.testing.assert_array_equal(result, blank_board)


# ---------------------------------------------------------------------------
# EMA convergence
# ---------------------------------------------------------------------------


def test_zero_mask_model_converges_to_frame(blank_board: np.ndarray) -> None:
    """All-zero mask (no person) — board model must converge to the input frame."""
    rec = BoardReconstructor()
    mask = np.zeros(blank_board.shape[:2], dtype=np.uint8)
    _warm_up(rec, blank_board, mask, n=10)
    result = rec.process(blank_board, mask)
    # After many identical frames with no person, model == frame
    np.testing.assert_array_equal(result, blank_board)


def test_full_person_mask_freezes_model(blank_board: np.ndarray) -> None:
    """All-one mask (full person) — board model must not update toward new frame."""
    rec = BoardReconstructor()
    all_person = np.ones(blank_board.shape[:2], dtype=np.uint8)

    # Initialise model with blank_board
    rec.process(blank_board, all_person)

    # Feed a visually distinct frame; model should stay frozen at blank_board values
    blue_frame = np.zeros_like(blank_board)
    blue_frame[:] = (200, 0, 0)

    result = rec.process(blue_frame, all_person)
    # Result should not match blue_frame — it is sourced from the frozen model
    assert not np.array_equal(result, blue_frame)


def test_partial_person_mask_updates_board_side(blank_board: np.ndarray) -> None:
    """Board-side pixels (mask=0) update; person-side pixels (mask=1) stay frozen."""
    h, w = blank_board.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[:, w // 2 :] = 1  # right half is "person"

    rec = BoardReconstructor()
    _warm_up(rec, blank_board, mask, n=10)

    # Inject a blue frame; left side (board) should update, right side should not
    blue_frame = np.full_like(blank_board, 0)
    blue_frame[:] = (200, 0, 0)
    result = rec.process(blue_frame, mask)

    # Person side should NOT equal the blue frame (model is frozen there)
    assert not np.array_equal(result[:, w // 2 :], blue_frame[:, w // 2 :])
