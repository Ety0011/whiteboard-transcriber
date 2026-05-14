"""Tests for Stage 1 — Board Detection (src/board_detector.py).

SAM 3 model loading is mocked in all tests that instantiate BoardDetector so
no model download or GPU is required.

Tests are split into three groups:

1. Geometric tests — call ``BoardDetector._mask_to_corners`` and
   ``_sort_corners`` directly. No mocking needed; pure NumPy/OpenCV functions.

2. Corner-cache tests — test ``_are_corners_shifted`` filtering logic.

3. Interface tests — test ``submit_frame`` / ``get_corners`` with a mocked
   ``_detect_corners``.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from src.board_detector import BoardDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_detector(**kwargs) -> BoardDetector:
    """Return a BoardDetector with SAM 3 model load patched out."""
    with patch("src.board_detector.SAM3SemanticPredictor"):
        return BoardDetector(**kwargs)


def _make_quad_mask(h: int, w: int, corners: np.ndarray) -> np.ndarray:
    """Return a binary uint8 mask with a filled quadrilateral at *corners*."""
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [corners.astype(np.int32)], 1)
    return mask


# ---------------------------------------------------------------------------
# Geometric tests — _mask_to_corners
# ---------------------------------------------------------------------------


def test_mask_to_corners_synthetic() -> None:
    """Clean quad mask → exactly 4 corners returned."""
    corners = np.array([[150, 80], [1100, 50], [1130, 640], [120, 660]], np.int32)
    mask = _make_quad_mask(720, 1280, corners)
    result = BoardDetector._mask_to_corners(mask)
    assert result is not None
    assert result.shape == (4, 2)
    assert result.dtype == np.float32


def test_mask_to_corners_empty_mask() -> None:
    """All-zero mask → None (no contour to extract)."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    assert BoardDetector._mask_to_corners(mask) is None


def test_mask_to_corners_single_pixel() -> None:
    """A single-pixel mask has no meaningful polygon → None."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[50, 50] = 1
    assert BoardDetector._mask_to_corners(mask) is None


# ---------------------------------------------------------------------------
# Geometric tests — _sort_corners
# ---------------------------------------------------------------------------


def test_sort_corners_order() -> None:
    """_sort_corners returns TL, TR, BR, BL regardless of input order."""
    pts = np.array(
        [[1100, 50], [120, 660], [150, 80], [1130, 640]], dtype=np.float32
    )
    rect = BoardDetector._sort_corners(pts)
    tl, tr, br, bl = rect
    assert tl[0] < tr[0]   # TL is left of TR
    assert bl[0] < br[0]   # BL is left of BR
    assert tl[1] < bl[1]   # TL is above BL
    assert tr[1] < br[1]   # TR is above BR


# ---------------------------------------------------------------------------
# Corner-cache filter — _are_corners_shifted
# ---------------------------------------------------------------------------


def test_are_corners_shifted_none_cached() -> None:
    """Always returns True when no corners are cached yet."""
    det = _make_detector()
    corners = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
    assert det._are_corners_shifted(corners)


def test_are_corners_shifted_identical() -> None:
    """Identical corners should return False (nothing changed)."""
    det = _make_detector(cache_threshold=10.0)
    corners = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
    det._cached_corners = corners.copy()
    assert not det._are_corners_shifted(corners.copy())


def test_are_corners_shifted_large_move() -> None:
    """All corners displaced by more than threshold → returns True."""
    det = _make_detector(cache_threshold=10.0)
    base = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
    det._cached_corners = base.copy()
    shifted = base + 50.0   # well above threshold=10
    assert det._are_corners_shifted(shifted)


def test_are_corners_shifted_area_guard() -> None:
    """Shrinking area (< 98% of cached) → returns False even if corners moved."""
    det = _make_detector(cache_threshold=5.0)
    # Cached: large quad
    det._cached_corners = np.array(
        [[0, 0], [200, 0], [200, 200], [0, 200]], dtype=np.float32
    )
    # New: tiny quad (10% of original area — simulates occlusion)
    small = np.array([[50, 50], [60, 50], [60, 60], [50, 60]], dtype=np.float32)
    assert not det._are_corners_shifted(small)


# ---------------------------------------------------------------------------
# Interface tests — submit_frame / get_corners
# ---------------------------------------------------------------------------


def test_get_corners_returns_none_initially() -> None:
    """Before any detection, get_corners() returns None."""
    det = _make_detector()
    assert det.get_corners() is None


def test_corners_set_after_detection(
    synthetic_board_frame: tuple[np.ndarray, np.ndarray],
) -> None:
    """After submit_frame triggers a detection, get_corners() returns the result."""
    frame, corners = synthetic_board_frame
    det = _make_detector(recompute_interval=0.0)
    with patch.object(det, "_detect_corners", return_value=corners.copy()):
        det.submit_frame(frame)
        time.sleep(0.1)
    assert det.get_corners() is not None
    assert det.get_corners().shape == (4, 2)


def test_submit_frame_respects_interval(
    synthetic_board_frame: tuple[np.ndarray, np.ndarray],
) -> None:
    """_detect_corners is not called more often than recompute_interval allows."""
    frame, corners = synthetic_board_frame
    det = _make_detector(recompute_interval=9999.0)

    call_count = 0

    def counting_detect(f: np.ndarray):
        nonlocal call_count
        call_count += 1
        return None

    with patch.object(det, "_detect_corners", side_effect=counting_detect):
        for _ in range(10):
            det.submit_frame(frame)
        time.sleep(0.05)

    # Only the very first call fires (interval=9999 s blocks all subsequent)
    assert call_count == 1


def test_detect_corners_uses_text_prompt(blank_board: np.ndarray) -> None:
    """_detect_corners must call SAM 3 with text=['whiteboard']."""
    det = _make_detector()
    from unittest.mock import MagicMock

    mock_result = MagicMock()
    mock_result.masks = None
    det._sam.return_value = [mock_result]
    det._detect_corners(blank_board)
    det._sam.assert_called_once_with(blank_board, text=["whiteboard"])
