"""Tests for Stage 3 — Perspective Rectification (src/rectifier.py).

No model loading or mocking required: Rectifier uses only OpenCV and NumPy.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.board_detector import BoardDetector
from src.rectifier import Rectifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _blank_mask(frame: np.ndarray) -> np.ndarray:
    """Return an all-zero (no person) mask matching *frame*'s spatial dims."""
    return np.zeros(frame.shape[:2], dtype=np.uint8)


def _person_mask(frame: np.ndarray) -> np.ndarray:
    """Return an all-one (full person) mask matching *frame*'s spatial dims."""
    return np.ones(frame.shape[:2], dtype=np.uint8)


# ---------------------------------------------------------------------------
# Fallback behaviour (no corners)
# ---------------------------------------------------------------------------


def test_fallback_frame_shape(blank_board: np.ndarray) -> None:
    """With no corners, process() resizes frame to output_size."""
    r = Rectifier(output_size=(1920, 1080))
    rect_frame, _ = r.process(blank_board, _blank_mask(blank_board), corners=None)
    assert rect_frame.shape == (1080, 1920, 3)


def test_fallback_mask_shape(blank_board: np.ndarray) -> None:
    """With no corners, process() resizes mask to output_size."""
    r = Rectifier(output_size=(1920, 1080))
    _, rect_mask = r.process(blank_board, _blank_mask(blank_board), corners=None)
    assert rect_mask.shape == (1080, 1920)


def test_fallback_frame_dtype(blank_board: np.ndarray) -> None:
    """Output frame dtype is always uint8."""
    r = Rectifier()
    rect_frame, _ = r.process(blank_board, _blank_mask(blank_board), corners=None)
    assert rect_frame.dtype == np.uint8


def test_fallback_mask_dtype(blank_board: np.ndarray) -> None:
    """Output mask dtype is always uint8."""
    r = Rectifier()
    _, rect_mask = r.process(blank_board, _blank_mask(blank_board), corners=None)
    assert rect_mask.dtype == np.uint8


# ---------------------------------------------------------------------------
# Warp with known corners
# ---------------------------------------------------------------------------


def test_warp_frame_shape(
    synthetic_board_frame: tuple[np.ndarray, np.ndarray],
) -> None:
    """With valid corners, warped frame has the canonical output shape."""
    frame, corners = synthetic_board_frame
    sorted_c = BoardDetector._sort_corners(corners)
    r = Rectifier(output_size=(1280, 720))
    rect_frame, _ = r.process(frame, _blank_mask(frame), corners=sorted_c)
    assert rect_frame.shape == (720, 1280, 3)


def test_warp_mask_shape(
    synthetic_board_frame: tuple[np.ndarray, np.ndarray],
) -> None:
    """With valid corners, warped mask has the canonical spatial shape."""
    frame, corners = synthetic_board_frame
    sorted_c = BoardDetector._sort_corners(corners)
    r = Rectifier(output_size=(1280, 720))
    _, rect_mask = r.process(frame, _blank_mask(frame), corners=sorted_c)
    assert rect_mask.shape == (720, 1280)


def test_warp_output_is_mostly_light(
    synthetic_board_frame: tuple[np.ndarray, np.ndarray],
) -> None:
    """Warping a frame whose board is off-white should produce a bright output."""
    frame, corners = synthetic_board_frame
    sorted_c = BoardDetector._sort_corners(corners)
    r = Rectifier(output_size=(1280, 720))
    rect_frame, _ = r.process(frame, _blank_mask(frame), corners=sorted_c)
    assert rect_frame.mean() > 128


def test_warp_mask_stays_binary(
    synthetic_board_frame: tuple[np.ndarray, np.ndarray],
) -> None:
    """Mask values must remain exactly 0 or 1 after INTER_NEAREST warp."""
    frame, corners = synthetic_board_frame
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[: h // 2, :] = 1  # top half is person

    sorted_c = BoardDetector._sort_corners(corners)
    r = Rectifier(output_size=(1280, 720))
    _, rect_mask = r.process(frame, mask, corners=sorted_c)
    unique = set(np.unique(rect_mask).tolist())
    assert unique <= {0, 1}, f"Mask contains non-binary values: {unique}"


# ---------------------------------------------------------------------------
# Homography caching
# ---------------------------------------------------------------------------


def test_homography_cached_on_same_corners(
    synthetic_board_frame: tuple[np.ndarray, np.ndarray],
) -> None:
    """Passing identical corners twice must not change the homography object."""
    frame, corners = synthetic_board_frame
    sorted_c = BoardDetector._sort_corners(corners)
    r = Rectifier()
    r.process(frame, _blank_mask(frame), corners=sorted_c)
    h1 = r._homography
    r.process(frame, _blank_mask(frame), corners=sorted_c.copy())
    h2 = r._homography
    assert h1 is h2, "Homography object should be reused when corners are unchanged"


def test_homography_updated_on_new_corners(
    synthetic_board_frame: tuple[np.ndarray, np.ndarray],
) -> None:
    """Passing shifted corners must produce a new homography object."""
    frame, corners = synthetic_board_frame
    sorted_c = BoardDetector._sort_corners(corners)
    r = Rectifier()
    r.process(frame, _blank_mask(frame), corners=sorted_c)
    h1 = r._homography

    shifted = sorted_c + 30.0
    r.process(frame, _blank_mask(frame), corners=shifted)
    h2 = r._homography
    assert h1 is not h2, "Homography must be recomputed when corners change"
