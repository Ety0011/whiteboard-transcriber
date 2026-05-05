"""Tests for Stage 1 — Spatial Registration (src/registration.py).

SAM2 model loading is mocked in all tests so no model download or GPU is
required. Tests are split into two groups:

1. Geometric tests — call ``Registrar._mask_to_corners`` and ``_sort_corners``
   directly. No mocking needed; these are pure NumPy/OpenCV functions.

2. Pipeline tests — instantiate ``Registrar`` via ``_make_registrar()``
   (which patches ``SAM.__init__``) and drive ``process()`` by either
   pre-populating the corner cache or patching ``_detect_corners``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from src.registration import Registrar


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_registrar(**kwargs) -> Registrar:
    """Return a Registrar with SAM model loading patched out."""
    with patch("src.registration.SAM"):
        return Registrar(**kwargs)


def _make_quad_mask(
    h: int, w: int, corners: np.ndarray
) -> np.ndarray:
    """Return a binary uint8 mask with a filled quadrilateral at *corners*."""
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [corners.astype(np.int32)], 1)
    return mask


# ---------------------------------------------------------------------------
# Geometric tests — _mask_to_corners (no mocking required)
# ---------------------------------------------------------------------------


def test_mask_to_corners_synthetic() -> None:
    """Clean quad mask → exactly 4 corners returned."""
    corners = np.array([[150, 80], [1100, 50], [1130, 640], [120, 660]], np.int32)
    mask = _make_quad_mask(720, 1280, corners)
    result = Registrar._mask_to_corners(mask)
    assert result is not None
    assert result.shape == (4, 2)
    assert result.dtype == np.float32


def test_mask_to_corners_empty_mask() -> None:
    """All-zero mask → None (no contour to extract)."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    assert Registrar._mask_to_corners(mask) is None


def test_mask_to_corners_single_pixel() -> None:
    """A single-pixel mask has no meaningful polygon → None."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[50, 50] = 1
    assert Registrar._mask_to_corners(mask) is None


def test_sort_corners_order() -> None:
    """_sort_corners returns TL, TR, BR, BL regardless of input order."""
    pts = np.array(
        [[1100, 50], [120, 660], [150, 80], [1130, 640]], dtype=np.float32
    )
    rect = Registrar._sort_corners(pts)
    tl, tr, br, bl = rect
    assert tl[0] < tr[0]  # TL is left of TR
    assert bl[0] < br[0]  # BL is left of BR
    assert tl[1] < bl[1]  # TL is above BL
    assert tr[1] < br[1]  # TR is above BR


# ---------------------------------------------------------------------------
# Pipeline tests — output shape / dtype / fallback
# ---------------------------------------------------------------------------


def test_output_shape_no_detection(blank_board: np.ndarray) -> None:
    """With no board detected the fallback resize still returns output_size."""
    r = _make_registrar()
    with patch.object(r, "_detect_corners", return_value=None):
        assert r.process(blank_board).shape == (720, 1280, 3)


def test_output_dtype_no_detection(blank_board: np.ndarray) -> None:
    """Output dtype is always uint8."""
    r = _make_registrar()
    with patch.object(r, "_detect_corners", return_value=None):
        assert r.process(blank_board).dtype == np.uint8


def test_output_shape_with_known_corners(
    synthetic_board_frame: tuple[np.ndarray, np.ndarray],
) -> None:
    """With pre-set corners, warp produces the canonical output shape."""
    frame, corners = synthetic_board_frame
    r = _make_registrar()
    sorted_c = Registrar._sort_corners(corners)
    r._cached_corners = sorted_c
    r._homography = r._compute_homography(sorted_c)
    # Suppress detection so the pre-set homography is used as-is
    with patch.object(r, "_detect_corners", return_value=None):
        assert r.process(frame).shape == (720, 1280, 3)


def test_warp_output_is_mostly_light(
    synthetic_board_frame: tuple[np.ndarray, np.ndarray],
) -> None:
    """Warping a frame whose board is off-white should produce a bright output."""
    frame, corners = synthetic_board_frame
    r = _make_registrar()
    sorted_c = Registrar._sort_corners(corners)
    r._cached_corners = sorted_c
    r._homography = r._compute_homography(sorted_c)
    with patch.object(r, "_detect_corners", return_value=None):
        assert r.process(frame).mean() > 128


# ---------------------------------------------------------------------------
# Pipeline tests — detection flow via patched _detect_corners
# ---------------------------------------------------------------------------


def test_homography_set_after_detection(
    synthetic_board_frame: tuple[np.ndarray, np.ndarray],
) -> None:
    """When _detect_corners returns corners, the homography cache is populated."""
    frame, corners = synthetic_board_frame
    r = _make_registrar()
    with patch.object(r, "_detect_corners", return_value=corners):
        r.process(frame)
    assert r._homography is not None
    assert r._cached_corners is not None


def test_homography_cached_on_second_call(
    synthetic_board_frame: tuple[np.ndarray, np.ndarray],
) -> None:
    """_cached_corners must be the same object on consecutive calls with the
    same corners (no recompute if nothing shifted)."""
    frame, corners = synthetic_board_frame
    r = _make_registrar(recompute_every=1)  # detect every call for this test
    with patch.object(r, "_detect_corners", return_value=corners.copy()):
        r.process(frame)
        first = r._cached_corners
        r.process(frame)
        second = r._cached_corners
    assert first is second


def test_cache_invalidation_on_shifted_corners(
    synthetic_board_frame: tuple[np.ndarray, np.ndarray],
) -> None:
    """Corners shifted > threshold must trigger a new homography object."""
    frame, corners = synthetic_board_frame
    r = _make_registrar(recompute_every=1)
    with patch.object(r, "_detect_corners", return_value=corners.copy()):
        r.process(frame)
    original_H = r._homography

    shifted = corners.copy() + 50.0  # well beyond 20 px threshold
    with patch.object(r, "_detect_corners", return_value=shifted):
        r.process(frame)
    assert r._homography is not original_H


def test_recompute_interval_respected(
    synthetic_board_frame: tuple[np.ndarray, np.ndarray],
) -> None:
    """_detect_corners is called only on the first frame and then every
    recompute_every frames, not on every single call."""
    frame, corners = synthetic_board_frame
    r = _make_registrar(recompute_every=5)

    call_count = 0

    def counting_detect(f: np.ndarray):
        nonlocal call_count
        call_count += 1
        return corners.copy()

    with patch.object(r, "_detect_corners", side_effect=counting_detect):
        for _ in range(11):  # 11 calls: detect on call 1, 6, 11
            r.process(frame)

    assert call_count == 3


def test_debug_mode_no_crash(
    synthetic_board_frame: tuple[np.ndarray, np.ndarray],
) -> None:
    """debug=True completes without exception and returns the canonical shape."""
    frame, corners = synthetic_board_frame
    r = _make_registrar(debug=True)
    with patch.object(r, "_detect_corners", return_value=corners):
        result = r.process(frame)
    assert result.shape == (720, 1280, 3)


def test_fallback_returns_output_size_when_no_board() -> None:
    """Detection returning None falls back to resize, still correct shape."""
    plain_gray = np.full((720, 1280, 3), 128, dtype=np.uint8)
    r = _make_registrar()
    with patch.object(r, "_detect_corners", return_value=None):
        result = r.process(plain_gray)
    assert result.shape == (720, 1280, 3)
