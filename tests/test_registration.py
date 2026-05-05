"""Tests for Stage 1 — Spatial Registration (src/registration.py).

All tests use synthetic fixture images so no camera is required.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.registration import Registrar


# ---------------------------------------------------------------------------
# Output shape / dtype
# ---------------------------------------------------------------------------


def test_output_shape_blank(blank_board: np.ndarray) -> None:
    """process() always returns an image of the canonical output size."""
    result = Registrar().process(blank_board)
    assert result.shape == (720, 1280, 3)


def test_output_dtype_blank(blank_board: np.ndarray) -> None:
    """process() output is uint8 BGR."""
    result = Registrar().process(blank_board)
    assert result.dtype == np.uint8


# ---------------------------------------------------------------------------
# Board detection on a synthetic perspective frame
# ---------------------------------------------------------------------------


def test_warp_on_synthetic_board(
    synthetic_board_frame: tuple[np.ndarray, np.ndarray],
) -> None:
    """process() detects the synthetic board, updates the homography cache,
    and returns an image at the canonical resolution."""
    frame, _ = synthetic_board_frame
    r = Registrar()
    result = r.process(frame)

    assert result.shape == (720, 1280, 3)
    assert result.dtype == np.uint8
    assert r._homography is not None, "Homography should be cached after detection"
    assert r._cached_corners is not None


def test_warp_output_is_mostly_light(
    synthetic_board_frame: tuple[np.ndarray, np.ndarray],
) -> None:
    """After warping a frame whose board is off-white, the output should be
    predominantly light (mean brightness > 128)."""
    frame, _ = synthetic_board_frame
    result = Registrar().process(frame)
    assert result.mean() > 128, "Warped board should be predominantly bright"


# ---------------------------------------------------------------------------
# Homography caching
# ---------------------------------------------------------------------------


def test_homography_cached_on_second_call(
    synthetic_board_frame: tuple[np.ndarray, np.ndarray],
) -> None:
    """The cached corners object must be identical (same id) after a second
    call with the same frame — no recomputation should occur."""
    frame, _ = synthetic_board_frame
    r = Registrar()

    r.process(frame)
    corners_after_first = r._cached_corners

    r.process(frame)
    corners_after_second = r._cached_corners

    # Same object in memory → no recompute
    assert corners_after_first is corners_after_second


def test_cache_invalidation_on_shifted_corners(
    synthetic_board_frame: tuple[np.ndarray, np.ndarray],
) -> None:
    """When we manually shift the cached corners by > threshold, the next
    call must recompute and produce a new homography array."""
    frame, _ = synthetic_board_frame
    r = Registrar(cache_threshold=20.0)

    r.process(frame)
    original_homography = r._homography

    # Shift cached corners far enough to exceed the threshold
    assert r._cached_corners is not None
    r._cached_corners = r._cached_corners + 50.0  # +50 px shift

    r.process(frame)
    new_homography = r._homography

    assert new_homography is not original_homography, (
        "Homography must be recomputed when cached corners shift beyond threshold"
    )


# ---------------------------------------------------------------------------
# Debug mode
# ---------------------------------------------------------------------------


def test_debug_mode_no_crash(
    synthetic_board_frame: tuple[np.ndarray, np.ndarray],
) -> None:
    """Registrar(debug=True).process() completes without exception and
    returns the canonical output shape."""
    frame, _ = synthetic_board_frame
    result = Registrar(debug=True).process(frame)
    assert result.shape == (720, 1280, 3)


# ---------------------------------------------------------------------------
# Fallback when no board is detected
# ---------------------------------------------------------------------------


def test_fallback_returns_output_size_when_no_board() -> None:
    """A featureless gray frame has no detectable board quad. process() must
    still return an image at the canonical size (resized fallback)."""
    plain_gray = np.full((720, 1280, 3), 128, dtype=np.uint8)
    result = Registrar().process(plain_gray)
    assert result.shape == (720, 1280, 3)
