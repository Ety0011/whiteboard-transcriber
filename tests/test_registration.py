"""Tests for Stage 1 — Spatial Registration (src/registration.py).

Both SAM model loads are mocked in all tests so no model download or GPU is
required. Tests are split into three groups:

1. Geometric tests — call ``Registrar._mask_to_corners`` and ``_sort_corners``
   directly. No mocking needed; these are pure NumPy/OpenCV functions.

2. Pipeline tests — instantiate ``Registrar`` via ``_make_registrar()``
   (which patches ``SAM.__init__``) and drive ``process()`` by either
   pre-populating the corner cache or patching ``_detect_corners``.

3. Movement-check tests — test ``_has_board_moved()`` by mocking the
   ``_sam_tracker`` callable and verifying IoU comparisons.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from src.registration import Registrar


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_registrar(**kwargs) -> Registrar:
    """Return a Registrar with both SAM model loads patched out."""
    with patch("src.registration.SAM"):
        return Registrar(**kwargs)


def _make_quad_mask(h: int, w: int, corners: np.ndarray) -> np.ndarray:
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
    assert tl[0] < tr[0]   # TL is left of TR
    assert bl[0] < br[0]   # BL is left of BR
    assert tl[1] < bl[1]   # TL is above BL
    assert tr[1] < br[1]   # TR is above BR


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
# Pipeline tests — detection flow
# ---------------------------------------------------------------------------


def test_homography_set_after_detection(
    synthetic_board_frame: tuple[np.ndarray, np.ndarray],
) -> None:
    """When _detect_corners returns corners, the homography cache is populated."""
    frame, corners = synthetic_board_frame
    r = _make_registrar()
    with patch.object(r, "_detect_corners", return_value=corners):
        r.process(frame)
    # Give background thread time to run
    time.sleep(0.05)
    assert r._homography is not None
    assert r._cached_corners is not None


def test_homography_cached_on_second_call(
    synthetic_board_frame: tuple[np.ndarray, np.ndarray],
) -> None:
    """Same corners on consecutive calls must not replace _cached_corners."""
    frame, corners = synthetic_board_frame
    r = _make_registrar(recompute_every=1)
    with patch.object(r, "_detect_corners", return_value=corners.copy()):
        r.process(frame)
        time.sleep(0.05)
        first = r._cached_corners
        r.process(frame)
        time.sleep(0.05)
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
    time.sleep(0.05)
    original_H = r._homography

    shifted = corners.copy() + 50.0
    with patch.object(r, "_detect_corners", return_value=shifted):
        r.process(frame)
    time.sleep(0.05)
    assert r._homography is not original_H


def test_recompute_interval_respected(
    synthetic_board_frame: tuple[np.ndarray, np.ndarray],
) -> None:
    """_detect_corners fires on call 1 and then every recompute_every frames.
    The SAM 2.1 movement check must not fire (ref_mask starts as None)."""
    frame, corners = synthetic_board_frame
    r = _make_registrar(recompute_every=5)

    call_count = 0

    def counting_detect(f: np.ndarray):
        nonlocal call_count
        call_count += 1
        return corners.copy()

    with patch.object(r, "_detect_corners", side_effect=counting_detect):
        for _ in range(11):   # detect on calls 1, 6, 11
            r.process(frame)
    time.sleep(0.1)
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


# ---------------------------------------------------------------------------
# SAM 3 text-prompt verification
# ---------------------------------------------------------------------------


def test_detect_corners_uses_center_point_prompt(blank_board: np.ndarray) -> None:
    """_detect_corners must call SAM 3 with a centre-point prompt."""
    r = _make_registrar()
    mock_result = MagicMock()
    mock_result.masks = None
    r._sam.return_value = [mock_result]
    r._detect_corners(blank_board)
    h, w = blank_board.shape[:2]
    r._sam.assert_called_once_with(
        blank_board, points=[[w // 2, h // 2]], labels=[1], verbose=False
    )


def test_detect_corners_stores_last_mask(blank_board: np.ndarray) -> None:
    """After a successful detection, _last_mask is set."""
    r = _make_registrar()
    mask = np.ones((720, 1280), dtype=np.float32)
    mock_result = MagicMock()
    mock_result.masks = MagicMock()
    mock_result.masks.data.cpu().numpy.return_value = mask[np.newaxis]
    r._sam.return_value = [mock_result]

    # Use a quad-shaped mask so _mask_to_corners can find 4 corners
    corners = np.array([[150, 80], [1100, 50], [1130, 640], [120, 660]], np.int32)
    quad_mask = _make_quad_mask(720, 1280, corners).astype(np.float32)
    mock_result.masks.data.cpu().numpy.return_value = quad_mask[np.newaxis]

    r._detect_corners(blank_board)
    assert r._last_mask is not None


# ---------------------------------------------------------------------------
# _has_board_moved tests
# ---------------------------------------------------------------------------


class TestHasBoardMoved:
    """Tests for Registrar._has_board_moved() via mocked _sam_tracker."""

    MASK_H, MASK_W = 720, 1280

    def _make_registrar_with_ref(self, ref_mask: np.ndarray) -> Registrar:
        r = _make_registrar()
        r._ref_mask = ref_mask
        return r

    def _tracker_result(self, mask: np.ndarray | None):
        """Build a mock _sam_tracker return value."""
        result = MagicMock()
        if mask is None:
            result.masks = None
        else:
            result.masks = MagicMock()
            result.masks.data.cpu().numpy.return_value = mask[np.newaxis].astype(np.float32)
        return [result]

    def test_returns_true_when_tracker_returns_no_mask(
        self, blank_board: np.ndarray
    ) -> None:
        ref = np.ones((self.MASK_H, self.MASK_W), dtype=np.uint8)
        r = self._make_registrar_with_ref(ref)
        r._sam_tracker.return_value = self._tracker_result(None)
        assert r._has_board_moved(blank_board) is True

    def test_returns_false_when_masks_identical(
        self, blank_board: np.ndarray
    ) -> None:
        ref = np.ones((self.MASK_H, self.MASK_W), dtype=np.uint8)
        r = self._make_registrar_with_ref(ref)
        r._sam_tracker.return_value = self._tracker_result(ref.copy())
        assert r._has_board_moved(blank_board) is False

    def test_returns_true_when_iou_below_threshold(
        self, blank_board: np.ndarray
    ) -> None:
        """Non-overlapping masks → IoU = 0 → board moved."""
        ref = np.zeros((self.MASK_H, self.MASK_W), dtype=np.uint8)
        ref[:, : self.MASK_W // 2] = 1          # left half
        current = np.zeros_like(ref)
        current[:, self.MASK_W // 2 :] = 1      # right half — no overlap
        r = self._make_registrar_with_ref(ref)
        r._iou_threshold = 0.80
        r._sam_tracker.return_value = self._tracker_result(current)
        assert r._has_board_moved(blank_board) is True

    def test_returns_false_when_iou_above_threshold(
        self, blank_board: np.ndarray
    ) -> None:
        """Masks that differ only by a thin strip → IoU ≈ 0.986 → no movement."""
        ref = np.ones((self.MASK_H, self.MASK_W), dtype=np.uint8)
        current = ref.copy()
        current[:10, :] = 0   # remove top 10 rows
        r = self._make_registrar_with_ref(ref)
        r._iou_threshold = 0.80
        r._sam_tracker.return_value = self._tracker_result(current)
        assert r._has_board_moved(blank_board) is False

    def test_returns_true_when_empty_mask_tensor(
        self, blank_board: np.ndarray
    ) -> None:
        """Empty mask tensor (0, H, W) → conservative True."""
        ref = np.ones((self.MASK_H, self.MASK_W), dtype=np.uint8)
        r = self._make_registrar_with_ref(ref)
        result = MagicMock()
        result.masks = MagicMock()
        result.masks.data.cpu().numpy.return_value = np.zeros(
            (0, self.MASK_H, self.MASK_W), dtype=np.float32
        )
        r._sam_tracker.return_value = [result]
        assert r._has_board_moved(blank_board) is True

    def test_movement_check_triggers_sam3(
        self, synthetic_board_frame: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """When has_board_moved returns True inside _detection_loop, SAM 3
        runs and updates the homography."""
        frame, corners = synthetic_board_frame
        r = _make_registrar(recompute_every=9999, check_every=1)

        # Pre-populate ref_mask so the check branch activates
        r._ref_mask = np.ones((720, 1280), dtype=np.uint8)

        detect_called = []

        def fake_detect(f):
            detect_called.append(True)
            return corners.copy()

        with (
            patch.object(r, "_has_board_moved", return_value=True),
            patch.object(r, "_detect_corners", side_effect=fake_detect),
        ):
            r.process(frame)
            time.sleep(0.1)

        assert detect_called, "_detect_corners should have been called after movement"

    def test_no_check_before_first_sam3_run(
        self, blank_board: np.ndarray
    ) -> None:
        """Movement check must not fire before SAM 3 has produced a ref_mask."""
        r = _make_registrar(recompute_every=9999, check_every=1)
        # _ref_mask is None — the check branch is gated and must not fire
        check_called = []
        with patch.object(r, "_has_board_moved", side_effect=lambda f: check_called.append(True)):
            for _ in range(5):
                r.process(blank_board)
        assert not check_called, "_has_board_moved must not be called before _ref_mask is set"
