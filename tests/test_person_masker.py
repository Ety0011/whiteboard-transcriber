"""Tests for Stage 2 — Person Masking (src/person_masker.py).

The entire module is skipped when MediaPipe is not installed so CI
environments without the heavy dependency stay green.

All model loading and MediaPipe inference is mocked — no camera, network
access, or model weights are required to run these tests.

Design: ``_make_masker()`` patches both ``_ensure_model`` (prevents
download) and ``ImageSegmenter.create_from_options`` (prevents model load),
then replaces ``_run_inference`` via the instance to control what the
masker "sees".
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

mediapipe = pytest.importorskip("mediapipe")

from src.person_masker import PersonMasker  # noqa: E402 — import after importorskip


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_masker(**kwargs) -> PersonMasker:
    """Return a PersonMasker with model download and MediaPipe init mocked out."""
    with (
        patch("src.person_masker._ensure_model", return_value=Path("models/fake.tflite")),
        patch.object(
            mediapipe.tasks.vision.ImageSegmenter,
            "create_from_options",
            return_value=MagicMock(),
        ),
    ):
        return PersonMasker(**kwargs)


def _zero_inference(shape: tuple[int, int]):
    """Return a _run_inference replacement that always yields a zero mask."""
    return lambda rgb: np.zeros(shape, dtype=np.float32)


def _const_inference(shape: tuple[int, int], value: float):
    """Return a _run_inference replacement that yields a uniform mask."""
    return lambda rgb: np.full(shape, value, dtype=np.float32)


def _sparse_inference(shape: tuple[int, int], row: int, col: int, value: float):
    """Return a _run_inference replacement with one nonzero pixel."""
    def _infer(rgb: np.ndarray) -> np.ndarray:
        m = np.zeros(shape, dtype=np.float32)
        m[row, col] = value
        return m
    return _infer


# ---------------------------------------------------------------------------
# Output shape / dtype / values
# ---------------------------------------------------------------------------


def test_output_shape(blank_board: np.ndarray) -> None:
    """process() returns a 2-D mask whose dimensions match the input frame."""
    masker = _make_masker()
    masker._run_inference = _zero_inference((720, 1280))
    assert masker.process(blank_board).shape == (720, 1280)


def test_output_dtype(blank_board: np.ndarray) -> None:
    """Mask dtype must be uint8."""
    masker = _make_masker()
    masker._run_inference = _zero_inference((720, 1280))
    assert masker.process(blank_board).dtype == np.uint8


def test_output_is_binary(blank_board: np.ndarray) -> None:
    """All mask values must be exactly 0 or 1 — no intermediate floats."""
    masker = _make_masker()
    masker._run_inference = _const_inference((720, 1280), 0.3)
    mask = masker.process(blank_board)
    assert set(np.unique(mask).tolist()) <= {0, 1}


# ---------------------------------------------------------------------------
# No-person result
# ---------------------------------------------------------------------------


def test_no_person_returns_zero_mask(blank_board: np.ndarray) -> None:
    """When inference returns all zeros, the final mask must also be all zeros."""
    masker = _make_masker()
    masker._run_inference = _zero_inference((720, 1280))
    assert masker.process(blank_board).sum() == 0


# ---------------------------------------------------------------------------
# Dilation behaviour
# ---------------------------------------------------------------------------


def test_dilation_expands_mask() -> None:
    """A single-pixel person region should expand after elliptical dilation.

    An ellipse with radius 5 covers at least π×5² ≈ 78 pixels.
    """
    h, w = 100, 100
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    masker = _make_masker(dilation_px=5)
    masker._run_inference = _sparse_inference((h, w), 50, 50, 1.0)

    mask = masker.process(frame)
    assert mask.sum() >= 50, f"Dilation produced too few pixels: {mask.sum()}"


def test_no_dilation_when_disabled() -> None:
    """With dilation_px=0 a single-pixel confidence patch stays one pixel."""
    h, w = 100, 100
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    masker = _make_masker(dilation_px=0)
    masker._run_inference = _sparse_inference((h, w), 50, 50, 1.0)

    mask = masker.process(frame)
    assert mask.sum() == 1


# ---------------------------------------------------------------------------
# Threshold behaviour
# ---------------------------------------------------------------------------


def test_threshold_includes_pixel_above_cutoff() -> None:
    """A pixel with confidence 0.4 is included when threshold=0.3."""
    h, w = 50, 50
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    masker = _make_masker(threshold=0.3, dilation_px=0)
    masker._run_inference = _sparse_inference((h, w), 25, 25, 0.4)

    assert masker.process(frame)[25, 25] == 1


def test_threshold_excludes_pixel_below_cutoff() -> None:
    """A pixel with confidence 0.4 is excluded when threshold=0.5."""
    h, w = 50, 50
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    masker = _make_masker(threshold=0.5, dilation_px=0)
    masker._run_inference = _sparse_inference((h, w), 25, 25, 0.4)

    assert masker.process(frame)[25, 25] == 0
