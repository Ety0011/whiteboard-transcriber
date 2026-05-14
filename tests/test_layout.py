"""Tests for Stage 4 — Layout Detection (src/layout.py).

All tests mock LayoutDetection so no model weights are downloaded and no camera
is required. Synthetic whiteboard images are created inline.

Architecture note
-----------------
Production: LayoutDetector uses multiprocessing.Process so PaddlePaddle runs
in a child process (separate GIL — avoids blocking cv2.waitKey on macOS).

Tests: the patch_process fixture replaces mp.Process with threading.Thread so
tests stay in-process and monkeypatching LayoutDetection works normally.

Filtering / crop tests call _run_detection() directly (synchronous).
Async-behaviour tests use _flush(), which submits a frame and polls process()
until the worker thread finishes.
"""

from __future__ import annotations

import queue
import threading
import time
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

import src.layout as layout_mod
from src.layout import LayoutDetector, LayoutRegion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _white_board(h: int = 480, w: int = 640) -> np.ndarray:
    """Return a white BGR image with two black filled rectangles."""
    img = np.ones((h, w, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (50, 50), (300, 150), (0, 0, 0), -1)
    cv2.rectangle(img, (50, 200), (500, 400), (0, 0, 0), -1)
    return img


def _make_det_result(boxes: list[dict]) -> MagicMock:
    result = MagicMock()
    result.get = lambda key, default=None: boxes if key == "boxes" else default
    return result


def _default_boxes() -> list[dict]:
    return [
        {"label": "text",  "score": 0.95, "coordinate": [50,  50,  300, 150]},
        {"label": "image", "score": 0.80, "coordinate": [50,  200, 500, 400]},
        {"label": "seal",  "score": 0.90, "coordinate": [10,  10,  40,  40]},  # bad label
        {"label": "text",  "score": 0.30, "coordinate": [0,   0,   100, 50]},  # low conf
    ]


def _make_detector(monkeypatch, boxes: list[dict]) -> LayoutDetector:
    """Return a LayoutDetector with a mocked engine returning *boxes*."""
    engine = MagicMock()
    engine.predict = MagicMock(return_value=[_make_det_result(boxes)])
    monkeypatch.setattr("src.layout.LayoutDetection", lambda **kw: engine)
    return LayoutDetector()


def _flush(detector: LayoutDetector, image: np.ndarray, timeout: float = 2.0) -> list[Region]:
    """Submit *image* (bypassing interval) and block until ONE detection completes.

    Polls _result_queue directly so the loop never triggers a second submission.
    Restores recompute_interval before returning.
    """
    saved = detector._recompute_interval
    detector._recompute_interval = 0.0
    detector.process(image)  # submit
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            box_dicts = detector._result_queue.get_nowait()
            if detector._pending_image is not None:
                detector._cached_regions = layout_mod._build_regions(
                    box_dicts, detector._pending_image
                )
            detector._detecting = False
            break
        except queue.Empty:
            time.sleep(0.005)
    detector._recompute_interval = saved
    return list(detector._cached_regions)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _FakeProcess(threading.Thread):
    """Drop-in replacement for mp.Process that runs the target in a thread.

    Keeps everything in-process so monkeypatching LayoutDetection works.
    """
    def __init__(self, target, args=(), daemon=None, name=None, **kwargs):
        super().__init__(target=target, args=args, daemon=daemon, name=name)


@pytest.fixture(autouse=True)
def patch_process(monkeypatch):
    """Replace mp.Process+mp.Queue with thread/queue equivalents.

    mp.Queue uses OS-level pipes that break when used inside threads on macOS.
    queue.Queue is identical in API and raises the same Empty exception class.
    """
    monkeypatch.setattr("src.layout.mp.Process", _FakeProcess)
    monkeypatch.setattr("src.layout.mp.Queue",
                        lambda maxsize=0: queue.Queue(maxsize=maxsize))


@pytest.fixture(autouse=True)
def reset_global(monkeypatch):
    """Reset the module-level singleton before each test."""
    monkeypatch.setattr(layout_mod, "_global_detector", None)


# ---------------------------------------------------------------------------
# LayoutRegion dataclass
# ---------------------------------------------------------------------------

def test_region_dataclass_fields():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    r = LayoutRegion(bbox=(0, 0, 10, 10), label="text", confidence=0.9, crop=img)
    assert r.bbox == (0, 0, 10, 10)
    assert r.label == "text"
    assert r.confidence == pytest.approx(0.9)
    assert r.crop is img


# ---------------------------------------------------------------------------
# _run_detection() — filtering (synchronous, no worker involved)
# ---------------------------------------------------------------------------

def test_run_detection_returns_only_valid_regions(monkeypatch):
    detector = _make_detector(monkeypatch, _default_boxes())
    results = detector._run_detection(_white_board())
    assert len(results) == 2
    assert {r.label for r in results} == {"text", "image"}


def test_label_filtering_discards_non_whiteboard(monkeypatch):
    detector = _make_detector(monkeypatch, _default_boxes())
    results = detector._run_detection(_white_board())
    for r in results:
        assert r.label in layout_mod._WHITEBOARD_LABELS


def test_confidence_filtering_discards_low_score(monkeypatch):
    detector = _make_detector(monkeypatch, _default_boxes())
    results = detector._run_detection(_white_board())
    for r in results:
        assert r.confidence >= 0.5


def test_custom_threshold_keeps_more(monkeypatch):
    boxes = [{"label": "text", "score": 0.35, "coordinate": [0, 0, 100, 50]}]
    detector = _make_detector(monkeypatch, boxes)
    detector._confidence_threshold = 0.3
    results = detector._run_detection(_white_board())
    assert len(results) == 1
    assert results[0].confidence == pytest.approx(0.35)


def test_crop_shape_matches_bbox(monkeypatch):
    detector = _make_detector(monkeypatch, _default_boxes())
    results = detector._run_detection(_white_board())
    for r in results:
        x1, y1, x2, y2 = r.bbox
        assert r.crop.shape == (y2 - y1, x2 - x1, 3)


def test_crop_is_copy_not_view(monkeypatch):
    detector = _make_detector(monkeypatch, _default_boxes())
    img = _white_board()
    results = detector._run_detection(img)
    assert not np.shares_memory(results[0].crop, img)


def test_empty_boxes_returns_empty_list(monkeypatch):
    detector = _make_detector(monkeypatch, [])
    assert detector._run_detection(_white_board()) == []


def test_predict_returns_empty_list(monkeypatch):
    engine = MagicMock()
    engine.predict = MagicMock(return_value=[])
    monkeypatch.setattr("src.layout.LayoutDetection", lambda **kw: engine)
    detector = LayoutDetector()
    assert detector._run_detection(_white_board()) == []


def test_missing_score_field_defaults_to_passing(monkeypatch):
    boxes = [{"label": "text", "coordinate": [10, 10, 100, 80]}]
    detector = _make_detector(monkeypatch, boxes)
    results = detector._run_detection(_white_board())
    assert len(results) == 1
    assert results[0].confidence == pytest.approx(1.0)


def test_malformed_coordinate_skipped(monkeypatch):
    boxes = [
        {"label": "text", "score": 0.9, "coordinate": [10, 20]},
        {"label": "text", "score": 0.9, "coordinate": [10, 20, 100, 80]},
    ]
    detector = _make_detector(monkeypatch, boxes)
    assert len(detector._run_detection(_white_board())) == 1


def test_bbox_clamped_to_image_bounds(monkeypatch):
    h, w = 480, 640
    boxes = [{"label": "text", "score": 0.9, "coordinate": [-10, -5, w + 50, h + 30]}]
    detector = _make_detector(monkeypatch, boxes)
    results = detector._run_detection(_white_board(h, w))
    assert len(results) == 1
    x1, y1, x2, y2 = results[0].bbox
    assert x1 >= 0 and y1 >= 0
    assert x2 <= w and y2 <= h


# ---------------------------------------------------------------------------
# process() — async / interval behaviour (uses _flush + worker thread)
# ---------------------------------------------------------------------------

def test_process_initially_returns_empty(monkeypatch):
    engine = MagicMock()
    engine.predict = MagicMock(side_effect=lambda img: time.sleep(10) or [])
    monkeypatch.setattr("src.layout.LayoutDetection", lambda **kw: engine)
    detector = LayoutDetector(recompute_interval=0.0)
    assert detector.process(_white_board()) == []


def test_process_returns_regions_after_flush(monkeypatch):
    detector = _make_detector(monkeypatch, _default_boxes())
    results = _flush(detector, _white_board())
    assert len(results) == 2


def test_process_respects_recompute_interval(monkeypatch):
    call_count = 0

    def _counting_predict(img):
        nonlocal call_count
        call_count += 1
        return []

    engine = MagicMock()
    engine.predict = _counting_predict
    monkeypatch.setattr("src.layout.LayoutDetection", lambda **kw: engine)

    detector = LayoutDetector(recompute_interval=10.0)
    _flush(detector, _white_board())       # first detection fires (interval bypassed)
    detector.process(_white_board())       # interval not elapsed — must NOT fire again

    assert call_count == 1


def test_process_fires_again_after_interval(monkeypatch):
    call_count = 0

    def _counting_predict(img):
        nonlocal call_count
        call_count += 1
        return []

    engine = MagicMock()
    engine.predict = _counting_predict
    monkeypatch.setattr("src.layout.LayoutDetection", lambda **kw: engine)

    detector = LayoutDetector(recompute_interval=0.0)
    _flush(detector, _white_board())
    _flush(detector, _white_board())

    assert call_count == 2


# ---------------------------------------------------------------------------
# Singleton behaviour
# ---------------------------------------------------------------------------

def test_init_creates_singleton(monkeypatch):
    engine = MagicMock()
    engine.predict = MagicMock(return_value=[])
    monkeypatch.setattr("src.layout.LayoutDetection", lambda **kw: engine)

    assert layout_mod._global_detector is None
    layout_mod.init()
    assert layout_mod._global_detector is not None


def test_process_lazy_creates_singleton(monkeypatch):
    engine = MagicMock()
    engine.predict = MagicMock(return_value=[])
    monkeypatch.setattr("src.layout.LayoutDetection", lambda **kw: engine)

    assert layout_mod._global_detector is None
    layout_mod.process(_white_board())
    assert layout_mod._global_detector is not None


def test_init_respects_confidence_threshold(monkeypatch):
    engine = MagicMock()
    engine.predict = MagicMock(return_value=[])
    monkeypatch.setattr("src.layout.LayoutDetection", lambda **kw: engine)

    layout_mod.init(confidence_threshold=0.7)
    assert layout_mod._global_detector._confidence_threshold == pytest.approx(0.7)


def test_init_respects_recompute_interval(monkeypatch):
    engine = MagicMock()
    engine.predict = MagicMock(return_value=[])
    monkeypatch.setattr("src.layout.LayoutDetection", lambda **kw: engine)

    layout_mod.init(recompute_interval=5.0)
    assert layout_mod._global_detector._recompute_interval == pytest.approx(5.0)
