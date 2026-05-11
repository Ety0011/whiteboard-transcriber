"""Tests for Stage 5 — Text Line Detection (src/text_detection.py).

All tests mock TextDetection so no model weights are downloaded and no camera
is required. Synthetic crops are created inline.

Architecture note
-----------------
Production: TextDetector uses multiprocessing.Process so PaddlePaddle runs in
a child process (separate GIL — avoids blocking cv2.waitKey on macOS).

Tests: the patch_process fixture replaces mp.Process with threading.Thread so
tests stay in-process and monkeypatching TextDetection works normally.

Filtering / crop tests call _run_detection() directly (synchronous).
Async-behaviour tests use _flush(), which submits regions and polls the result
queue until the worker thread finishes — the same pattern as test_layout.py.
"""

from __future__ import annotations

import queue
import threading
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

import src.text_detection as td_mod
from src.layout import Region
from src.text_detection import (
    RegionWithLines,
    TextDetector,
    TextLine,
    _build_regions_with_lines,
    _parse_lines,
    _polygon_to_bbox,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _crop(h: int = 100, w: int = 200) -> np.ndarray:
    """Return a white BGR crop."""
    return np.ones((h, w, 3), dtype=np.uint8) * 255


def _region(label: str = "text", h: int = 100, w: int = 200) -> Region:
    return Region(bbox=(0, 0, w, h), label=label, confidence=0.9, crop=_crop(h, w))


def _make_detector(monkeypatch, raw_result: list) -> TextDetector:
    """Return a TextDetector whose engine returns raw_result on predict()."""
    engine = MagicMock()
    engine.predict = MagicMock(return_value=raw_result)
    monkeypatch.setattr("src.text_detection.TextDetection", lambda **kw: engine)
    return TextDetector()


def _rect_poly(x1: int, y1: int, x2: int, y2: int) -> list[list[int]]:
    """Four-point polygon for a rectangle."""
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def _flush(
    detector: TextDetector,
    regions: list[Region],
    timeout: float = 2.0,
) -> list[RegionWithLines]:
    """Submit *regions* (bypassing idle check) and block until ONE detection completes.

    Polls _result_queue directly so the loop never triggers a second submission.
    """
    detector._detecting = False
    detector.process(regions)  # submit
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            poly_lists = detector._result_queue.get_nowait()
            if detector._pending_regions is not None:
                detector._cached_results = _build_regions_with_lines(
                    poly_lists, detector._pending_regions
                )
            detector._detecting = False
            break
        except queue.Empty:
            time.sleep(0.005)
    return list(detector._cached_results)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _FakeProcess(threading.Thread):
    """Drop-in replacement for mp.Process that runs the target in a thread.

    Keeps everything in-process so monkeypatching TextDetection works.
    """

    def __init__(self, target, args=(), daemon=None, name=None, **kwargs):
        super().__init__(target=target, args=args, daemon=daemon, name=name)


@pytest.fixture(autouse=True)
def patch_process(monkeypatch):
    """Replace mp.Process+mp.Queue with thread/queue equivalents.

    mp.Queue uses OS-level pipes that break when used inside threads on macOS.
    queue.Queue is identical in API and raises the same Empty exception class.
    """
    monkeypatch.setattr("src.text_detection.mp.Process", _FakeProcess)
    monkeypatch.setattr(
        "src.text_detection.mp.Queue", lambda maxsize=0: queue.Queue(maxsize=maxsize)
    )


@pytest.fixture(autouse=True)
def reset_global(monkeypatch):
    """Reset the module-level singleton before each test."""
    monkeypatch.setattr(td_mod, "_global_detector", None)


# ---------------------------------------------------------------------------
# TextLine dataclass
# ---------------------------------------------------------------------------


def test_textline_fields():
    img = np.zeros((10, 20, 3), dtype=np.uint8)
    line = TextLine(bbox=(0, 0, 20, 10), crop=img)
    assert line.bbox == (0, 0, 20, 10)
    assert line.crop is img


# ---------------------------------------------------------------------------
# RegionWithLines dataclass
# ---------------------------------------------------------------------------


def test_regionwithlines_inherits_region_fields():
    crop = _crop()
    r = RegionWithLines(bbox=(0, 0, 200, 100), label="text", confidence=0.85, crop=crop)
    assert r.bbox == (0, 0, 200, 100)
    assert r.label == "text"
    assert r.confidence == pytest.approx(0.85)
    assert r.crop is crop
    assert r.lines == []


def test_regionwithlines_lines_field():
    crop = _crop()
    line = TextLine(bbox=(5, 5, 45, 15), crop=_crop(10, 40))
    r = RegionWithLines(
        bbox=(0, 0, 200, 100), label="text", confidence=0.9, crop=crop, lines=[line]
    )
    assert len(r.lines) == 1
    assert r.lines[0] is line


def test_regionwithlines_default_lines_not_shared():
    crop = _crop()
    r1 = RegionWithLines(bbox=(0, 0, 200, 100), label="text", confidence=0.9, crop=crop)
    r2 = RegionWithLines(bbox=(0, 0, 200, 100), label="text", confidence=0.9, crop=crop)
    r1.lines.append(TextLine(bbox=(0, 0, 10, 10), crop=_crop(10, 10)))
    assert r2.lines == []


# ---------------------------------------------------------------------------
# _polygon_to_bbox
# ---------------------------------------------------------------------------


def test_polygon_to_bbox_basic():
    assert _polygon_to_bbox([[10, 20], [80, 20], [80, 60], [10, 60]], 100, 200) == (
        10, 20, 80, 60,
    )


def test_polygon_to_bbox_clamps_to_image():
    assert _polygon_to_bbox([[-5, -10], [250, -10], [250, 120], [-5, 120]], 100, 200) == (
        0, 0, 200, 100,
    )


def test_polygon_to_bbox_partial_clamp():
    x1, y1, x2, y2 = _polygon_to_bbox([[190, 90], [210, 90], [210, 110], [190, 110]], 100, 200)
    assert (x1, y1, x2, y2) == (190, 90, 200, 100)


def test_polygon_to_bbox_non_rectangular():
    x1, y1, x2, y2 = _polygon_to_bbox([[10, 30], [90, 10], [110, 70], [30, 90]], 200, 200)
    assert (x1, y1, x2, y2) == (10, 10, 110, 90)


# ---------------------------------------------------------------------------
# _parse_lines
# ---------------------------------------------------------------------------


def test_parse_lines_empty_result():
    assert _parse_lines([], _crop()) == []


def test_parse_lines_no_polys():
    assert _parse_lines([{"dt_polys": []}], _crop()) == []


def test_parse_lines_single_line():
    lines = _parse_lines([{"dt_polys": [_rect_poly(10, 20, 80, 50)]}], _crop(100, 200))
    assert len(lines) == 1
    assert lines[0].bbox == (10, 20, 80, 50)


def test_parse_lines_crop_shape_matches_bbox():
    line = _parse_lines([{"dt_polys": [_rect_poly(10, 20, 80, 50)]}], _crop(100, 200))[0]
    x1, y1, x2, y2 = line.bbox
    assert line.crop.shape == (y2 - y1, x2 - x1, 3)


def test_parse_lines_crop_is_copy():
    crop = _crop(100, 200)
    line = _parse_lines([{"dt_polys": [_rect_poly(10, 20, 80, 50)]}], crop)[0]
    assert not np.shares_memory(line.crop, crop)


def test_parse_lines_degenerate_bbox_skipped():
    crop = _crop(100, 200)
    lines = _parse_lines(
        [{"dt_polys": [_rect_poly(10, 30, 80, 30), _rect_poly(10, 50, 80, 80)]}], crop
    )
    assert len(lines) == 1
    assert lines[0].bbox == (10, 50, 80, 80)


def test_parse_lines_multiple_lines():
    crop = _crop(200, 400)
    polys = [_rect_poly(0, 0, 100, 30), _rect_poly(0, 50, 200, 80), _rect_poly(0, 100, 300, 130)]
    assert len(_parse_lines([{"dt_polys": polys}], crop)) == 3


# ---------------------------------------------------------------------------
# _run_detection() — synchronous, no worker involved
# ---------------------------------------------------------------------------


def test_run_detection_returns_textlines(monkeypatch):
    detector = _make_detector(monkeypatch, [{"dt_polys": [_rect_poly(5, 10, 60, 40)]}])
    lines = detector._run_detection(_crop())
    assert len(lines) == 1
    assert isinstance(lines[0], TextLine)
    assert lines[0].bbox == (5, 10, 60, 40)


def test_run_detection_empty_result(monkeypatch):
    detector = _make_detector(monkeypatch, [])
    assert detector._run_detection(_crop()) == []


def test_run_detection_crop_shape_matches_bbox(monkeypatch):
    detector = _make_detector(monkeypatch, [{"dt_polys": [_rect_poly(10, 15, 90, 55)]}])
    lines = detector._run_detection(_crop())
    x1, y1, x2, y2 = lines[0].bbox
    assert lines[0].crop.shape == (y2 - y1, x2 - x1, 3)


def test_run_detection_crop_is_copy(monkeypatch):
    detector = _make_detector(monkeypatch, [{"dt_polys": [_rect_poly(10, 15, 90, 55)]}])
    crop = _crop()
    assert not np.shares_memory(detector._run_detection(crop)[0].crop, crop)


# ---------------------------------------------------------------------------
# process() — async / worker behaviour (uses _flush)
# ---------------------------------------------------------------------------


def test_process_initially_returns_empty(monkeypatch):
    engine = MagicMock()
    engine.predict = MagicMock(side_effect=lambda img: time.sleep(10) or [])
    monkeypatch.setattr("src.text_detection.TextDetection", lambda **kw: engine)
    detector = TextDetector()
    assert detector.process([_region()]) == []


def test_process_returns_results_after_flush(monkeypatch):
    detector = _make_detector(monkeypatch, [{"dt_polys": [_rect_poly(5, 10, 60, 40)]}])
    results = _flush(detector, [_region("text")])
    assert len(results) == 1
    assert len(results[0].lines) == 1


def test_process_does_not_resubmit_while_detecting(monkeypatch):
    call_count = 0

    def _counting_predict(img):
        nonlocal call_count
        call_count += 1
        return []

    engine = MagicMock()
    engine.predict = _counting_predict
    monkeypatch.setattr("src.text_detection.TextDetection", lambda **kw: engine)
    detector = TextDetector()

    _flush(detector, [_region()])           # first detection
    detector._detecting = True              # simulate busy worker
    detector.process([_region()])           # must be ignored
    detector.process([_region()])           # must be ignored
    detector._detecting = False

    assert call_count == 1


def test_process_resubmits_after_detection_completes(monkeypatch):
    call_count = 0

    def _counting_predict(img):
        nonlocal call_count
        call_count += 1
        return []

    engine = MagicMock()
    engine.predict = _counting_predict
    monkeypatch.setattr("src.text_detection.TextDetection", lambda **kw: engine)
    detector = TextDetector()

    _flush(detector, [_region()])
    _flush(detector, [_region()])

    assert call_count == 2


# ---------------------------------------------------------------------------
# process() — region-level behaviour (via _flush)
# ---------------------------------------------------------------------------


def test_process_text_region_gets_lines(monkeypatch):
    detector = _make_detector(monkeypatch, [{"dt_polys": [_rect_poly(5, 10, 60, 40)]}])
    results = _flush(detector, [_region("text")])
    assert len(results[0].lines) == 1


def test_process_paragraph_title_gets_lines(monkeypatch):
    detector = _make_detector(monkeypatch, [{"dt_polys": [_rect_poly(5, 10, 60, 40)]}])
    results = _flush(detector, [_region("paragraph_title")])
    assert len(results[0].lines) == 1


def test_process_figure_region_skipped(monkeypatch):
    engine = MagicMock()
    engine.predict = MagicMock(return_value=[{"dt_polys": [_rect_poly(0, 0, 50, 30)]}])
    monkeypatch.setattr("src.text_detection.TextDetection", lambda **kw: engine)
    detector = TextDetector()
    results = _flush(detector, [_region("figure")])
    assert results[0].lines == []
    engine.predict.assert_not_called()


def test_process_table_region_skipped(monkeypatch):
    engine = MagicMock()
    engine.predict = MagicMock(return_value=[{"dt_polys": [_rect_poly(0, 0, 50, 30)]}])
    monkeypatch.setattr("src.text_detection.TextDetection", lambda **kw: engine)
    detector = TextDetector()
    results = _flush(detector, [_region("table")])
    assert results[0].lines == []
    engine.predict.assert_not_called()


def test_process_mixed_regions(monkeypatch):
    detector = _make_detector(monkeypatch, [{"dt_polys": [_rect_poly(5, 10, 60, 40)]}])
    results = _flush(detector, [_region("text"), _region("figure"), _region("text")])
    assert len(results) == 3
    assert len(results[0].lines) == 1
    assert results[1].lines == []
    assert len(results[2].lines) == 1


def test_process_preserves_region_fields(monkeypatch):
    detector = _make_detector(monkeypatch, [])
    source = _region("text", h=80, w=160)
    result = _flush(detector, [source])[0]
    assert result.bbox == source.bbox
    assert result.label == source.label
    assert result.confidence == pytest.approx(source.confidence)
    assert result.crop is source.crop


def test_process_empty_regions_list(monkeypatch):
    detector = _make_detector(monkeypatch, [])
    assert _flush(detector, []) == []


def test_process_returns_regionwithlines_instances(monkeypatch):
    detector = _make_detector(monkeypatch, [])
    results = _flush(detector, [_region("text")])
    assert isinstance(results[0], RegionWithLines)


# ---------------------------------------------------------------------------
# Singleton behaviour
# ---------------------------------------------------------------------------


def test_init_creates_singleton(monkeypatch):
    monkeypatch.setattr("src.text_detection.TextDetection", lambda **kw: MagicMock())
    assert td_mod._global_detector is None
    td_mod.init()
    assert td_mod._global_detector is not None


def test_process_lazy_creates_singleton(monkeypatch):
    monkeypatch.setattr("src.text_detection.TextDetection", lambda **kw: MagicMock())
    assert td_mod._global_detector is None
    td_mod.process([])
    assert td_mod._global_detector is not None


def test_process_warns_on_lazy_init(monkeypatch, caplog):
    monkeypatch.setattr("src.text_detection.TextDetection", lambda **kw: MagicMock())
    import logging
    with caplog.at_level(logging.WARNING, logger="src.text_detection"):
        td_mod.process([])
    assert any("before init()" in r.message for r in caplog.records)


def test_init_replaces_existing_singleton(monkeypatch):
    monkeypatch.setattr("src.text_detection.TextDetection", lambda **kw: MagicMock())
    td_mod.init()
    first = td_mod._global_detector
    td_mod.init()
    assert td_mod._global_detector is not first
