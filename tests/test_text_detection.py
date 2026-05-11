"""Tests for Stage 5 — Text Line Detection (src/text_detection.py).

All tests mock TextDetection so no model weights are downloaded and no camera
is required. Synthetic crops are created inline.

TextDetector is synchronous (no child process), so monkeypatching TextDetection
affects _run_detection() and process() directly without additional fixtures.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

import src.text_detection as td_mod
from src.layout import Region
from src.text_detection import (
    RegionWithLines,
    TextDetector,
    TextLine,
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
    crop = _crop(h, w)
    return Region(bbox=(0, 0, w, h), label=label, confidence=0.9, crop=crop)


def _make_detector(monkeypatch, raw_result: list) -> TextDetector:
    """Return a TextDetector whose engine returns raw_result on predict()."""
    engine = MagicMock()
    engine.predict = MagicMock(return_value=raw_result)
    monkeypatch.setattr("src.text_detection.TextDetection", lambda **kw: engine)
    return TextDetector()


def _rect_poly(x1: int, y1: int, x2: int, y2: int) -> list[list[int]]:
    """Four-point polygon for a rectangle."""
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
    line_img = _crop(10, 40)
    line = TextLine(bbox=(5, 5, 45, 15), crop=line_img)
    r = RegionWithLines(
        bbox=(0, 0, 200, 100), label="text", confidence=0.9, crop=crop, lines=[line]
    )
    assert len(r.lines) == 1
    assert r.lines[0] is line


def test_regionwithlines_default_lines_is_empty_list():
    crop = _crop()
    r = RegionWithLines(bbox=(0, 0, 200, 100), label="text", confidence=0.9, crop=crop)
    assert r.lines == []
    # Confirm it's not a shared mutable default
    r2 = RegionWithLines(bbox=(0, 0, 200, 100), label="text", confidence=0.9, crop=crop)
    r.lines.append(TextLine(bbox=(0, 0, 10, 10), crop=_crop(10, 10)))
    assert r2.lines == []


# ---------------------------------------------------------------------------
# _polygon_to_bbox
# ---------------------------------------------------------------------------


def test_polygon_to_bbox_basic():
    poly = [[10, 20], [80, 20], [80, 60], [10, 60]]
    assert _polygon_to_bbox(poly, 100, 200) == (10, 20, 80, 60)


def test_polygon_to_bbox_clamps_to_image():
    poly = [[-5, -10], [250, -10], [250, 120], [-5, 120]]
    assert _polygon_to_bbox(poly, 100, 200) == (0, 0, 200, 100)


def test_polygon_to_bbox_partial_clamp():
    poly = [[190, 90], [210, 90], [210, 110], [190, 110]]
    x1, y1, x2, y2 = _polygon_to_bbox(poly, 100, 200)
    assert x1 == 190
    assert y1 == 90
    assert x2 == 200   # clamped from 210
    assert y2 == 100   # clamped from 110


def test_polygon_to_bbox_non_rectangular():
    # Skewed quadrilateral
    poly = [[10, 30], [90, 10], [110, 70], [30, 90]]
    x1, y1, x2, y2 = _polygon_to_bbox(poly, 200, 200)
    assert x1 == 10
    assert y1 == 10
    assert x2 == 110
    assert y2 == 90


# ---------------------------------------------------------------------------
# _parse_lines
# ---------------------------------------------------------------------------


def test_parse_lines_empty_result():
    assert _parse_lines([], _crop()) == []


def test_parse_lines_no_polys():
    assert _parse_lines([{"dt_polys": []}], _crop()) == []


def test_parse_lines_single_line():
    crop = _crop(100, 200)
    poly = _rect_poly(10, 20, 80, 50)
    lines = _parse_lines([{"dt_polys": [poly]}], crop)
    assert len(lines) == 1
    assert lines[0].bbox == (10, 20, 80, 50)


def test_parse_lines_crop_shape_matches_bbox():
    crop = _crop(100, 200)
    poly = _rect_poly(10, 20, 80, 50)
    line = _parse_lines([{"dt_polys": [poly]}], crop)[0]
    x1, y1, x2, y2 = line.bbox
    assert line.crop.shape == (y2 - y1, x2 - x1, 3)


def test_parse_lines_crop_is_copy():
    crop = _crop(100, 200)
    poly = _rect_poly(10, 20, 80, 50)
    line = _parse_lines([{"dt_polys": [poly]}], crop)[0]
    assert not np.shares_memory(line.crop, crop)


def test_parse_lines_degenerate_bbox_skipped():
    crop = _crop(100, 200)
    # Zero-height polygon
    degenerate = _rect_poly(10, 30, 80, 30)
    valid = _rect_poly(10, 50, 80, 80)
    lines = _parse_lines([{"dt_polys": [degenerate, valid]}], crop)
    assert len(lines) == 1
    assert lines[0].bbox == (10, 50, 80, 80)


def test_parse_lines_multiple_lines():
    crop = _crop(200, 400)
    polys = [_rect_poly(0, 0, 100, 30), _rect_poly(0, 50, 200, 80), _rect_poly(0, 100, 300, 130)]
    lines = _parse_lines([{"dt_polys": polys}], crop)
    assert len(lines) == 3


# ---------------------------------------------------------------------------
# _run_detection — mocked engine
# ---------------------------------------------------------------------------


def test_run_detection_returns_textlines(monkeypatch):
    poly = _rect_poly(5, 10, 60, 40)
    detector = _make_detector(monkeypatch, [{"dt_polys": [poly]}])
    crop = _crop()
    lines = detector._run_detection(crop)
    assert len(lines) == 1
    assert isinstance(lines[0], TextLine)
    assert lines[0].bbox == (5, 10, 60, 40)


def test_run_detection_empty_result(monkeypatch):
    detector = _make_detector(monkeypatch, [])
    assert detector._run_detection(_crop()) == []


def test_run_detection_crop_shape_matches_bbox(monkeypatch):
    poly = _rect_poly(10, 15, 90, 55)
    detector = _make_detector(monkeypatch, [{"dt_polys": [poly]}])
    lines = detector._run_detection(_crop())
    x1, y1, x2, y2 = lines[0].bbox
    assert lines[0].crop.shape == (y2 - y1, x2 - x1, 3)


def test_run_detection_crop_is_copy(monkeypatch):
    poly = _rect_poly(10, 15, 90, 55)
    detector = _make_detector(monkeypatch, [{"dt_polys": [poly]}])
    crop = _crop()
    lines = detector._run_detection(crop)
    assert not np.shares_memory(lines[0].crop, crop)


# ---------------------------------------------------------------------------
# process() — region-level behaviour
# ---------------------------------------------------------------------------


def test_process_text_region_gets_lines(monkeypatch):
    poly = _rect_poly(5, 10, 60, 40)
    detector = _make_detector(monkeypatch, [{"dt_polys": [poly]}])
    results = detector.process([_region("text")])
    assert len(results) == 1
    assert len(results[0].lines) == 1


def test_process_paragraph_title_gets_lines(monkeypatch):
    poly = _rect_poly(5, 10, 60, 40)
    detector = _make_detector(monkeypatch, [{"dt_polys": [poly]}])
    results = detector.process([_region("paragraph_title")])
    assert len(results[0].lines) == 1


def test_process_figure_region_skipped(monkeypatch):
    detector = _make_detector(monkeypatch, [{"dt_polys": [_rect_poly(0, 0, 50, 30)]}])
    results = detector.process([_region("figure")])
    assert results[0].lines == []
    # Engine should not have been called
    detector._engine.predict.assert_not_called()


def test_process_table_region_skipped(monkeypatch):
    detector = _make_detector(monkeypatch, [{"dt_polys": [_rect_poly(0, 0, 50, 30)]}])
    results = detector.process([_region("table")])
    assert results[0].lines == []
    detector._engine.predict.assert_not_called()


def test_process_mixed_regions(monkeypatch):
    poly = _rect_poly(5, 10, 60, 40)
    detector = _make_detector(monkeypatch, [{"dt_polys": [poly]}])
    regions = [_region("text"), _region("figure"), _region("text")]
    results = detector.process(regions)
    assert len(results) == 3
    assert len(results[0].lines) == 1   # text — detected
    assert results[1].lines == []       # figure — skipped
    assert len(results[2].lines) == 1   # text — detected


def test_process_preserves_region_fields(monkeypatch):
    detector = _make_detector(monkeypatch, [])
    source = _region("text", h=80, w=160)
    result = detector.process([source])[0]
    assert result.bbox == source.bbox
    assert result.label == source.label
    assert result.confidence == pytest.approx(source.confidence)
    assert result.crop is source.crop


def test_process_empty_regions_list(monkeypatch):
    detector = _make_detector(monkeypatch, [])
    assert detector.process([]) == []


def test_process_returns_regionwithlines_instances(monkeypatch):
    detector = _make_detector(monkeypatch, [])
    results = detector.process([_region("text")])
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
