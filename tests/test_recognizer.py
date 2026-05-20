"""Tests for Stage 6 — src/recognition.py.

All tests use synthetic data. The TextRecognition model is replaced with a
lightweight mock so no model weights are required.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import numpy as np

from src.document import WhiteboardDoc
from src.text_recognizer import TextRecognizer
from src.anchor_service.entity_registry import SemanticEntity as Region, EntityState as RegionState, EntityUpdate as TrackerResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_region(
    region_id: int = 1,
    bbox: tuple[int, int, int, int] = (0, 0, 200, 40),
    ocr_text: str | None = None,
    line_bboxes: list[np.ndarray] | None = None,
) -> Region:
    """Build a minimal Region in STABLE state for testing."""
    crop = np.full((40, 200, 3), fill_value=240, dtype=np.uint8)
    bbox_arr = np.array(bbox, dtype=np.int32)
    return Region(
        id=region_id,
        bbox=bbox_arr,
        confidence=0.9,
        state=RegionState.READABLE,
        first_seen=time.monotonic(),
        last_modified=time.monotonic(),
        last_seen=time.monotonic(),
        ocr_text=ocr_text,
        ocr_confidence=None if ocr_text is None else 0.95,
        last_stable_crop=crop,
        line_bboxes=line_bboxes if line_bboxes is not None else [],
    )


def _make_tracker_result(
    newly_stable: list[Region] | None = None,
    newly_erased: list[Region] | None = None,
) -> TrackerResult:
    stable = newly_stable or []
    erased = newly_erased or []
    return TrackerResult(
        entities=stable + erased,
        newly_readable=stable,
        newly_erased=erased,
    )


def _make_recognizer(mock_predict_return: list[dict]) -> TextRecognizer:
    """Return a Recognizer whose internal TextRecognition is mocked."""
    recognizer = object.__new__(TextRecognizer)
    mock_engine = MagicMock()
    mock_engine.predict.return_value = mock_predict_return
    recognizer._recognizer = mock_engine
    return recognizer


def _make_mock_tracker() -> MagicMock:
    """Return a mock RegionTracker whose mark_ocr_done writes fields correctly."""
    mock = MagicMock()

    def _mark_ocr_done(region, text: str, confidence: float) -> None:
        region.ocr_text = text
        region.ocr_confidence = confidence
        region.last_modified = time.monotonic()

    mock.mark_ocr_done.side_effect = _mark_ocr_done
    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFirstStabilization:
    def test_block_created_with_ocr_text(self):
        region = _make_region(region_id=1, ocr_text=None)
        recognizer = _make_recognizer(
            [
                {"rec_text": "hello", "rec_score": 0.95},
                {"rec_text": "world", "rec_score": 0.90},
            ]
        )
        doc = WhiteboardDoc()
        result = recognizer.process(
            _make_tracker_result(newly_stable=[region]), _make_mock_tracker(), doc
        )

        assert result.blocks[1] == "hello\nworld"

    def test_region_ocr_text_mutated(self):
        region = _make_region(region_id=2, ocr_text=None)
        recognizer = _make_recognizer([{"rec_text": "foo", "rec_score": 0.88}])
        doc = WhiteboardDoc()
        recognizer.process(
            _make_tracker_result(newly_stable=[region]), _make_mock_tracker(), doc
        )

        assert region.ocr_text == "foo"
        assert region.ocr_confidence is not None

    def test_empty_ocr_result_stored(self):
        region = _make_region(region_id=3, ocr_text=None)
        recognizer = _make_recognizer([])
        doc = WhiteboardDoc()
        recognizer.process(
            _make_tracker_result(newly_stable=[region]), _make_mock_tracker(), doc
        )

        assert doc.blocks[3] == ""
        assert region.ocr_text == ""


class TestRestabilizationWithAdditions:
    def test_block_updated_in_place_on_change(self):
        region = _make_region(region_id=1, ocr_text="hello")
        recognizer = _make_recognizer(
            [
                {"rec_text": "hello", "rec_score": 0.95},
                {"rec_text": "world", "rec_score": 0.90},
            ]
        )
        doc = WhiteboardDoc(blocks={1: "hello"})
        recognizer.process(
            _make_tracker_result(newly_stable=[region]), _make_mock_tracker(), doc
        )

        assert len(doc.blocks) == 1  # same entry, updated in-place
        assert doc.blocks[1] == "hello\nworld"
        assert region.ocr_text == "hello\nworld"

    def test_region_last_modified_updated(self):
        before = time.monotonic()
        region = _make_region(region_id=1, ocr_text="old")
        recognizer = _make_recognizer([{"rec_text": "new", "rec_score": 0.9}])
        doc = WhiteboardDoc(blocks={1: "old"})
        recognizer.process(
            _make_tracker_result(newly_stable=[region]), _make_mock_tracker(), doc
        )

        assert region.last_modified >= before


class TestRestabilizationWithRemovals:
    def test_block_updated_in_place_with_removed_line(self):
        region = _make_region(region_id=1, ocr_text="hello\nworld")
        recognizer = _make_recognizer([{"rec_text": "hello", "rec_score": 0.95}])
        doc = WhiteboardDoc(blocks={1: "hello\nworld"})
        recognizer.process(
            _make_tracker_result(newly_stable=[region]), _make_mock_tracker(), doc
        )

        assert len(doc.blocks) == 1
        assert doc.blocks[1] == "hello"
        assert region.ocr_text == "hello"


class TestNoChangeSkip:
    def test_block_unchanged_when_content_identical(self):
        region = _make_region(region_id=1, ocr_text="hello")
        original_modified = region.last_modified
        recognizer = _make_recognizer([{"rec_text": "hello", "rec_score": 0.95}])
        doc = WhiteboardDoc(blocks={1: "hello"})
        recognizer.process(
            _make_tracker_result(newly_stable=[region]), _make_mock_tracker(), doc
        )

        assert doc.blocks[1] == "hello"
        assert region.last_modified == original_modified  # no mutation when unchanged


class TestErasedRegion:
    def test_erased_region_leaves_doc_unchanged(self):
        region = _make_region(region_id=1)
        recognizer = _make_recognizer([])
        doc = WhiteboardDoc(blocks={1: "some text"})
        recognizer.process(
            _make_tracker_result(newly_erased=[region]), _make_mock_tracker(), doc
        )

        assert doc.blocks == {1: "some text"}

    def test_erased_region_on_empty_doc_leaves_doc_unchanged(self):
        region = _make_region(region_id=99)
        recognizer = _make_recognizer([])
        doc = WhiteboardDoc()
        recognizer.process(
            _make_tracker_result(newly_erased=[region]), _make_mock_tracker(), doc
        )

        assert doc.blocks == {}


class TestLineCropExtraction:
    def test_full_crop_returned_when_no_line_bboxes(self):
        region = _make_region(region_id=1, line_bboxes=[])
        recognizer = _make_recognizer([])
        crops = recognizer._extract_line_crops(region)

        assert len(crops) == 1
        assert crops[0] is region.last_stable_crop

    def test_line_bboxes_sliced_from_crop(self):
        # Region at board position (10, 20, 210, 60); crop is 40×200
        bbox = (10, 20, 210, 60)
        line_bbox = np.array([10, 20, 210, 40], dtype=np.int32)  # full top half
        region = _make_region(region_id=1, bbox=bbox, line_bboxes=[line_bbox])
        recognizer = _make_recognizer([])
        crops = recognizer._extract_line_crops(region)

        assert len(crops) == 1
        assert crops[0].shape[0] == 20  # height = 40 - 20 = 20

    def test_none_crop_returns_empty(self):
        region = _make_region(region_id=1)
        region.last_stable_crop = None
        recognizer = _make_recognizer([])
        crops = recognizer._extract_line_crops(region)

        assert crops == []
