"""Tests for src/document.py — WhiteboardDoc."""

from __future__ import annotations

from src.document import WhiteboardDoc


class TestToMarkdown:
    def test_empty_doc_returns_empty_string(self):
        assert WhiteboardDoc().to_markdown() == ""

    def test_single_block(self):
        doc = WhiteboardDoc(blocks={1: "hello world"})
        assert doc.to_markdown() == "hello world"

    def test_insertion_order_preserved(self):
        doc = WhiteboardDoc(blocks={1: "first", 2: "second", 3: "third"})
        assert doc.to_markdown() == "first\n\nsecond\n\nthird"

    def test_restabilization_updates_in_place(self):
        doc = WhiteboardDoc(blocks={1: "like", 2: "other"})
        doc.blocks[1] = "like this"
        assert doc.to_markdown() == "like this\n\nother"

    def test_to_markdown_idempotent(self):
        doc = WhiteboardDoc(blocks={1: "alpha", 2: "beta"})
        assert doc.to_markdown() == doc.to_markdown()
