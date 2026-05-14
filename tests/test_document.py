"""Tests for src/document.py — WhiteboardDoc append-only log."""

from __future__ import annotations

from src.document import WhiteboardDoc


class TestToMarkdown:
    def test_empty_doc_returns_empty_string(self):
        assert WhiteboardDoc().to_markdown() == ""

    def test_append_single_block(self):
        doc = WhiteboardDoc()
        doc.append("hello world")
        assert doc.to_markdown() == "hello world"

    def test_append_order_preserved(self):
        doc = WhiteboardDoc()
        doc.append("first")
        doc.append("second")
        doc.append("third")
        assert doc.to_markdown() == "first\n\nsecond\n\nthird"

    def test_to_markdown_idempotent(self):
        doc = WhiteboardDoc()
        doc.append("alpha")
        doc.append("beta")
        assert doc.to_markdown() == doc.to_markdown()
