"""Persistent whiteboard document model.

WhiteboardDoc is the pipeline's output artifact — an append-only Markdown log
that accumulates OCR results across the whole session. It is written to disk by
pipeline.py and updated in-place by the Recognizer each frame.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class WhiteboardDoc:
    """Append-only Markdown log for the whiteboard session.

    Blocks are appended in recognition order. Erasing is treated as progression —
    the professor clears space to write the next thing — so erased content is kept
    and new content is appended after it.
    """

    blocks: list[str] = dataclasses.field(default_factory=list)

    def append(self, text: str) -> None:
        """Append a newly recognised text block to the document.

        Args:
            text: Markdown text produced by the Recognizer for one stable region.
        """
        self.blocks.append(text)

    def to_markdown(self) -> str:
        """Return the full Markdown document as a single string.

        Blocks are joined with a blank line between them, preserving insertion order.

        Returns:
            Markdown string, or an empty string when no blocks have been appended.
        """
        return "\n\n".join(self.blocks)
