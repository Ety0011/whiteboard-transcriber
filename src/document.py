"""Persistent whiteboard document model.

WhiteboardDoc is the pipeline's output artifact — a Markdown document keyed by
region ID. It is written to disk by pipeline.py and updated in-place by the
Recognizer each frame.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class WhiteboardDoc:
    """Persistent Markdown document for the whiteboard session.

    blocks maps region_id to the current Markdown text for that region.
    Dict insertion order equals recognition order (top-to-bottom as the
    professor writes). Re-stabilization updates the existing entry in-place,
    preserving its position. Erased regions are left as-is — their text
    remains part of the document.
    """

    blocks: dict[int, str] = dataclasses.field(default_factory=dict)

    def to_markdown(self) -> str:
        """Return the full Markdown document as a single string.

        Blocks are joined with a blank line between them, in insertion order.

        Returns:
            Markdown string, or an empty string when no blocks have been added.
        """
        return "\n\n".join(self.blocks.values())
