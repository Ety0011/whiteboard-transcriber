"""Persistent whiteboard document model.

WhiteboardDoc is the pipeline's output artifact — a Markdown document
that accumulates OCR results across the whole session. It is written to
disk by pipeline.py and updated in-place by the Recognizer each frame.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class WhiteboardDoc:
    """Persistent Markdown document for the whiteboard session.

    blocks maps region_id to the current Markdown text for that region.
    Erased regions are wrapped in Markdown strikethrough to preserve history.
    """

    blocks: dict[int, str] = dataclasses.field(default_factory=dict)
