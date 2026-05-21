"""Stage 8 — Session Ledger.

Append-only in-memory record of every Semantic Entity seen during a session.
Mutations (update, mark_erased) automatically write live.md and
lecture_history.md to the configured output directory.

Nothing is ever deleted — erasure is recorded, not enacted.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class TextVersion:
    text: str
    timestamp: float  # time.monotonic(); rendered as HH:MM by _mono_to_wall_str


@dataclass
class LedgerEntry:
    entity_id: int
    bbox: np.ndarray  # (4,) int32: x1,y1,x2,y2 in rectified space
    first_seen: float  # time.monotonic()
    versions: list[TextVersion] = field(
        default_factory=list
    )  # index 0 = first OCR; append-only
    erased_at: float | None = None


# ---------------------------------------------------------------------------
# Atomic write helper
# ---------------------------------------------------------------------------


def _write_atomic(path: Path, content: str) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.rename(path)


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


class Ledger:
    """Append-only session ledger that auto-writes output files on every mutation.

    Args:
        output_dir: Directory where live.md and lecture_history.md are written.
                    Created automatically if it does not exist.
    """

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._entries: dict[int, LedgerEntry] = {}
        self._session_start_mono: float = time.monotonic()
        self._session_start_wall: float = time.time()

    # ------------------------------------------------------------------
    # Mutations — each auto-synthesizes output files
    # ------------------------------------------------------------------

    def update(self, entity_id: int, bbox: np.ndarray, text: str) -> None:
        """Record a new or updated OCR result for *entity_id*.

        - New entity → creates entry with first version.
        - Existing entity, different text → appends VERSIONED event.
        - Existing entity, identical text → no-op (no file write).
        """
        now = time.monotonic()
        if entity_id not in self._entries:
            self._entries[entity_id] = LedgerEntry(
                entity_id=entity_id,
                bbox=bbox.copy(),
                first_seen=now,
                versions=[TextVersion(text=text, timestamp=now)],
            )
        else:
            entry = self._entries[entity_id]
            if entry.versions[-1].text == text:
                return  # identical — skip write
            entry.versions.append(TextVersion(text=text, timestamp=now))

        self._synthesize()

    def mark_erased(self, entity_id: int) -> None:
        """Record that *entity_id* is no longer visible on the board."""
        entry = self._entries.get(entity_id)
        if entry is not None and entry.erased_at is None:
            entry.erased_at = time.monotonic()
            self._synthesize()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_active(self) -> list[LedgerEntry]:
        """All non-erased entries, sorted top-to-bottom by bbox y1."""
        active = [e for e in self._entries.values() if e.erased_at is None]
        return sorted(active, key=lambda e: int(e.bbox[1]))

    def get_all(self) -> list[LedgerEntry]:
        """All entries, sorted chronologically by first_seen."""
        return sorted(self._entries.values(), key=lambda e: e.first_seen)

    # ------------------------------------------------------------------
    # Private — synthesis
    # ------------------------------------------------------------------

    def _synthesize(self) -> None:
        _write_atomic(self._output_dir / "live.md", self._render_live())
        _write_atomic(self._output_dir / "lecture_history.md", self._render_history())

    # ------------------------------------------------------------------
    # Private — renderers
    # ------------------------------------------------------------------

    def _render_live(self) -> str:
        active = self.get_active()
        if not active:
            return "# Whiteboard\n\n*(board empty)*\n"

        blocks = ["# Whiteboard\n"]
        for entry in active:
            blocks.append(entry.versions[-1].text)
            blocks.append("---")
        # drop trailing separator
        if blocks[-1] == "---":
            blocks.pop()
        return "\n\n".join(blocks) + "\n"

    def _render_history(self) -> str:
        all_entries = self.get_all()
        if not all_entries:
            return "# Lecture Notes\n\n*(no content yet)*\n"

        toc_lines = ["## Contents\n"]
        for entry in all_entries:
            ts = self._mono_to_wall_str(entry.first_seen)
            anchor = ts.replace(":", "")
            content_lines = [l for l in entry.versions[-1].text.splitlines() if l.strip()]
            preview = content_lines[0] if content_lines else ""
            if len(content_lines) > 1 or len(preview) > 60:
                preview = preview[:60].rstrip() + "…"
            if len(preview) > 60:
                preview = preview[:60].rstrip() + "…"
            toc_lines.append(f"- [{ts}](#{anchor}): {preview}")
        toc = "\n".join(toc_lines)

        sections: list[str] = ["# Lecture Notes\n", toc, "---"]
        for entry in all_entries:
            sections.append(self._render_entry(entry))

        return "\n\n".join(sections)

    def _render_entry(self, entry: LedgerEntry) -> str:
        ts = self._mono_to_wall_str(entry.first_seen)
        parts = [f"## {ts}", "", entry.versions[-1].text]

        if len(entry.versions) > 1:
            n = len(entry.versions) - 1
            label = f"{n} revision" if n == 1 else f"{n} revisions"
            parts.append("")
            parts.append("<details>")
            parts.append(f"<summary>{label}</summary>")
            parts.append("")
            for i, v in enumerate(entry.versions[:-1], start=1):
                compact = "<br>".join(
                    line for line in v.text.splitlines() if line.strip()
                )
                parts.append(f"{i}. {compact}")
            parts.append("")
            parts.append("</details>")

        parts.append("")
        return "\n".join(parts)

    def _mono_to_wall_str(self, mono: float) -> str:
        import time as _time

        wall = self._session_start_wall + (mono - self._session_start_mono)
        return _time.strftime("%H:%M", _time.localtime(wall))
