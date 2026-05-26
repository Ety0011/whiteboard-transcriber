"""Stage 10 — Ledger Synthesis.

Append-only in-memory record of every Note seen during a session.
Mutations (update, mark_erased) automatically write live.md and
lecture_history.md to the configured output directory.

Nothing is ever deleted — erasure is recorded, not enacted.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from tracker import Note

log = logging.getLogger(__name__)

_TIMELAPSE_W, _TIMELAPSE_H = 1280, 720

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class TextVersion:
    """One OCR result snapshot for a Note.

    Attributes:
        text: The full OCR/LaTeX string returned by the VLM.
        timestamp: time.monotonic() at the moment the result was received.
    """

    text: str
    timestamp: float


@dataclass
class LedgerEntry:
    """Append-only record of a single Note across its full lifetime.

    Attributes:
        note_id: Stable integer ID from the NoteTracker.
        bbox: Last known bbox, shape (4,) int32: x1, y1, x2, y2 in rectified space.
        first_seen: time.monotonic() when the note was first submitted to the VLM.
        versions: Ordered list of OCR results; index 0 is the first, each subsequent
            entry is a VERSIONED correction.
        erased_at: time.monotonic() when the note left the board, or None if active.
    """

    note_id: int
    bbox: np.ndarray
    first_seen: float
    versions: list[TextVersion] = field(default_factory=list)
    erased_at: float | None = None


# ---------------------------------------------------------------------------
# Atomic write helper
# ---------------------------------------------------------------------------


def _write_atomic(path: Path, content: str) -> None:
    """Write *content* to *path* atomically via a temp-file rename.

    Prevents a markdown reader from observing a partial write of live.md
    mid-synthesis — the file is either the old version or the new version,
    never a partially written intermediate state.
    """
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
        self._snapshots: list[tuple[float, np.ndarray]] = []

    # ------------------------------------------------------------------
    # Mutations — each auto-synthesizes output files
    # ------------------------------------------------------------------

    def update(self, note_id: int, bbox: np.ndarray, text: str) -> None:
        """Record a new or updated OCR result for *note_id*.

        - New note → creates entry with first version.
        - Existing note, different text → appends VERSIONED event.
        - Existing note, identical text → no-op (no file write).
        """
        now = time.monotonic()
        if note_id not in self._entries:
            self._entries[note_id] = LedgerEntry(
                note_id=note_id,
                bbox=bbox.copy(),
                first_seen=now,
                versions=[TextVersion(text=text, timestamp=now)],
            )
        else:
            entry = self._entries[note_id]
            if entry.versions[-1].text == text:
                return  # identical — skip write
            entry.versions.append(TextVersion(text=text, timestamp=now))

        self._synthesize()

    def mark_erased(self, note_id: int) -> None:
        """Record that *note_id* is no longer visible on the board."""
        entry = self._entries.get(note_id)
        if entry is not None and entry.erased_at is None:
            entry.erased_at = time.monotonic()
            self._synthesize()

    def sync(
        self,
        erased: list[Note],
        newly_active: list[Note],
        composite: np.ndarray,
    ) -> None:
        """Apply one pipeline cycle's erasures, OCR activations, and timelapse snapshot.

        Args:
            erased:       Notes that left the board this frame.
            newly_active: Notes that just completed OCR transcription.
            composite:    Full-resolution BGR board composite from Stage 5.
        """
        for note in erased:
            self.mark_erased(note.id)
        for note in newly_active:
            self.update(note.id, note.bbox, note.ocr_text or "")
        if newly_active:
            self.record_snapshot(composite, time.monotonic())

    # ------------------------------------------------------------------
    # Timelapse
    # ------------------------------------------------------------------

    def record_snapshot(self, composite: np.ndarray, timestamp: float) -> None:
        """Store a downscaled board snapshot at a stabilization event.

        Args:
            composite: Full-resolution BGR board composite from Stage 5.
            timestamp: time.monotonic() at the moment of capture.
        """
        thumb = cv2.resize(
            composite,
            (_TIMELAPSE_W, _TIMELAPSE_H),
            interpolation=cv2.INTER_AREA,
        )
        self._snapshots.append((timestamp, thumb))

    def synthesize_timelapse(
        self,
        fps: int = 24,
        seconds_per_frame: float = 1.0,
    ) -> Path | None:
        """Write a timelapse MP4 to the output directory.

        Each snapshot is held for *seconds_per_frame* seconds at *fps* frames
        per second. Returns None if no snapshots were recorded.

        Args:
            fps: Output video frame rate.
            seconds_per_frame: How long each snapshot is visible in the output.

        Returns:
            Path to the written file, or None if there were no snapshots.
        """
        if not self._snapshots:
            return None

        out_path = self._output_dir / "timelapse.mp4"
        frames_per_snapshot = max(1, round(fps * seconds_per_frame))
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(
            str(out_path), fourcc, fps, (_TIMELAPSE_W, _TIMELAPSE_H)
        )
        try:
            for _, thumb in self._snapshots:
                for _ in range(frames_per_snapshot):
                    writer.write(thumb)
        finally:
            writer.release()

        log.info(
            "Timelapse written: %s (%d frames, %d events)",
            out_path,
            len(self._snapshots) * frames_per_snapshot,
            len(self._snapshots),
        )
        return out_path

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
        """Re-render and atomically overwrite both output files."""
        _write_atomic(self._output_dir / "live.md", self._render_live())
        _write_atomic(self._output_dir / "lecture_history.md", self._render_history())

    # ------------------------------------------------------------------
    # Private — renderers
    # ------------------------------------------------------------------

    def _render_live(self) -> str:
        """Render live.md: current board content, one block per active note."""
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
        """Render lecture_history.md: full chronological ledger with TOC."""
        all_entries = self.get_all()
        if not all_entries:
            return "# Lecture Notes\n\n*(no content yet)*\n"

        toc_lines = ["## Contents\n"]
        for entry in all_entries:
            ts = self._mono_to_wall_str(entry.first_seen)
            anchor = f"note{entry.note_id}-{ts.replace(':', '')}"
            content_lines = [
                line for line in entry.versions[-1].text.splitlines() if line.strip()
            ]
            preview = content_lines[0] if content_lines else ""
            if len(content_lines) > 1 or len(preview) > 60:
                preview = preview[:60].rstrip() + "…"
            toc_lines.append(f"- [{ts}](#{anchor}): {preview}")
        toc = "\n".join(toc_lines)

        sections: list[str] = ["# Lecture Notes\n", toc, "---"]
        for entry in all_entries:
            sections.append(self._render_entry(entry))

        return "\n\n".join(sections)

    def _render_entry(self, entry: LedgerEntry) -> str:
        """Render one history section: heading, latest text, collapsible revisions."""
        ts = self._mono_to_wall_str(entry.first_seen)
        anchor = f"note{entry.note_id}-{ts.replace(':', '')}"
        parts = [f"## {ts} {{#{anchor}}}", "", entry.versions[-1].text]

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
        """Convert a time.monotonic() timestamp to a wall-clock HH:MM string."""
        wall = self._session_start_wall + (mono - self._session_start_mono)
        return time.strftime("%H:%M", time.localtime(wall))
