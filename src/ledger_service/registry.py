"""Stage 8 — Ledger Registry.

Append-only in-memory record of every Semantic Entity seen during a session.
Entries are keyed by entity_id and track the full text version history plus
erasure timestamp. Nothing is ever deleted — erasure is recorded, not enacted.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class TextVersion:
    text: str
    timestamp: float  # time.monotonic() value; rendered as HH:MM by assembly


@dataclass
class LedgerEntry:
    entity_id: int
    bbox: np.ndarray              # (4,) int32: x1,y1,x2,y2 in rectified space
    first_seen: float             # time.monotonic()
    versions: list[TextVersion]   # index 0 = first OCR; append-only
    erased_at: float | None = None


class LedgerRegistry:
    """Append-only session ledger.

    All mutations are monotone: text versions accumulate, erased_at is only
    ever set once (and never cleared). There is no delete operation.
    """

    def __init__(self) -> None:
        self._entries: dict[int, LedgerEntry] = {}
        self._session_start_mono: float = time.monotonic()
        self._session_start_wall: float = time.time()

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def update(self, entity_id: int, bbox: np.ndarray, text: str) -> None:
        """Record a new or updated OCR result for *entity_id*.

        - New entity → creates entry with first version.
        - Existing entity, different text → appends VERSIONED event.
        - Existing entity, identical text → no-op.
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
            if entry.versions[-1].text != text:
                entry.versions.append(TextVersion(text=text, timestamp=now))

    def mark_erased(self, entity_id: int) -> None:
        """Record that *entity_id* is no longer visible on the board."""
        if entity_id in self._entries and self._entries[entity_id].erased_at is None:
            self._entries[entity_id].erased_at = time.monotonic()

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
    # Timestamp helper
    # ------------------------------------------------------------------

    def mono_to_wall_str(self, mono: float) -> str:
        """Convert a time.monotonic() value to a HH:MM wall-clock string."""
        wall = self._session_start_wall + (mono - self._session_start_mono)
        import time as _time
        return _time.strftime("%H:%M", _time.localtime(wall))
