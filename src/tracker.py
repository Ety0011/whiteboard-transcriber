"""NoteTracker — cross-frame persistence and lifecycle management.

Maintains a persistent registry of Note objects across frames. Each
frame, layout blocks are matched to existing notes using IoU + centroid
scoring, bounding boxes are EMA-smoothed, and the state machine is advanced.

Lifecycle: STABILIZING → INFERRING → ACTIVE
                 ↑______________|  (edit detected)
                                   → ERASED  (anchors gone)
"""

from __future__ import annotations

import dataclasses
import enum
import logging
import math
import time

import numpy as np

from layout import Block

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TranscriptionResult:
    """OCR result produced by the transcription worker subprocess.

    Attributes:
        note_id: NoteTracker ID of the note whose crop was transcribed.
        text: Recognised text (and/or LaTeX) returned by the VLM backend.
    """

    note_id: int
    text: str


class NoteState(enum.Enum):
    """Lifecycle states for a board note."""

    STABILIZING = "STABILIZING"  # ink writing/editing in progress or settling
    INFERRING = "INFERRING"  # crop submitted to VLM, awaiting result
    ACTIVE = "ACTIVE"  # OCR complete; note visible on board
    ERASED = "ERASED"  # anchors gone from clean board; note archived


@dataclasses.dataclass
class Note:
    """A piece of writing on the board, tracked persistently across frames.

    Bounding box is kept EMA-smoothed to reduce jitter. All timestamps are
    from time.monotonic().
    """

    id: int
    bbox: np.ndarray  # shape (4,) int32: x1, y1, x2, y2 — EMA-smoothed
    confidence: float
    state: NoteState
    first_seen: float
    last_modified: float
    last_seen: float
    ocr_text: str = ""
    last_stable_center: np.ndarray | None = None  # shape (2,) float64 cx,cy
    crop: np.ndarray | None = None  # board region snapshot taken at INFERRING dispatch




# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Intersection-over-Union of two (x1,y1,x2,y2) bounding boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _centroid_similarity(
    a: np.ndarray,
    b: np.ndarray,
    frame_diagonal: float,
) -> float:
    """Centroid proximity normalized to [0, 1] via frame diagonal."""
    cx_a = (a[0] + a[2]) / 2.0
    cy_a = (a[1] + a[3]) / 2.0
    cx_b = (b[0] + b[2]) / 2.0
    cy_b = (b[1] + b[3]) / 2.0
    dist = math.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2)
    return max(0.0, 1.0 - dist / frame_diagonal) if frame_diagonal > 0 else 0.0


def _match_score(
    det_bbox: np.ndarray,
    reg_bbox: np.ndarray,
    frame_diagonal: float,
) -> float:
    """Combined detection-to-note match score: 0.7*IoU + 0.3*centroid_similarity."""
    return 0.7 * _iou(det_bbox, reg_bbox) + 0.3 * _centroid_similarity(
        det_bbox, reg_bbox, frame_diagonal
    )


# ---------------------------------------------------------------------------
# NoteTracker
# ---------------------------------------------------------------------------


class NoteTracker:
    """Tracks board notes across frames.

    Matches blocks from Stage 7 to existing notes, applies EMA bbox
    smoothing, advances the state machine, and exposes newly inferring or
    erased notes each frame.

    Args:
        stable_time_threshold: Seconds without significant change required
            before STABILIZING → INFERRING (VLM dispatch).
        tombstone_retention: Seconds to retain ERASED notes before deletion.
        match_threshold: Minimum combined score to match a block to a note.
        drift_threshold_px: Centroid drift (px) on ACTIVE/INFERRING note
            that triggers reset to STABILIZING.
        erase_grace_period: Seconds a note must be absent before erasure.
    """

    def __init__(
        self,
        stable_time_threshold: float = 10.0,
        tombstone_retention: float = 3.0,
        match_threshold: float = 0.5,
        drift_threshold_px: float = 50.0,
        erase_grace_period: float = 1.0,
    ) -> None:
        self._stable_time_threshold = stable_time_threshold
        self._tombstone_retention = tombstone_retention
        self._match_threshold = match_threshold
        self._drift_threshold_px = drift_threshold_px
        self._erase_grace_period = erase_grace_period

        self._notes: dict[int, Note] = {}
        self._next_id: int = 0
        self._pending_ocr: dict[int, Note] = {}

    @property
    def notes(self) -> list[Note]:
        """All currently visible (non-ERASED) notes."""
        return [n for n in self._notes.values() if n.state != NoteState.ERASED]

    @property
    def all_notes(self) -> list[Note]:
        """All notes including ERASED tombstones (retained for tombstone_retention seconds)."""
        return list(self._notes.values())

    def _commit_ocr_result(self, note_id: int, text: str) -> Note | None:
        """Transition INFERRING → ACTIVE with the completed OCR text.

        Args:
            note_id: ID of the note whose OCR completed.
            text:    Recognised text from the VLM.

        Returns:
            The activated note, or None if it drifted before the result arrived.
        """
        note = self._pending_ocr.pop(note_id, None)
        if note is None or note.state != NoteState.INFERRING:
            return None
        note.ocr_text = text
        note.state = NoteState.ACTIVE
        note.last_modified = time.monotonic()
        log.debug("Note %d → ACTIVE: %r", note.id, text[:30])
        return note

    def update(
        self,
        blocks: list[Block],
        composite: np.ndarray,
        transcriptions: list[TranscriptionResult],
    ) -> tuple[list[Note], list[Note], list[Note]]:
        """Match blocks to notes and advance the state machine.

        Args:
            blocks:         Layout blocks from Stage 7 (LayoutWorker).
            composite:      Clean board composite from Stage 5; used to snapshot
                            note crops at INFERRING dispatch.
            transcriptions: Completed OCR results from the transcription worker.

        Returns:
            (newly_inferring, newly_erased, newly_active) — notes that transitioned
            this frame. Spatial matching runs before OCR activation so drift-reset
            notes correctly reject stale results.
        """
        now = time.monotonic()
        h, w = composite.shape[:2]
        frame_diagonal = math.sqrt(h * h + w * w)

        active_notes = [n for n in self._notes.values() if n.state != NoteState.ERASED]

        assignments, matched_block_ids, matched_note_ids = self._get_assignments(
            blocks, active_notes, frame_diagonal
        )

        newly_inferring: list[Note] = []
        for blk_id, note_id in assignments.items():
            self._update_note(
                blocks[blk_id], self._notes[note_id], now, newly_inferring, composite
            )

        newly_erased: list[Note] = []
        self._erase_unmatched(active_notes, matched_note_ids, now, newly_erased)
        self._create_new_notes(blocks, matched_block_ids, now)
        self._prune_tombstones(now)

        for note in newly_erased:
            self._pending_ocr.pop(note.id, None)
        for note in newly_inferring:
            self._pending_ocr[note.id] = note

        newly_active = [
            n for r in transcriptions
            if (n := self._commit_ocr_result(r.note_id, r.text)) is not None
        ]
        return newly_inferring, newly_erased, newly_active

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _get_assignments(
        self,
        blocks: list[Block],
        active_notes: list[Note],
        frame_diagonal: float,
    ) -> tuple[dict[int, int], set[int], set[int]]:
        """Match detected blocks to existing notes using greedy one-to-one assignment.

        Scores all (block, note) pairs above match_threshold, sorts by score
        descending, then greedily assigns the highest-scoring pair first,
        consuming each block and note at most once.

        Returns:
            Tuple of (assignments, matched_block_ids, matched_note_ids) where
            assignments maps block index → note id.
        """
        candidates = []
        for blk_id, block in enumerate(blocks):
            for note in active_notes:
                score = _match_score(block.bbox, note.bbox, frame_diagonal)
                if score > self._match_threshold:
                    candidates.append((score, blk_id, note.id))

        # Highest score first — greedy assignment gives each block its best note.
        candidates.sort(key=lambda x: -x[0])

        matched_block_ids: set[int] = set()
        matched_note_ids: set[int] = set()
        assignments: dict[int, int] = {}

        for _, blk_id, note_id in candidates:
            if blk_id not in matched_block_ids and note_id not in matched_note_ids:
                assignments[blk_id] = note_id
                matched_block_ids.add(blk_id)
                matched_note_ids.add(note_id)
        return assignments, matched_block_ids, matched_note_ids

    def _update_note(
        self,
        block: Block,
        note: Note,
        now: float,
        newly_inferring: list[Note],
        composite: np.ndarray,
    ) -> None:
        """Advance state machine for a single matched note."""

        # Detect movement — resets stabilization timer for all non-ERASED states.
        # Covers: professor still writing (STABILIZING), and post-commit edits
        # (INFERRING/ACTIVE). Without this, a block moving for 9s dispatches at s10.
        if note.last_stable_center is not None:
            cur_center = (block.bbox[:2] + block.bbox[2:]) / 2.0
            drift = float(np.linalg.norm(cur_center - note.last_stable_center))
            if drift > self._drift_threshold_px:
                note.state = NoteState.STABILIZING
                note.ocr_text = ""
                note.crop = None
                note.last_modified = now
                note.last_stable_center = cur_center  # anchor new baseline

        # Physical update — EMA bbox smoothing
        note.bbox = (0.2 * block.bbox + 0.8 * note.bbox).astype(np.int32)
        note.confidence, note.last_seen = block.confidence, now

        if note.state == NoteState.STABILIZING:
            if now - note.last_modified >= self._stable_time_threshold:
                self._dispatch_for_inference(note, now, newly_inferring, composite)

    def _dispatch_for_inference(
        self,
        note: Note,
        now: float,
        newly_inferring: list[Note],
        composite: np.ndarray,
    ) -> None:
        """Transition STABILIZING → INFERRING, snapshot crop, anchor stable center."""
        note.last_stable_center = (note.bbox[:2] + note.bbox[2:]) / 2.0
        x1, y1, x2, y2 = note.bbox
        note.crop = composite[y1:y2, x1:x2].copy()
        note.state, note.last_modified = NoteState.INFERRING, now
        newly_inferring.append(note)
        log.debug("Note %d → INFERRING", note.id)

    def _erase_unmatched(
        self,
        active_notes: list[Note],
        matched_note_ids: set[int],
        now: float,
        newly_erased: list[Note],
    ) -> None:
        """Erase notes absent for longer than erase_grace_period seconds."""
        for note in active_notes:
            if note.id not in matched_note_ids:
                if now - note.last_seen >= self._erase_grace_period:
                    note.state, note.last_modified = NoteState.ERASED, now
                    newly_erased.append(note)
                    log.debug("Note %d → ERASED", note.id)

    def _create_new_notes(
        self, blocks: list[Block], matched_indices: set[int], now: float
    ) -> None:
        """Create STABILIZING notes for blocks that had no matching note."""
        for blk_id, block in enumerate(blocks):
            if blk_id not in matched_indices:
                new_id = self._next_id
                self._next_id += 1
                cx = (block.bbox[0] + block.bbox[2]) / 2.0
                cy = (block.bbox[1] + block.bbox[3]) / 2.0
                self._notes[new_id] = Note(
                    id=new_id,
                    bbox=block.bbox.copy(),
                    confidence=block.confidence,
                    state=NoteState.STABILIZING,
                    first_seen=now,
                    last_seen=now,
                    last_modified=now,
                    ocr_text="",
                    last_stable_center=np.array([cx, cy], dtype=np.float64),
                )

    def _prune_tombstones(self, now: float) -> None:
        """Remove ERASED notes that have exceeded the tombstone retention window."""
        to_remove = [
            note_id
            for note_id, note in self._notes.items()
            if note.state == NoteState.ERASED
            and now - note.last_modified > self._tombstone_retention
        ]
        for note_id in to_remove:
            del self._notes[note_id]
