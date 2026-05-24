"""Entity Registry — cross-frame persistence and lifecycle management.

Maintains a persistent registry of SemanticEntity objects across frames. Each
frame, grouped anchors are matched to existing entities using IoU + centroid
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


class EntityState(enum.Enum):
    """Lifecycle states for a tracked semantic entity."""

    STABILIZING = "STABILIZING"  # ink writing/editing in progress or settling
    INFERRING = "INFERRING"  # crop submitted to GOT-OCR 2.0, awaiting result
    ACTIVE = "ACTIVE"  # OCR complete; entity visible on board
    ERASED = "ERASED"  # anchors gone from clean board; entity archived


@dataclasses.dataclass
class SemanticEntity:
    """A persistent entity tracked across frames.

    Bounding box is kept EMA-smoothed to reduce jitter. All timestamps are
    from time.monotonic().
    """

    id: int
    bbox: np.ndarray  # shape (4,) int32: x1, y1, x2, y2 — EMA-smoothed
    confidence: float
    state: EntityState
    first_seen: float
    last_modified: float
    last_seen: float
    ocr_text: str | None
    last_stable_center: np.ndarray | None = None  # shape (2,) float64 cx,cy


@dataclasses.dataclass
class EntityUpdate:
    """Output of one Registry processing cycle."""

    entities: list[SemanticEntity]  # all non-ERASED entities
    newly_inferring: list[SemanticEntity]  # transitioned to INFERRING this frame
    newly_erased: list[SemanticEntity]  # transitioned to ERASED this frame


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
    """Combined detection-to-entity match score: 0.7*IoU + 0.3*centroid_similarity."""
    return 0.7 * _iou(det_bbox, reg_bbox) + 0.3 * _centroid_similarity(
        det_bbox, reg_bbox, frame_diagonal
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class Registry:
    """Persistent entity registry for the whiteboard pipeline.

    Matches blocks from Stage 7 to existing entities, applies EMA bbox
    smoothing, advances the state machine, and exposes newly inferring or
    erased entities each frame.

    Args:
        stable_time_threshold: Seconds without significant change required
            before STABILIZING → INFERRING (VLM dispatch).
        tombstone_retention: Seconds to retain ERASED entries before deletion.
        match_threshold: Minimum combined score to match a block to an entity.
        drift_threshold_px: Centroid drift (px) on ACTIVE/INFERRING entity
            that triggers reset to STABILIZING.
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

        self._registry: dict[int, SemanticEntity] = {}
        self._next_id: int = 0

    def mark_active(self, entity: SemanticEntity, text: str) -> None:
        """Record VLM result and transition INFERRING → ACTIVE."""
        entity.ocr_text = text
        entity.state = EntityState.ACTIVE
        entity.last_modified = time.monotonic()
        log.debug("Entity %d → ACTIVE: %r", entity.id, text[:30])

    def reset_to_stabilizing(self, entity: SemanticEntity) -> None:
        """Reset INFERRING entity back to STABILIZING (e.g. degenerate crop)."""
        entity.state = EntityState.STABILIZING
        entity.last_modified = time.monotonic()
        log.debug("Entity %d reset → STABILIZING (empty crop)", entity.id)

    def tick(
        self,
        blocks: list[Block],
        frame_shape: tuple[int, int],
    ) -> EntityUpdate:
        """Run one lifecycle cycle: match blocks, advance state machine.

        Args:
            blocks:      Layout blocks from Stage 7 (LayoutWorker).
            frame_shape: (height, width) of the rectified board composite.

        Returns:
            EntityUpdate with all non-ERASED entities and transition lists.
        """
        now = time.monotonic()
        h, w = frame_shape
        frame_diagonal = math.sqrt(h * h + w * w)

        active_entities = [
            e for e in self._registry.values() if e.state != EntityState.ERASED
        ]

        assignments, matched_block_ids, matched_entity_ids = self._get_assignments(
            blocks, active_entities, frame_diagonal
        )

        newly_inferring: list[SemanticEntity] = []
        for blk_id, ent_id in assignments.items():
            self._update_entity(
                blocks[blk_id], self._registry[ent_id], now, newly_inferring
            )

        newly_erased: list[SemanticEntity] = []
        self._erase_unmatched(active_entities, matched_entity_ids, now, newly_erased)
        self._create_new_entities(blocks, matched_block_ids, now)
        self._prune_tombstones(now)

        return EntityUpdate(
            entities=[
                e for e in self._registry.values() if e.state != EntityState.ERASED
            ],
            newly_inferring=newly_inferring,
            newly_erased=newly_erased,
        )

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _get_assignments(self, blocks, active_entities, frame_diagonal):
        """Match detected blocks to existing entities using a greedy one-to-one assignment.

        Scores all (block, entity) pairs above match_threshold, sorts by score
        descending, then greedily assigns the highest-scoring pair first,
        consuming each block and entity at most once.

        Args:
            blocks: Layout blocks from the current frame.
            active_entities: All non-ERASED entities in the registry.
            frame_diagonal: Normalisation constant for centroid similarity.

        Returns:
            Tuple of (assignments, matched_block_ids, matched_entity_ids) where
            assignments maps block index → entity id.
        """
        candidates = []
        for blk_id, block in enumerate(blocks):
            for ent in active_entities:
                score = _match_score(block.bbox, ent.bbox, frame_diagonal)
                if score > self._match_threshold:
                    candidates.append((score, blk_id, ent.id))

        # Highest score first — greedy assignment gives each block its best entity.
        candidates.sort(key=lambda x: -x[0])

        matched_block_ids: set[int] = set()
        matched_entity_ids: set[int] = set()
        assignments: dict[int, int] = {}

        for _, blk_id, ent_id in candidates:
            if blk_id not in matched_block_ids and ent_id not in matched_entity_ids:
                assignments[blk_id] = ent_id
                matched_block_ids.add(blk_id)
                matched_entity_ids.add(ent_id)
        return assignments, matched_block_ids, matched_entity_ids

    def _update_entity(
        self,
        block: Block,
        ent: SemanticEntity,
        now: float,
        newly_inferring: list[SemanticEntity],
    ) -> None:
        """Advance state machine for a single matched entity."""

        # Detect movement — resets stabilization timer for all non-ERASED states.
        # Covers: professor still writing (STABILIZING), and post-commit edits
        # (INFERRING/ACTIVE). Without this, a block moving for 9s dispatches at s10.
        if ent.last_stable_center is not None:
            cur_center = (block.bbox[:2] + block.bbox[2:]) / 2.0
            drift = float(np.linalg.norm(cur_center - ent.last_stable_center))
            if drift > self._drift_threshold_px:
                ent.state = EntityState.STABILIZING
                ent.ocr_text = None
                ent.last_modified = now
                ent.last_stable_center = cur_center  # anchor new baseline

        # Physical update — EMA bbox smoothing
        ent.bbox = (0.2 * block.bbox + 0.8 * ent.bbox).astype(np.int32)
        ent.confidence, ent.last_seen = block.confidence, now

        if ent.state == EntityState.STABILIZING:
            if now - ent.last_modified >= self._stable_time_threshold:
                self._dispatch_for_inference(ent, now, newly_inferring)

    def _dispatch_for_inference(
        self,
        ent: SemanticEntity,
        now: float,
        newly_inferring: list[SemanticEntity],
    ) -> None:
        """Transition STABILIZING → INFERRING and anchor the stable center."""
        ent.last_stable_center = (ent.bbox[:2] + ent.bbox[2:]) / 2.0
        ent.state, ent.last_modified = EntityState.INFERRING, now
        newly_inferring.append(ent)
        log.debug("Entity %d → INFERRING", ent.id)

    def _erase_unmatched(
        self,
        active_entities: list[SemanticEntity],
        matched_ent_ids: set[int],
        now: float,
        newly_erased: list[SemanticEntity],
    ) -> None:
        """Erase entities absent for longer than erase_grace_period seconds."""
        for ent in active_entities:
            if ent.id not in matched_ent_ids:
                if now - ent.last_seen >= self._erase_grace_period:
                    ent.state, ent.last_modified = EntityState.ERASED, now
                    newly_erased.append(ent)
                    log.debug("Entity %d → ERASED", ent.id)

    def _create_new_entities(
        self, blocks: list[Block], matched_indices: set[int], now: float
    ) -> None:
        """Create STABILIZING entities for blocks that had no matching entity."""
        for blk_id, block in enumerate(blocks):
            if blk_id not in matched_indices:
                new_id = self._next_id
                self._next_id += 1
                cx = (block.bbox[0] + block.bbox[2]) / 2.0
                cy = (block.bbox[1] + block.bbox[3]) / 2.0
                self._registry[new_id] = SemanticEntity(
                    id=new_id,
                    bbox=block.bbox.copy(),
                    confidence=block.confidence,
                    state=EntityState.STABILIZING,
                    first_seen=now,
                    last_seen=now,
                    last_modified=now,
                    ocr_text=None,
                    last_stable_center=np.array([cx, cy], dtype=np.float64),
                )

    def _prune_tombstones(self, now):
        """Remove ERASED entities that have exceeded the tombstone retention window."""
        to_remove = [
            ent_id
            for ent_id, ent in self._registry.items()
            if ent.state == EntityState.ERASED
            and now - ent.last_modified > self._tombstone_retention
        ]
        for ent_id in to_remove:
            del self._registry[ent_id]
