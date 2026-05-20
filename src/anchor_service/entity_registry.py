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

from anchor_service.grouper import EntityGroup

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class EntityState(enum.Enum):
    """Lifecycle states for a tracked semantic entity."""

    STABILIZING = "STABILIZING"  # ink writing/editing in progress or settling
    INFERRING   = "INFERRING"    # crop submitted to GOT-OCR 2.0, awaiting result
    ACTIVE      = "ACTIVE"       # OCR complete; entity visible on board
    ERASED      = "ERASED"       # anchors gone from clean board; entity archived


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
    ocr_confidence: float | None
    last_stable_crop: np.ndarray | None  # BGR uint8 crop captured at inference dispatch
    last_stable_center: np.ndarray | None = None  # shape (2,) float64 cx,cy
    line_bboxes: list[np.ndarray] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class EntityUpdate:
    """Output of one EntityRegistry processing cycle."""

    entities: list[SemanticEntity]         # all non-ERASED entities
    newly_inferring: list[SemanticEntity]  # transitioned to INFERRING this frame
    newly_erased: list[SemanticEntity]     # transitioned to ERASED this frame


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
    frame_diag: float,
) -> float:
    """Centroid proximity normalized to [0, 1] via frame diagonal."""
    cx_a = (a[0] + a[2]) / 2.0
    cy_a = (a[1] + a[3]) / 2.0
    cx_b = (b[0] + b[2]) / 2.0
    cy_b = (b[1] + b[3]) / 2.0
    dist = math.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2)
    return max(0.0, 1.0 - dist / frame_diag) if frame_diag > 0 else 0.0


def _match_score(
    det_bbox: np.ndarray,
    reg_bbox: np.ndarray,
    frame_diag: float,
) -> float:
    # TODO: put coefficients as parameters
    """Combined detection-to-entity match score: 0.7*IoU + 0.3*centroid_similarity."""
    return 0.7 * _iou(det_bbox, reg_bbox) + 0.3 * _centroid_similarity(
        det_bbox, reg_bbox, frame_diag
    )


# ---------------------------------------------------------------------------
# EntityRegistry
# ---------------------------------------------------------------------------


# TODO: fix duplicate entities
class EntityRegistry:
    """Persistent entity registry for the whiteboard pipeline.

    Matches grouped anchors from Stage 6 to existing entities, applies EMA
    bbox smoothing, advances the state machine, and exposes newly inferring or
    erased entities each frame.

    Args:
        stable_time_threshold: Seconds without significant change required
            before STABILIZING → INFERRING (VLM dispatch).
        tombstone_retention: Seconds to retain ERASED entries before deletion.
        match_threshold: Minimum combined score to match a group to an entity.
        drift_threshold_px: Centroid drift (px) on ACTIVE/INFERRING entity
            that triggers reset to STABILIZING.
    """

    def __init__(
        self,
        stable_time_threshold: float = 5.0,
        tombstone_retention: float = 5.0,
        match_threshold: float = 0.4,
        drift_threshold_px: float = 20.0,
    ) -> None:
        self._stable_time_threshold = stable_time_threshold
        self._tombstone_retention = tombstone_retention
        self._match_threshold = match_threshold
        self._drift_threshold_px = drift_threshold_px

        self._registry: dict[int, SemanticEntity] = {}
        self._next_id: int = 0

    def mark_active(
        self,
        entity: SemanticEntity,
        text: str,
        confidence: float,
    ) -> None:
        """Record VLM result and transition INFERRING → ACTIVE."""
        entity.ocr_text = text
        entity.ocr_confidence = confidence
        entity.state = EntityState.ACTIVE
        entity.last_modified = time.monotonic()
        log.debug("Entity %d → ACTIVE: %r", entity.id, text[:30])

    def process(
        self,
        groups: list[EntityGroup],
        frame: np.ndarray,
    ) -> EntityUpdate:
        """Run one lifecycle cycle: match groups, advance state machine.

        Args:
            groups: Semantic entity groups from Stage 6 EntityGrouper.
            frame:  Current BGR board composite (Stage 4).

        Returns:
            EntityUpdate with all active entities and transition lists.
        """
        now = time.monotonic()
        h, w = frame.shape[:2]
        frame_diag = math.sqrt(h * h + w * w)

        active_entities = [
            e for e in self._registry.values() if e.state != EntityState.ERASED
        ]

        assignments, matched_grp, matched_ent = self._get_assignments(
            groups, active_entities, frame_diag
        )

        for grp_id, ent_id in assignments.items():
            self._update_entity(groups[grp_id], self._registry[ent_id], frame, now)

        self._erase_unmatched(active_entities, matched_ent, now)
        self._create_new_entities(groups, matched_grp, now)
        self._prune_tombstones(now)

        return EntityUpdate(
            entities=list(self._registry.values()),
            newly_inferring=[
                e
                for e in self._registry.values()
                if e.state == EntityState.INFERRING and e.last_modified == now
            ],
            newly_erased=[
                e
                for e in self._registry.values()
                if e.state == EntityState.ERASED and e.last_modified == now
            ],
        )

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    # TODO: if board moves too much we lose all assignments
    def _get_assignments(self, groups, active_entities, diag):
        candidates = []
        for grp_id, grp in enumerate(groups):
            for ent in active_entities:
                score = _match_score(grp.bbox, ent.bbox, diag)
                if score > self._match_threshold:
                    candidates.append((score, grp_id, ent.id))

        candidates.sort(key=lambda x: -x[0])

        matched_grp: set[int] = set()
        matched_ent: set[int] = set()
        assignments: dict[int, int] = {}

        for _, grp_id, ent_id in candidates:
            if grp_id not in matched_grp and ent_id not in matched_ent:
                assignments[grp_id] = ent_id
                matched_grp.add(grp_id)
                matched_ent.add(ent_id)
        return assignments, matched_grp, matched_ent

    def _update_entity(self, grp: EntityGroup, ent: SemanticEntity, frame, now):
        """Advance state machine for a single matched entity."""

        # Detect edit: significant centroid drift on a committed entity resets it.
        if ent.last_stable_center is not None and ent.state in (
            EntityState.INFERRING,
            EntityState.ACTIVE,
        ):
            cur_center = (grp.bbox[:2] + grp.bbox[2:]) / 2.0
            drift = float(np.linalg.norm(cur_center - ent.last_stable_center))
            if drift > self._drift_threshold_px:
                ent.state, ent.ocr_text = EntityState.STABILIZING, None
                ent.last_modified = now

        # Physical update — EMA bbox smoothing
        ent.bbox = (0.2 * grp.bbox + 0.8 * ent.bbox).astype(np.int32)
        ent.confidence, ent.last_seen = grp.confidence, now
        ent.line_bboxes = [a.bbox for a in grp.anchors]

        if ent.state == EntityState.STABILIZING:
            if now - ent.last_modified >= self._stable_time_threshold:
                self._dispatch_for_inference(ent, frame, now)

    def _dispatch_for_inference(self, ent: SemanticEntity, frame, now):
        """Capture crop and transition STABILIZING → INFERRING."""
        x1, y1, x2, y2 = ent.bbox
        crop = frame[y1:y2, x1:x2]
        if crop.size > 0:
            ent.last_stable_crop = crop.copy()
            ent.last_stable_center = (ent.bbox[:2] + ent.bbox[2:]) / 2.0
            ent.state, ent.last_modified = EntityState.INFERRING, now
            log.debug("Entity %d → INFERRING", ent.id)

    def _erase_unmatched(self, active_entities, matched_ent_ids, now):
        """Immediately erase any entity not matched by a current anchor group."""
        for ent in active_entities:
            if ent.id not in matched_ent_ids:
                ent.state, ent.last_modified = EntityState.ERASED, now
                log.debug("Entity %d → ERASED", ent.id)

    def _create_new_entities(self, groups, matched_indices, now):
        for grp_id, grp in enumerate(groups):
            if grp_id not in matched_indices:
                new_id = self._next_id
                self._next_id += 1
                self._registry[new_id] = SemanticEntity(
                    id=new_id,
                    bbox=grp.bbox.copy(),
                    confidence=grp.confidence,
                    state=EntityState.STABILIZING,
                    first_seen=now,
                    last_seen=now,
                    last_modified=now,
                    ocr_text=None,
                    ocr_confidence=None,
                    last_stable_crop=None,
                    line_bboxes=[a.bbox for a in grp.anchors],
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
