"""Stage 5 — Region Tracker.

Maintains a persistent registry of Region objects across frames. Each frame,
raw detections are matched to existing regions using IoU + centroid scoring,
bounding boxes are EMA-smoothed, and the state machine is advanced.

The whiteboard is modeled as a set of physical regions. OCR status is handled
as metadata (ocr_text) rather than a lifecycle state. DINOv2-base embeddings
are used to detect content changes and verify region presence during occlusion.
"""

from __future__ import annotations

import dataclasses
import enum
import logging
import math
import time
import warnings

import cv2
import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class RegionState(enum.Enum):
    """Lifecycle states for a tracked region."""

    CANDIDATE = "CANDIDATE"
    STABILIZING = "STABILIZING"
    STABLE = "STABLE"
    MISSING = "MISSING"
    REMOVED = "REMOVED"


@dataclasses.dataclass
class Detection:
    """A single text-line detection produced by Stage 4."""

    bbox: np.ndarray  # shape (4,) int32: x1, y1, x2, y2
    confidence: float
    # axis-aligned bboxes for sub-lines within this detection
    line_bboxes: list[np.ndarray] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Region:
    """A persistent tracked region across frames.

    Bounding box is kept EMA-smoothed to reduce jitter. All timestamps are
    from time.monotonic().
    """

    id: int
    bbox: np.ndarray  # shape (4,) int32: x1, y1, x2, y2 — EMA-smoothed
    confidence: float
    state: RegionState
    first_seen: float
    last_modified: float
    last_seen: float
    ocr_text: str | None
    ocr_confidence: float | None
    last_stable_crop: np.ndarray | None  # BGR uint8 crop at last stabilization
    last_stable_center: np.ndarray | None = None  # shape (2,) float64 cx,cy
    last_stable_embedding: torch.Tensor | None = None  # Cached DINOv2 embedding
    line_bboxes: list[np.ndarray] = dataclasses.field(default_factory=list)
    _consecutive_visible: int = dataclasses.field(default=0, repr=False)


@dataclasses.dataclass
class TrackerResult:
    """Output of one tracker processing cycle."""

    regions: list[Region]  # all active (non-ERASED) regions
    newly_stable: list[Region]  # transitioned to STABLE this frame
    newly_erased: list[Region]  # transitioned to ERASED this frame


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
    """Combined detection-to-region match score: 0.7*IoU + 0.3*centroid_similarity."""
    return 0.7 * _iou(det_bbox, reg_bbox) + 0.3 * _centroid_similarity(
        det_bbox, reg_bbox, frame_diag
    )


# ---------------------------------------------------------------------------
# DINOv2 preprocessing
# ---------------------------------------------------------------------------


def _preprocess_crop(crop_bgr: np.ndarray) -> torch.Tensor:
    """Convert a BGR uint8 crop to a DINOv2-ready (1,3,H,W) float32 tensor.

    Steps:
      1. BGR → RGB
      2. Resize so both dimensions are multiples of 14 (minimum 14×14)
      3. Normalize [0,255] → [0.0,1.0] then apply ImageNet mean/std
      4. Add batch dimension

    Args:
        crop_bgr: BGR uint8 numpy array from OpenCV.

    Returns:
        Float32 tensor of shape (1, 3, H', W').
    """
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    new_h = max(14, (h // 14) * 14)
    new_w = max(14, (w // 14) * 14)
    if new_h != h or new_w != w:
        rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    t = (t - mean) / std
    return t.unsqueeze(0)


# ---------------------------------------------------------------------------
# RegionTracker
# ---------------------------------------------------------------------------


# TODO: fix duplicate regions
class RegionTracker:
    """Persistent region registry for the whiteboard tracker (Stage 5).

    Matches frame detections to existing regions, applies EMA bbox smoothing,
    advances the state machine, and exposes newly stable or erased regions each
    frame. DINOv2-base detects content change on STABLE regions.

    Args:
        stable_time_threshold: Duration (seconds) without significant bbox shift
            required before GROWING → STABLE.
        missing_time_threshold: Duration (seconds) without matches before ERASED.
        new_to_growing_time: Duration (seconds) of presence required for NEW → GROWING.
        grace_period_threshold: Duration (seconds) before unmatched regions transition
            to MISSING.
        match_threshold: Minimum combined score to match a detection to a region.
        significant_shift_iou: Raw detection IoU below this resets timers.
        drift_threshold_px: Cumulative Euclidean drift from last stable center.
        content_change_threshold: DINOv2 cosine similarity below this triggers re-OCR.
        max_checks_per_frame: Number of STABLE regions to check with DINO per frame.
    """

    def __init__(
        self,
        stabilizing_time_threshold: float = 5.0,
        stable_time_threshold: float = 5.0,
        grace_time_threshold: float = 5.0,
        missing_time_threshold: float = 5.0,
        removed_time_threshold: float = 5.0,
        match_threshold: float = 0.4,
        drift_threshold_px: float = 20.0,
    ) -> None:
        self._stabilizing_time_threshold = stabilizing_time_threshold
        self._stable_time_threshold = stable_time_threshold
        self._grace_time_threshold = grace_time_threshold
        self._missing_time_threshold = missing_time_threshold
        self._removed_retention_threshold = removed_time_threshold
        self._match_threshold = match_threshold
        self._drift_threshold_px = drift_threshold_px

        self._registry: dict[int, Region] = {}
        self._next_id: int = 0

        warnings.filterwarnings("ignore", message="xFormers is not available")
        log.info("Loading DINOv2-base (ViT-B/14) …")
        self._dino: torch.nn.Module = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vitb14",
            pretrained=True,
        )
        self._dino.eval()
        log.info("DINOv2-base loaded.")

    def _embed(self, crop_bgr: np.ndarray) -> torch.Tensor:
        """Return the L2-normalized DINOv2 CLS token embedding for *crop_bgr*.

        Args:
            crop_bgr: BGR uint8 numpy array.

        Returns:
            Float32 tensor of shape (768,), L2-normalized.
        """
        t = _preprocess_crop(crop_bgr)
        with torch.no_grad():
            feats = self._dino.forward_features(t)
        cls_token = feats["x_norm_clstoken"].squeeze(0)
        return F.normalize(cls_token, dim=0)

    def _cosine_similarity(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Cosine similarity of two embedding vectors.

        Clamps to [-1, 1] to handle floating-point rounding on L2-normalized inputs.
        """
        return float(torch.dot(a, b).clamp(-1.0, 1.0))

    def mark_ocr_done(
        self,
        region_id: int,
        text: str,
        confidence: float,
    ) -> None:
        """Record OCR result metadata for a region.

        The physical state remains STABLE. Orchestrators should check
        if ocr_text is None to identify regions needing recognition.
        """
        region = self._registry[region_id]
        region.ocr_text = text
        region.ocr_confidence = confidence
        region.last_modified = time.monotonic()
        log.debug("Metadata updated for Region %d: %r", region_id, text[:30])

    def process(
        self,
        detections: list[Detection],
        frame: np.ndarray,
    ) -> TrackerResult:
        """Run one tracking cycle: match detections, advance state machine.

        Args:
            detections: Raw text-line detections from Stage 4.
            frame:      Current BGR background composite (Stage 3).

        Returns:
            List with all active regions.
        """
        now = time.monotonic()
        h, w = frame.shape[:2]
        frame_diag = math.sqrt(h * h + w * w)

        active_regions = [
            r for r in self._registry.values() if r.state != RegionState.REMOVED
        ]

        # Step A: Matching
        assignments, matched_det, matched_reg = self._get_assignments(
            detections, active_regions, frame_diag
        )

        # Step C: Update Matched Regions
        for det_id, reg_id in assignments.items():
            self._update_region(detections[det_id], self._registry[reg_id], frame, now)

        # Step D: Handle Lost Regions (regions not in assignments)
        self._handle_unmatched(active_regions, matched_reg, now)

        # Step E: Create New Regions (detections not in assignments)
        self._create_new_regions(detections, matched_det, now)

        # Step F: Cleanup & Pardon Logic
        self._remove_missing_regions(now)

        return TrackerResult(
            regions=list(self._registry.values()),
            newly_stable=[
                r
                for r in self._registry.values()
                if r.state == RegionState.STABLE and r.last_modified == now
            ],
            newly_erased=[
                r
                for r in self._registry.values()
                if r.state == RegionState.REMOVED and r.last_modified == now
            ],
        )

    # -----------------------------------------------------------------------
    # Private Orchestration Helpers
    # -----------------------------------------------------------------------

    # TODO: if board moves too much we lose all assignments
    def _get_assignments(self, detections, active_regions, diag):
        candidates = []
        for det_id, det in enumerate(detections):
            for reg in active_regions:
                score = _match_score(det.bbox, reg.bbox, diag)
                if score > self._match_threshold:
                    candidates.append((score, det_id, reg.id))

        candidates.sort(key=lambda x: -x[0])

        matched_det: set[int] = set()
        matched_reg: set[int] = set()
        assignments: dict[int, int] = {}

        for _, det_id, reg_id in candidates:
            if det_id not in matched_det and reg_id not in matched_reg:
                assignments[det_id] = reg_id
                matched_det.add(det_id)
                matched_reg.add(reg_id)
        return assignments, matched_det, matched_reg

    def _update_region(self, det, reg, frame, now):
        """Logic for a single matched region."""

        if reg.last_stable_center is not None:
            cur_center = (det.bbox[:2] + det.bbox[2:]) / 2.0
            drift = float(np.linalg.norm(cur_center - reg.last_stable_center))
            if drift > self._drift_threshold_px and reg.state == RegionState.STABLE:
                reg.state, reg.ocr_text = RegionState.STABILIZING, None
                reg.last_modified = now

        # Physical Update — EMA bbox smoothing
        reg.bbox = (0.2 * det.bbox + 0.8 * reg.bbox).astype(np.int32)
        reg.confidence, reg.last_seen = det.confidence, now
        reg.line_bboxes = det.line_bboxes

        # Transitions
        if reg.state == RegionState.CANDIDATE:
            if now - reg.first_seen >= self._stabilizing_time_threshold:
                reg.state, reg.last_modified = RegionState.STABILIZING, now

        elif reg.state == RegionState.STABILIZING:
            if now - reg.last_modified >= self._stable_time_threshold:
                self._stabilize_region(reg, frame, now)

        elif reg.state == RegionState.MISSING:
            reg.state = RegionState.STABILIZING
            reg.last_modified = now

    def _stabilize_region(self, reg, frame, now):
        x1, y1, x2, y2 = reg.bbox
        crop = frame[y1:y2, x1:x2]
        if crop.size > 0:
            reg.last_stable_crop = crop.copy()
            reg.last_stable_center = (reg.bbox[:2] + reg.bbox[2:]) / 2.0
            reg.last_stable_embedding = self._embed(crop)
            reg.state, reg.last_modified = RegionState.STABLE, now
            log.debug("Region %d → STABLE", reg.id)

    def _handle_unmatched(self, active_regions, matched_reg_ids, now):
        for reg in active_regions:
            if reg.id not in matched_reg_ids:
                if reg.state in (RegionState.STABLE, RegionState.STABILIZING):
                    if now - reg.last_seen > self._grace_time_threshold:
                        reg.state, reg.last_modified = RegionState.MISSING, now

    def _create_new_regions(self, detections, matched_indices, now):
        for det_id, det in enumerate(detections):
            if det_id not in matched_indices:
                new_id = self._next_id
                self._next_id += 1
                self._registry[new_id] = Region(
                    id=new_id,
                    bbox=det.bbox.copy(),
                    confidence=det.confidence,
                    state=RegionState.CANDIDATE,
                    first_seen=now,
                    last_seen=now,
                    last_modified=now,
                    ocr_text=None,
                    ocr_confidence=None,
                    last_stable_crop=None,
                    line_bboxes=list(det.line_bboxes),
                )

    def _remove_missing_regions(self, now):
        to_remove = []
        for reg_id, reg in self._registry.items():
            if reg.state == RegionState.MISSING:
                if now - reg.last_modified > self._missing_time_threshold:
                    reg.state, reg.last_modified = RegionState.REMOVED, now

            elif reg.state == RegionState.REMOVED:
                if now - reg.last_modified > self._removed_retention_threshold:
                    to_remove.append(reg_id)

            elif reg.state == RegionState.CANDIDATE:
                if now - reg.last_seen > self._missing_time_threshold:
                    to_remove.append(reg_id)

        for reg_id in to_remove:
            del self._registry[reg_id]
