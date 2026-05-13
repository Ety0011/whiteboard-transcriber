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

    NEW = "NEW"
    GROWING = "GROWING"
    STABLE = "STABLE"
    MISSING = "MISSING"
    ERASED = "ERASED"


@dataclasses.dataclass
class Detection:
    """A single text-line detection produced by Stage 4."""

    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float


@dataclasses.dataclass
class Region:
    """A persistent tracked region across frames.

    Bounding box is kept EMA-smoothed to reduce jitter. All timestamps are
    from time.monotonic().
    """

    id: int
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 — EMA-smoothed
    confidence: float
    state: RegionState
    first_seen: float
    last_seen: float
    last_modified: float
    stable_frames: int
    missing_frames: int
    ocr_text: str | None
    ocr_confidence: float | None
    last_stable_crop: np.ndarray | None  # BGR uint8 crop at last stabilization
    last_stable_center: tuple[float, float] | None = None  # Baseline for drift
    last_stable_embedding: torch.Tensor | None = None  # Cached DINOv2 embedding
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


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
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
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
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
    det_bbox: tuple[int, int, int, int],
    reg_bbox: tuple[int, int, int, int],
    frame_diag: float,
) -> float:
    """Combined detection-to-region match score: 0.7*IoU + 0.3*centroid_similarity."""
    return 0.7 * _iou(det_bbox, reg_bbox) + 0.3 * _centroid_similarity(
        det_bbox, reg_bbox, frame_diag
    )


def _ema_bbox(
    detected: tuple[int, int, int, int],
    previous: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Apply EMA smoothing: 0.2*detected + 0.8*previous, rounded to int."""
    return tuple(int(0.2 * d + 0.8 * p) for d, p in zip(detected, previous))


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
        stable_time_threshold: float = 5.0,
        missing_time_threshold: float = 5.0,
        new_to_growing_time: float = 0.5,
        grace_period_threshold: float = 1.0,
        match_threshold: float = 0.4,
        significant_shift_iou: float = 0.85,
        drift_threshold_px: float = 20.0,
        content_change_threshold: float = 0.92,
        max_checks_per_frame: int = 2,
    ) -> None:
        self._stable_time_threshold = stable_time_threshold
        self._missing_time_threshold = missing_time_threshold
        self._new_to_growing_time = new_to_growing_time
        self._grace_period_threshold = grace_period_threshold
        self._match_threshold = match_threshold
        self._significant_shift_iou = significant_shift_iou
        self._drift_threshold_px = drift_threshold_px
        self._content_change_threshold = content_change_threshold
        self._max_checks_per_frame = max_checks_per_frame

        self._registry: dict[int, Region] = {}
        self._next_id: int = 0
        self._check_index: int = 0  # Round-robin counter
        self._dino: torch.nn.Module | None = None

    def load_dino(self) -> None:
        """Load DINOv2-base (ViT-B/14) from torch.hub. Call once at startup."""
        log.info("Loading DINOv2-base (ViT-B/14) …")
        self._dino = torch.hub.load(
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

        Raises:
            RuntimeError: If DINOv2 was not loaded via load_dino().
        """
        if self._dino is None:
            raise RuntimeError("DINOv2 not loaded — call load_dino() first")
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
            TrackerResult with all active regions.
        """
        now = time.monotonic()
        h, w = frame.shape[:2]
        frame_diag = math.sqrt(h * h + w * w)
        active_regions = list(self._registry.values())

        # Step A: Matching
        assignments, matched_det, matched_reg = self._get_assignments(
            detections, active_regions, frame_diag
        )

        # Step B: Physics Consensus
        dx, dy = self._compute_global_motion(detections, assignments)

        # Step C: Update Matched Regions
        stable_to_check: list[Region] = []
        for di, rid in assignments.items():
            reg = self._update_region(
                detections[di], self._registry[rid], dx, dy, frame, now
            )
            if (
                reg.state == RegionState.STABLE
                and reg.last_stable_embedding is not None
            ):
                stable_to_check.append(reg)

        # Step C.2: Throttled Verification
        self._run_round_robin_check(stable_to_check, frame, now)

        # Step D: Handle Lost Regions (Grace Period)
        self._handle_unmatched(active_regions, matched_reg, now)

        # Step E: Create New Regions
        self._create_new_regions(detections, matched_det, now)

        # Step F: Cleanup & Pardon Logic
        return self._prune_and_pardon(frame, now)

    # -----------------------------------------------------------------------
    # Private Orchestration Helpers
    # -----------------------------------------------------------------------

    def _get_assignments(self, detections, active_regions, diag):
        candidates = []
        for di, det in enumerate(detections):
            for reg in active_regions:
                score = _match_score(det.bbox, reg.bbox, diag)
                if score > self._match_threshold:
                    candidates.append((score, di, reg.id))

        candidates.sort(key=lambda x: -x[0])
        matched_det, matched_reg, assignments = set(), set(), {}

        for _, di, rid in candidates:
            if di not in matched_det and rid not in matched_reg:
                assignments[di] = rid
                matched_det.add(di)
                matched_reg.add(rid)
        return assignments, matched_det, matched_reg

    def _compute_global_motion(self, detections, assignments):
        disps = []
        for di, rid in assignments.items():
            reg, det = self._registry[rid], detections[di]
            old_cx, old_cy = (
                (reg.bbox[0] + reg.bbox[2]) / 2.0,
                (reg.bbox[1] + reg.bbox[3]) / 2.0,
            )
            new_cx, new_cy = (
                (det.bbox[0] + det.bbox[2]) / 2.0,
                (det.bbox[1] + det.bbox[3]) / 2.0,
            )
            disps.append((new_cx - old_cx, new_cy - old_cy))

        if not disps:
            return 0.0, 0.0
        return float(np.median([d[0] for d in disps])), float(
            np.median([d[1] for d in disps])
        )

    def _update_region(self, det, reg, dx, dy, frame, now):
        """Logic for a single matched region."""
        if reg.state == RegionState.MISSING:
            reg.state = RegionState.GROWING
            reg.last_modified = now

        # Shift & Drift Checks
        comp_prev = (
            int(reg.bbox[0] + dx),
            int(reg.bbox[1] + dy),
            int(reg.bbox[2] + dx),
            int(reg.bbox[3] + dy),
        )

        significant_shift = _iou(det.bbox, comp_prev) < self._significant_shift_iou

        if reg.last_stable_center:
            cur_cx, cur_cy = (
                (det.bbox[0] + det.bbox[2]) / 2.0,
                (det.bbox[1] + det.bbox[3]) / 2.0,
            )
            drift = math.sqrt(
                (cur_cx - (reg.last_stable_center[0] + dx)) ** 2
                + (cur_cy - (reg.last_stable_center[1] + dy)) ** 2
            )
            if drift > self._drift_threshold_px:
                significant_shift = True

        # Physical Update
        reg.bbox = _ema_bbox(det.bbox, reg.bbox)
        reg.confidence, reg.last_seen = det.confidence, now

        if significant_shift:
            reg.stable_frames = 0
            reg.last_modified = now
            if reg.state == RegionState.STABLE:
                reg.state, reg.ocr_text = RegionState.GROWING, None
        else:
            reg.stable_frames += 1

        # Transitions
        if reg.state == RegionState.NEW and (
            now - reg.first_seen >= self._new_to_growing_time
        ):
            reg.state, reg.last_modified = RegionState.GROWING, now
        elif reg.state == RegionState.GROWING and (
            now - reg.last_modified >= self._stable_time_threshold
        ):
            self._stabilize_region(reg, frame, now)

        return reg

    def _stabilize_region(self, reg, frame, now):
        x1, y1, x2, y2 = reg.bbox
        crop = frame[y1:y2, x1:x2]
        if crop.size > 0:
            reg.last_stable_crop = crop.copy()
            reg.last_stable_center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            if self._dino:
                reg.last_stable_embedding = self._embed(crop)
            reg.state, reg.last_modified = RegionState.STABLE, now
            log.debug("Region %d → STABLE", reg.id)

    def _run_round_robin_check(self, regions, frame, now):
        if not regions:
            return
        num_to_check = min(len(regions), self._max_checks_per_frame)
        for _ in range(num_to_check):
            self._check_index %= len(regions)
            reg = regions[self._check_index]
            self._check_index += 1

            x1, y1, x2, y2 = reg.bbox
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                sim = self._cosine_similarity(
                    self._embed(crop), reg.last_stable_embedding
                )
                if sim < self._content_change_threshold:
                    reg.state, reg.last_modified, reg.ocr_text = (
                        RegionState.GROWING,
                        now,
                        None,
                    )

    def _handle_unmatched(self, active_regions, matched_reg_ids, now):
        for reg in active_regions:
            if reg.id not in matched_reg_ids:
                if reg.state in (RegionState.STABLE, RegionState.GROWING):
                    if now - reg.last_seen > self._grace_period_threshold:
                        reg.state, reg.last_modified = RegionState.MISSING, now

    def _create_new_regions(self, detections, matched_indices, now):
        for di, det in enumerate(detections):
            if di not in matched_indices:
                new_id = self._next_id
                self._next_id += 1
                self._registry[new_id] = Region(
                    id=new_id,
                    bbox=det.bbox,
                    confidence=det.confidence,
                    state=RegionState.NEW,
                    first_seen=now,
                    last_seen=now,
                    last_modified=now,
                    stable_frames=0,
                    missing_frames=0,
                    ocr_text=None,
                    ocr_confidence=None,
                    last_stable_crop=None,
                )

    def _prune_and_pardon(self, frame, now):
        newly_stable, newly_erased, to_remove = [], [], []
        for rid, reg in self._registry.items():
            if reg.state == RegionState.STABLE and (now - reg.last_modified) < 0.1:
                newly_stable.append(reg)

            if reg.state == RegionState.MISSING:
                if now - reg.last_seen > self._missing_time_threshold:
                    # DINO Pardon Check
                    if self._dino and reg.last_stable_embedding is not None:
                        x1, y1, x2, y2 = reg.bbox
                        crop = frame[y1:y2, x1:x2]
                        if (
                            crop.size > 0
                            and self._cosine_similarity(
                                self._embed(crop), reg.last_stable_embedding
                            )
                            >= self._content_change_threshold
                        ):
                            reg.last_seen = now
                            continue

                    reg.state, reg.last_modified = RegionState.ERASED, now
                    newly_erased.append(reg)
                    to_remove.append(rid)

            elif reg.state == RegionState.NEW and (
                now - reg.last_seen > self._missing_time_threshold
            ):
                to_remove.append(rid)

        for rid in to_remove:
            del self._registry[rid]
        return TrackerResult(list(self._registry.values()), newly_stable, newly_erased)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_global_tracker: RegionTracker | None = None


def init(
    stable_time_threshold: float = 2.0,
    missing_time_threshold: float = 2.0,
    new_to_growing_time: float = 0.5,
    match_threshold: float = 0.4,
    significant_shift_iou: float = 0.85,
    content_change_threshold: float = 0.92,
) -> None:
    """Create the module-level tracker singleton and load DINOv2. Call once at startup."""
    global _global_tracker
    _global_tracker = RegionTracker(
        stable_time_threshold=stable_time_threshold,
        missing_time_threshold=missing_time_threshold,
        new_to_growing_time=new_to_growing_time,
        match_threshold=match_threshold,
        significant_shift_iou=significant_shift_iou,
        content_change_threshold=content_change_threshold,
    )
    _global_tracker.load_dino()


def process(detections: list[Detection], frame: np.ndarray) -> TrackerResult:
    """Run one tracking cycle using the module-level singleton."""
    global _global_tracker
    if _global_tracker is None:
        log.warning("tracker.process() called before init() — using defaults")
        _global_tracker = RegionTracker()
        _global_tracker.load_dino()
    return _global_tracker.process(detections, frame)
