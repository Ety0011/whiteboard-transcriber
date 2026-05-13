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
        content_change_threshold: DINOv2 cosine similarity below this triggers re-OCR.
    """

    def __init__(
        self,
        stable_time_threshold: float = 5.0,
        missing_time_threshold: float = 5.0,
        new_to_growing_time: float = 0.5,
        grace_period_threshold: float = 1.0,
        match_threshold: float = 0.4,
        significant_shift_iou: float = 0.85,
        content_change_threshold: float = 0.92,
    ) -> None:
        self._stable_time_threshold = stable_time_threshold
        self._missing_time_threshold = missing_time_threshold
        self._new_to_growing_time = new_to_growing_time
        self._grace_period_threshold = grace_period_threshold
        self._match_threshold = match_threshold
        self._significant_shift_iou = significant_shift_iou
        self._content_change_threshold = content_change_threshold

        self._registry: dict[int, Region] = {}
        self._next_id: int = 0
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

        # ------------------------------------------------------------------
        # Step A: Build match candidates, greedy assign highest-score first
        # ------------------------------------------------------------------
        candidates: list[tuple[float, int, int]] = []  # (score, det_idx, region_id)
        for di, det in enumerate(detections):
            for reg in active_regions:
                score = _match_score(det.bbox, reg.bbox, frame_diag)
                if score > self._match_threshold:
                    candidates.append((score, di, reg.id))

        candidates.sort(key=lambda x: -x[0])

        matched_det: set[int] = set()
        matched_reg: set[int] = set()
        assignments: dict[int, int] = {}  # det_idx → region_id

        for score, di, rid in candidates:
            if di not in matched_det and rid not in matched_reg:
                assignments[di] = rid
                matched_det.add(di)
                matched_reg.add(rid)

        # ------------------------------------------------------------------
        # Step B: Calculate Global Motion Offset (Consensus)
        # ------------------------------------------------------------------
        global_dx, global_dy = 0.0, 0.0
        match_displacements = []

        for di, rid in assignments.items():
            det = detections[di]
            reg = self._registry[rid]

            # Calculate displacement from old smoothed center to new raw center
            old_cx = (reg.bbox[0] + reg.bbox[2]) / 2.0
            old_cy = (reg.bbox[1] + reg.bbox[3]) / 2.0
            new_cx = (det.bbox[0] + det.bbox[2]) / 2.0
            new_cy = (det.bbox[1] + det.bbox[3]) / 2.0
            match_displacements.append((new_cx - old_cx, new_cy - old_cy))

        if match_displacements:
            # Use Median to ignore outliers
            global_dx = float(np.median([d[0] for d in match_displacements]))
            global_dy = float(np.median([d[1] for d in match_displacements]))

        # ------------------------------------------------------------------
        # Step C: Update matched regions and advance state machine
        # ------------------------------------------------------------------
        for di, rid in assignments.items():
            det = detections[di]
            reg = self._registry[rid]

            # If a MISSING region returns, it must re-stabilize
            if reg.state == RegionState.MISSING:
                reg.state = RegionState.GROWING
                reg.last_modified = now  # Reset timer for stabilization
                log.debug("Region %d recovered from MISSING → GROWING", rid)

            # Compensate for global shift before checking for significant shift
            compensated_prev = (
                int(reg.bbox[0] + global_dx),
                int(reg.bbox[1] + global_dy),
                int(reg.bbox[2] + global_dx),
                int(reg.bbox[3] + global_dy),
            )
            raw_iou = _iou(det.bbox, compensated_prev)
            significant_shift = raw_iou < self._significant_shift_iou

            reg.bbox = _ema_bbox(det.bbox, reg.bbox)
            reg.confidence = det.confidence
            reg.last_seen = now
            reg.missing_frames = 0

            if significant_shift:
                reg.stable_frames = 0
                reg.last_modified = now
                # If a stable region moves, it becomes GROWING and its text is stale
                if reg.state == RegionState.STABLE:
                    reg.state = RegionState.GROWING
                    reg.ocr_text = None  # Clear metadata so Stage 6 re-runs
            else:
                reg.stable_frames += 1

            # State transitions
            if reg.state == RegionState.NEW:
                if now - reg.first_seen >= self._new_to_growing_time:
                    reg.state = RegionState.GROWING
                    reg.last_modified = now

            elif reg.state == RegionState.GROWING:
                if now - reg.last_modified >= self._stable_time_threshold:
                    # Capture baseline data immediately upon stabilization
                    x1, y1, x2, y2 = reg.bbox
                    stable_crop = frame[y1:y2, x1:x2]

                    if stable_crop.size > 0:
                        reg.last_stable_crop = stable_crop.copy()
                        if self._dino is not None:
                            reg.last_stable_embedding = self._embed(stable_crop)

                        reg.state = RegionState.STABLE
                        reg.last_modified = now
                        log.debug("Region %d → STABLE (baseline captured)", rid)

            elif reg.state == RegionState.STABLE:
                pass
                # TODO: we assume that new text appears only after erasing
                # what about typo corrections? therefore important to check after GROWING -> STABLE for changes

                # Content change check: if something is added to existing text
                # if (
                #     self._dino is not None
                #     and reg.last_stable_embedding is not None
                #     and not significant_shift
                # ):
                #     x1, y1, x2, y2 = reg.bbox
                #     current_crop = frame[y1:y2, x1:x2]
                #     if current_crop.size > 0:
                #         emb_current = self._embed(current_crop)
                #         sim = self._cosine_similarity(
                #             emb_current, reg.last_stable_embedding
                #         )

                #         if sim < self._content_change_threshold:
                #             reg.state = RegionState.GROWING
                #             reg.stable_frames = 0
                #             reg.last_modified = now
                #             reg.ocr_text = None  # Mark for re-OCR

        # ------------------------------------------------------------------
        # Step D: Unmatched regions — transition to MISSING
        # ------------------------------------------------------------------
        for reg in active_regions:
            if reg.id not in matched_reg:
                # TODO: missing_frames is useless
                reg.missing_frames += 1

                # Transition to MISSING only after the grace period
                if reg.state in (RegionState.STABLE, RegionState.GROWING):
                    if now - reg.last_seen > self._grace_period_threshold:
                        reg.state = RegionState.MISSING
                        reg.last_modified = now  # Track when it went missing
                        log.debug("Region %d lost detection → MISSING", reg.id)

        # ------------------------------------------------------------------
        # Step E: Unmatched detections — create new regions
        # ------------------------------------------------------------------
        for di, det in enumerate(detections):
            if di not in matched_det:
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
                    _consecutive_visible=1,
                )

        # ------------------------------------------------------------------
        # Step F: Collect newly_stable, newly_erased, prune registry
        # ------------------------------------------------------------------
        newly_stable: list[Region] = []
        newly_erased: list[Region] = []
        to_remove: list[int] = []

        for rid, reg in self._registry.items():
            if reg.state == RegionState.STABLE and reg.last_modified == now:
                newly_stable.append(reg)

            # Only prune regions that have been MISSING too long
            if reg.state == RegionState.MISSING:
                if now - reg.last_seen > self._missing_time_threshold:
                    # FINAL VERIFICATION: Priority check (always runs when threshold hit)
                    if self._dino is not None and reg.last_stable_embedding is not None:
                        x1, y1, x2, y2 = reg.bbox
                        current_crop = frame[y1:y2, x1:x2]
                        if current_crop.size > 0:
                            emb_current = self._embed(current_crop)
                            sim = self._cosine_similarity(
                                emb_current, reg.last_stable_embedding
                            )

                            if sim >= self._content_change_threshold:
                                reg.last_seen = now
                                continue

                    # If verification fails or it's unverified (no embedding), it's gone
                    reg.state = RegionState.ERASED
                    reg.last_modified = now
                    newly_erased.append(reg)
                    to_remove.append(rid)

            # Handle NEW regions that disappear immediately
            elif reg.state == RegionState.NEW:
                if now - reg.last_seen > self._missing_time_threshold:
                    to_remove.append(rid)

        for rid in to_remove:
            del self._registry[rid]

        return TrackerResult(
            regions=list(self._registry.values()),
            newly_stable=newly_stable,
            newly_erased=newly_erased,
        )


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
