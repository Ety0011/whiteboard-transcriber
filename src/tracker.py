"""Stage 5 — Region Tracker.

Maintains a persistent registry of Region objects across frames. Each frame,
raw detections are matched to existing regions using IoU + centroid scoring,
bounding boxes are EMA-smoothed, and the state machine is advanced. Newly
stable regions are returned for Stage 6 (OCR). Content change on OCR_DONE
regions is detected via DINOv2-base cosine similarity.
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
    OCR_DONE = "OCR_DONE"
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
    last_stable_crop: np.ndarray | None  # BGR uint8 crop at last OCR stabilization
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


class RegionTracker:
    """Persistent region registry for the whiteboard tracker (Stage 5).

    Matches frame detections to existing regions, applies EMA bbox smoothing,
    advances the state machine, and exposes newly stable or erased regions each
    frame. DINOv2-base detects content change on OCR_DONE regions.

    Args:
        stable_frames_threshold: Consecutive matched frames without significant
            bbox shift required before GROWING → STABLE.
        missing_frames_threshold: Consecutive unmatched frames before ERASED.
        new_to_growing_frames: Consecutive matched frames required for NEW → GROWING.
        match_threshold: Minimum combined score to match a detection to a region.
        significant_shift_iou: Raw detection IoU below this resets stable_frames.
        content_change_threshold: DINOv2 cosine similarity below this triggers
            OCR_DONE → GROWING.
    """

    # TODO: use time for thresholds
    def __init__(
        self,
        stable_frames_threshold: int = 100,
        missing_frames_threshold: int = 100,
        new_to_growing_frames: int = 3,
        match_threshold: float = 0.4,
        significant_shift_iou: float = 0.85,
        content_change_threshold: float = 0.92,
    ) -> None:
        self._stable_frames_threshold = stable_frames_threshold
        self._missing_frames_threshold = missing_frames_threshold
        self._new_to_growing_frames = new_to_growing_frames
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
        stable_crop: np.ndarray,
    ) -> None:
        """Record OCR result and transition the region to OCR_DONE.

        Called by pipeline.py after Stage 6 finishes on a newly_stable region.
        Stores OCR text, confidence, and the crop as the DINOv2 baseline for
        future content-change detection.

        Args:
            region_id:   The Region.id to update.
            text:        Full OCR text for the region.
            confidence:  OCR confidence score.
            stable_crop: BGR uint8 crop used during OCR (DINOv2 baseline).

        Raises:
            KeyError: If region_id is not in the active registry.
        """
        region = self._registry[region_id]
        region.ocr_text = text
        region.ocr_confidence = confidence
        region.last_stable_crop = stable_crop.copy()
        region.state = RegionState.OCR_DONE
        region.last_modified = time.monotonic()
        log.debug(
            "Region %d → OCR_DONE (conf=%.2f, text=%r…)",
            region_id,
            confidence,
            text[:40],
        )

    def process(
        self,
        detections: list[Detection],
        frame: np.ndarray,
    ) -> TrackerResult:
        """Run one tracking cycle: match detections, advance state machine.

        Args:
            detections: Raw text-line detections from Stage 4.
            frame:      Current BGR frame (used for DINOv2 crop extraction).

        Returns:
            TrackerResult with all active regions, plus newly-stable and
            newly-erased regions from this frame.
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
        # Step B+C: Update matched regions and advance state machine
        # ------------------------------------------------------------------
        for di, rid in assignments.items():
            det = detections[di]
            reg = self._registry[rid]

            # Compare raw detection against current smoothed bbox for stability
            raw_iou = _iou(det.bbox, reg.bbox)
            significant_shift = raw_iou < self._significant_shift_iou

            reg.bbox = _ema_bbox(det.bbox, reg.bbox)
            reg.confidence = det.confidence
            reg.last_seen = now
            reg.missing_frames = 0
            reg._consecutive_visible += 1

            if significant_shift:
                reg.stable_frames = 0
                reg.last_modified = now
            else:
                reg.stable_frames += 1

            # State transitions
            if reg.state == RegionState.NEW:
                if reg._consecutive_visible >= self._new_to_growing_frames:
                    reg.state = RegionState.GROWING
                    reg.last_modified = now

            # TODO: adjusting regions are flagged as stable, is this ok?
            elif reg.state == RegionState.GROWING:
                if reg.stable_frames > self._stable_frames_threshold:
                    reg.state = RegionState.STABLE
                    reg.last_modified = now

            elif reg.state == RegionState.OCR_DONE:
                if (
                    self._dino is not None
                    and reg.last_stable_crop is not None
                    and not significant_shift
                ):
                    x1, y1, x2, y2 = reg.bbox
                    current_crop = frame[y1:y2, x1:x2]
                    if current_crop.size > 0:
                        emb_current = self._embed(current_crop)
                        emb_stored = self._embed(reg.last_stable_crop)
                        sim = self._cosine_similarity(emb_current, emb_stored)
                        if sim < self._content_change_threshold:
                            reg.state = RegionState.GROWING
                            reg.stable_frames = 0
                            reg.last_modified = now
                            log.debug(
                                "Region %d content changed (cos=%.3f) → GROWING",
                                rid,
                                sim,
                            )

        # ------------------------------------------------------------------
        # Step D: Unmatched regions — increment missing counter
        # ------------------------------------------------------------------
        for reg in active_regions:
            if reg.id not in matched_reg:
                reg.missing_frames += 1
                reg._consecutive_visible = 0

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

            # TODO: not enough to just wait, check with DINO if actually erased
            if (
                reg.state != RegionState.ERASED
                and reg.missing_frames > self._missing_frames_threshold
            ):
                reg.state = RegionState.ERASED
                reg.last_modified = now
                newly_erased.append(reg)
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
    stable_frames_threshold: int = 20,
    missing_frames_threshold: int = 20,
    new_to_growing_frames: int = 3,
    match_threshold: float = 0.4,
    significant_shift_iou: float = 0.85,
    content_change_threshold: float = 0.92,
) -> None:
    """Create the module-level tracker singleton and load DINOv2. Call once at startup.

    Args:
        stable_frames_threshold: Frames before GROWING → STABLE.
        missing_frames_threshold: Frames before ERASED.
        new_to_growing_frames: Consecutive frames before NEW → GROWING.
        match_threshold: Minimum score to match a detection to a region.
        significant_shift_iou: Raw IoU below which stable_frames resets.
        content_change_threshold: DINOv2 cosine below which OCR_DONE → GROWING.
    """
    global _global_tracker
    _global_tracker = RegionTracker(
        stable_frames_threshold=stable_frames_threshold,
        missing_frames_threshold=missing_frames_threshold,
        new_to_growing_frames=new_to_growing_frames,
        match_threshold=match_threshold,
        significant_shift_iou=significant_shift_iou,
        content_change_threshold=content_change_threshold,
    )
    _global_tracker.load_dino()


def process(detections: list[Detection], frame: np.ndarray) -> TrackerResult:
    """Run one tracking cycle using the module-level singleton.

    Lazily initialises with default settings (and logs a warning) if init()
    was not called first.

    Args:
        detections: Raw text-line detections from Stage 4.
        frame:      Current BGR frame.

    Returns:
        TrackerResult with active, newly-stable, and newly-erased regions.
    """
    global _global_tracker
    if _global_tracker is None:
        log.warning("tracker.process() called before init() — using defaults")
        _global_tracker = RegionTracker()
        _global_tracker.load_dino()
    return _global_tracker.process(detections, frame)
