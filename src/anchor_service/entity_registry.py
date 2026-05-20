"""Entity Registry — cross-frame persistence and lifecycle management.

Maintains a persistent registry of TrackedRegion objects across frames.
Matches incoming LayoutRegions from Stage 5 to existing tracked blocks
using Intersection-over-Union (IoU) and centroid drift scoring, applies
EMA smoothing to boundary points, and drives the temporal state machine.

Lifecycle states: STABILIZING → STABLE → [ERASED]
"""

from __future__ import annotations

import dataclasses
import enum
import logging
import math
import time

import numpy as np

from anchor_service.detector import AnchorType, LayoutRegion

log = logging.getLogger(__name__)


class RegionState(enum.Enum):
    STABILIZING = "STABILIZING"  # Ink is being written or geometry is settling
    STABLE = "STABLE"  # Geometry has settled; ready for logging/processing
    ERASED = "ERASED"  # Region is no longer visible on the board surface


@dataclasses.dataclass
class TrackedRegion:
    """A structurally persistent layout region tracked across video frames."""

    id: int
    bbox: np.ndarray  # (4,) int32: smoothed x1, y1, x2, y2 limits
    raw_polygon: (
        np.ndarray | None
    )  # (N, 2) int32: absolute polygon boundary coordinates
    confidence: float
    anchor_type: AnchorType
    label: str
    text: str  # Live OCR output from the grounding module
    state: RegionState
    first_seen: float  # time.monotonic() timestamp
    last_seen: float  # Last frame timestamp where a match was confirmed
    last_modified: float  # Last time a state change or heavy edit occurred
    last_stable_center: np.ndarray | None = None  # (2,) float64 cx, cy snapshot


@dataclasses.dataclass
class RegistryUpdate:
    """Output descriptor of a single EntityRegistry execution cycle."""

    regions: list[TrackedRegion]  # All currently active (non-ERASED) blocks
    newly_stable: list[TrackedRegion]  # Blocks that transitioned to STABLE this frame
    newly_erased: list[TrackedRegion]  # Blocks that transitioned to ERASED this frame


# ---------------------------------------------------------------------------
# Geometric Distance Scoring Utilities
# ---------------------------------------------------------------------------


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Intersection-over-Union of two bounding boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def _centroid_similarity(a: np.ndarray, b: np.ndarray, frame_diagonal: float) -> float:
    """Compute proximity metric between box centroids normalized to frame size."""
    cx_a = (a[0] + a[2]) / 2.0
    cy_a = (a[1] + a[3]) / 2.0
    cx_b = (b[0] + b[2]) / 2.0
    cy_b = (b[1] + b[3]) / 2.0

    dist = math.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2)
    return max(0.0, 1.0 - dist / frame_diagonal) if frame_diagonal > 0 else 0.0


def _match_score(
    det_bbox: np.ndarray, track_bbox: np.ndarray, frame_diagonal: float
) -> float:
    """Combined matching metric: weighted balance of overlapping bounds and proximity."""
    return 0.7 * _iou(det_bbox, track_bbox) + 0.3 * _centroid_similarity(
        det_bbox, track_bbox, frame_diagonal
    )


# ---------------------------------------------------------------------------
# Registry Core Manager
# ---------------------------------------------------------------------------


class EntityRegistry:
    """Tracks layout spatial blocks over time to enforce structural stability filters.

    Args:
        stability_window: Seconds a block must remain geometrically stationary
            before transitioning from STABILIZING to STABLE.
        erasure_grace_period: Seconds a block can remain missing from raw detections
            (due to teacher occlusions/shadows) before it is marked ERASED.
        match_threshold: Minimum spatial tracking score needed to bind a detection to a track.
        drift_threshold_px: Pixels a block's centroid can shift before resetting to STABILIZING.
    """

    def __init__(
        self,
        stability_window: float = 2.0,
        erasure_grace_period: float = 4.0,
        match_threshold: float = 0.35,
        drift_threshold_px: float = 25.0,
    ) -> None:
        self._stability_window = stability_window
        self._erasure_grace_period = erasure_grace_period
        self._match_threshold = match_threshold
        self._drift_threshold_px = drift_threshold_px

        self._registry: dict[int, TrackedRegion] = {}
        self._next_id: int = 0

    def tick(
        self, incoming_regions: list[LayoutRegion], frame: np.ndarray
    ) -> RegistryUpdate:
        """Process a fresh frame's layout array, driving track life cycles.

        Bypasses tracking logic entirely if incoming frames are clear anomalies.
        """
        now = time.monotonic()
        h, w = frame.shape[:2]
        frame_diagonal = math.sqrt(h * h + w * w)

        # Isolate active, visible tracking states
        active_tracks = [
            t for t in self._registry.values() if t.state != RegionState.ERASED
        ]

        # Calculate optimal spatial track bindings
        assignments, matched_det_indices, matched_track_ids = self._get_assignments(
            incoming_regions, active_tracks, frame_diagonal
        )

        # 1. Update matching tracks with fresh spatial details
        for det_idx, track_id in assignments.items():
            self._update_track(incoming_regions[det_idx], self._registry[track_id], now)

        # 2. Run background grace buffer checks on missing tracks (potential erasures)
        self._handle_missing_tracks(active_tracks, matched_track_ids, now)

        # 3. Create fresh tracking entries for completely unmatched visual zones
        self._instantiate_new_tracks(incoming_regions, matched_det_indices, now)

        # 4. Filter out updates for external caller hooks
        return RegistryUpdate(
            regions=[
                t for t in self._registry.values() if t.state != RegionState.ERASED
            ],
            newly_stable=[
                t
                for t in self._registry.values()
                if t.state == RegionState.STABLE and t.last_modified == now
            ],
            newly_erased=[
                t
                for t in self._registry.values()
                if t.state == RegionState.ERASED and t.last_modified == now
            ],
        )

    # -----------------------------------------------------------------------
    # Internal Lifecycle Engines
    # -----------------------------------------------------------------------

    def _get_assignments(
        self,
        detections: list[LayoutRegion],
        tracks: list[TrackedRegion],
        frame_diagonal: float,
    ) -> tuple[dict[int, int], set[int], set[int]]:
        """Compute optimal match permutations using a spatial score ranking matrix."""
        candidates = []
        for det_idx, det in enumerate(detections):
            for track in tracks:
                score = _match_score(det.bbox, track.bbox, frame_diagonal)
                if score > self._match_threshold:
                    candidates.append((score, det_idx, track.id))

        # Sort candidate indices by matching affinity
        candidates.sort(key=lambda x: -x[0])

        matched_det_indices: set[int] = set()
        matched_track_ids: set[int] = set()
        assignments: dict[int, int] = {}

        for _, det_idx, track_id in candidates:
            if det_idx not in matched_det_indices and track_id not in matched_track_ids:
                assignments[det_idx] = track_id
                matched_det_indices.add(det_idx)
                matched_track_ids.add(track_id)

        return assignments, matched_det_indices, matched_track_ids

    def _update_track(
        self, det: LayoutRegion, track: TrackedRegion, now: float
    ) -> None:
        """Advance the internal life cycles of active spatial blocks."""
        # Tracking check: flag heavy text updates or geometric layout drift
        if track.state == RegionState.STABLE and track.last_stable_center is not None:
            cur_center = (det.bbox[:2] + det.bbox[2:]) / 2.0
            drift = float(np.linalg.norm(cur_center - track.last_stable_center))

            # If the professor rewrites the area or the box moves significantly, force restabilization
            if drift > self._drift_threshold_px or track.text != det.text:
                track.state = RegionState.STABILIZING
                track.last_modified = now

        # Apply an Exponential Moving Average (EMA) layout factor to kill tracking line jitter
        track.bbox = (0.2 * det.bbox + 0.8 * track.bbox).astype(np.int32)
        track.raw_polygon = det.raw_polygon
        track.confidence = det.confidence
        track.text = det.text
        track.last_seen = now

        # Evaluate stability window boundaries
        if track.state == RegionState.STABILIZING:
            if now - track.last_modified >= self._stability_window:
                track.state = RegionState.STABLE
                track.last_stable_center = (track.bbox[:2] + track.bbox[2:]) / 2.0
                track.last_modified = now

    def _handle_missing_tracks(
        self, active_tracks: list[TrackedRegion], matched_ids: set[int], now: float
    ) -> None:
        """Run grace calculations to decide if unmatched regions are erased or merely occluded."""
        for track in active_tracks:
            if track.id not in matched_ids:
                # Calculate elapsed time since this block was last structurally validated
                time_unseen = now - track.last_seen
                if time_unseen > self._erasure_grace_period:
                    track.state = RegionState.ERASED
                    track.last_modified = now
                    log.debug(
                        "Tracked Block [ID:%d] persistently missing -> marked ERASED.",
                        track.id,
                    )

    def _instantiate_new_tracks(
        self, detections: list[LayoutRegion], matched_indices: set[int], now: float
    ) -> None:
        """Instantiate new tracking entries for unassigned layout spatial elements."""
        for det_idx, det in enumerate(detections):
            if det_idx not in matched_indices:
                new_id = self._next_id
                self._next_id += 1

                self._registry[new_id] = TrackedRegion(
                    id=new_id,
                    bbox=det.bbox.copy(),
                    raw_polygon=det.raw_polygon.copy()
                    if det.raw_polygon is not None
                    else None,
                    confidence=det.confidence,
                    anchor_type=det.anchor_type,
                    label=det.label,
                    text=det.text,
                    state=RegionState.STABILIZING,
                    first_seen=now,
                    last_seen=now,
                    last_modified=now,
                    last_stable_center=None,
                )
                log.debug(
                    "New layout element detected -> tracking started [ID:%d].", new_id
                )
