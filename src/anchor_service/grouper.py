"""Stage 6 — Hierarchical Entity Grouper.

Clusters line-level Spatial Anchors from Stage 5 into Semantic Entities using
spatial proximity. Lines that are vertically close and horizontally aligned are
merged into one EntityGroup with a single merged bounding box.

This replaces the one-anchor-per-Detection mapping in the pipeline. Each
EntityGroup becomes one Detection fed to the region tracker, so OCR crops
cover full multi-line blocks rather than individual lines.

Stateless and synchronous — runs in the main process in O(N log N).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from anchor_service.detector import Anchor


@dataclass
class EntityGroup:
    anchors: list[Anchor]   # constituent anchors, top-to-bottom
    bbox: np.ndarray        # (4,) int32: min(x1), min(y1), max(x2), max(y2)
    confidence: float       # max confidence across constituent anchors


class EntityGrouper:
    """Stateless spatial proximity grouper.

    Two anchors merge into the same entity when:
      - vertical gap < gap_ratio × mean line-height of the current group
      - horizontal overlap > min_h_overlap fraction of the incoming anchor width

    Both conditions must hold — this prevents merging horizontally separated
    columns that happen to sit at the same vertical position.
    """

    def __init__(
        self,
        gap_ratio: float = 0.8,
        min_h_overlap: float = 0.2,
    ) -> None:
        self._gap_ratio = gap_ratio
        self._min_h_overlap = min_h_overlap

    def process(self, anchors: list[Anchor]) -> list[EntityGroup]:
        """Cluster *anchors* into EntityGroups and return them top-to-bottom.

        Args:
            anchors: Detected anchors from Stage 5 (any order).

        Returns:
            List of EntityGroup objects, each spanning one Semantic Entity.
            Empty list when *anchors* is empty.
        """
        if not anchors:
            return []

        # Sort top-to-bottom by vertical centre
        sorted_anchors = sorted(anchors, key=lambda a: (a.bbox[1] + a.bbox[3]) / 2.0)

        groups: list[list[Anchor]] = [[sorted_anchors[0]]]

        for anchor in sorted_anchors[1:]:
            x1, y1, x2, y2 = anchor.bbox.tolist()
            current = groups[-1]

            # Current group's merged extents
            gx1 = min(a.bbox[0] for a in current)
            gy2 = max(a.bbox[3] for a in current)
            gx2 = max(a.bbox[2] for a in current)

            # Mean line height of current group
            mean_h = np.mean([a.bbox[3] - a.bbox[1] for a in current])

            vertical_gap = y1 - gy2

            h_overlap = max(0, min(x2, gx2) - max(x1, gx1))
            anchor_width = x2 - x1 + 1e-6
            h_overlap_ratio = h_overlap / anchor_width

            if (
                vertical_gap < self._gap_ratio * mean_h
                and h_overlap_ratio > self._min_h_overlap
            ):
                current.append(anchor)
            else:
                groups.append([anchor])

        return [_make_group(g) for g in groups]


def _make_group(anchors: list[Anchor]) -> EntityGroup:
    bboxes = np.stack([a.bbox for a in anchors])
    merged = np.array(
        [bboxes[:, 0].min(), bboxes[:, 1].min(), bboxes[:, 2].max(), bboxes[:, 3].max()],
        dtype=np.int32,
    )
    return EntityGroup(
        anchors=anchors,
        bbox=merged,
        confidence=float(max(a.confidence for a in anchors)),
    )
