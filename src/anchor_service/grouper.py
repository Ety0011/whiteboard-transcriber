"""Stage 6 — Hierarchical Entity Grouper.

Clusters line-level Spatial Anchors from Stage 5 into Semantic Entities using
pairwise Union-Find clustering to robustly handle multi-column layouts.

Includes a visual masking utility to white-out any unrelated text that falls
inside a group's expanded axis-aligned bounding box.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from anchor_service.detector import Anchor


@dataclass
class EntityGroup:
    anchors: list[Anchor]  # constituent anchors
    bbox: np.ndarray  # (4,) int32: min(x1), min(y1), max(x2), max(y2)
    confidence: float  # max confidence across constituent anchors


class UnionFind:
    """Lightweight Disjoint-Set Forest for pairwise clustering."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i: int, j: int) -> bool:
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j
            return True
        return False


class EntityGrouper:
    """Stateless spatial proximity grouper.

    Clusters line anchors based on pairwise vertical adjacency and horizontal overlap.
    """

    def __init__(
        self,
        # fraction of median line height to expand up/down — bridges inter-line gaps
        vertical_expand_ratio: float = 0.5,
        # fraction of median line height to expand left/right
        horizontal_expand_ratio: float = 0.0,
        # min IoU to merge two (expanded) anchors
        iou_threshold: float = 0.02,
    ) -> None:
        self._vertical_expand_ratio = vertical_expand_ratio
        self._horizontal_expand_ratio = horizontal_expand_ratio
        self._iou_threshold = iou_threshold

    def group(self, anchors: list[Anchor]) -> list[EntityGroup]:
        """Cluster *anchors* into EntityGroups using Union-Find clustering.

        Args:
            anchors: Detected anchors from Stage 5 (any order).

        Returns:
            List of EntityGroup objects, sorted top-to-bottom.
        """
        if not anchors:
            return []

        heights = [a.bbox[3] - a.bbox[1] for a in anchors]
        median_h = float(np.median(heights)) if heights else 20.0
        v_expand = median_h * self._vertical_expand_ratio
        h_expand = median_h * self._horizontal_expand_ratio

        num_anchors = len(anchors)
        union_find = UnionFind(num_anchors)

        # Pairwise comparison to find connected components
        for i in range(num_anchors):
            for j in range(i + 1, num_anchors):
                if self._should_merge(anchors[i], anchors[j], v_expand, h_expand):
                    union_find.union(i, j)

        # Assemble the disjoint sets
        sets: dict[int, list[Anchor]] = {}
        for i in range(num_anchors):
            root = union_find.find(i)
            if root not in sets:
                sets[root] = []
            sets[root].append(anchors[i])

        # Build EntityGroups
        groups = []
        for group_anchors in sets.values():
            groups.append(self._make_group(group_anchors))

        # Sort groups top-to-bottom by vertical centre
        return sorted(groups, key=lambda g: (g.bbox[1] + g.bbox[3]) / 2.0)

    def _should_merge(
        self, a: Anchor, b: Anchor, v_expand: float, h_expand: float
    ) -> bool:
        """Return True if two anchors should be merged into the same semantic group."""
        ax1, ay1, ax2, ay2 = a.bbox.tolist()
        bx1, by1, bx2, by2 = b.bbox.tolist()

        # Bail if horizontal gap exceeds 35% of the wider line — different columns.
        gap_x = max(0, bx1 - ax2, ax1 - bx2)
        if gap_x > max(ax2 - ax1, bx2 - bx1) * 0.35:
            return False

        ax1e = ax1 - h_expand
        ax2e = ax2 + h_expand
        ay1e = ay1 - v_expand
        ay2e = ay2 + v_expand
        bx1e = bx1 - h_expand
        bx2e = bx2 + h_expand
        by1e = by1 - v_expand
        by2e = by2 + v_expand

        ix1, iy1 = max(ax1e, bx1e), max(ay1e, by1e)
        ix2, iy2 = min(ax2e, bx2e), min(ay2e, by2e)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter == 0.0:
            return False
        area_a = (ax2e - ax1e) * (ay2e - ay1e)
        area_b = (bx2e - bx1e) * (by2e - by1e)
        return inter / (area_a + area_b - inter) > self._iou_threshold

    def _make_group(self, group_anchors: list[Anchor]) -> EntityGroup:
        bboxes = np.stack([a.bbox for a in group_anchors])
        merged = np.array(
            [
                bboxes[:, 0].min(),
                bboxes[:, 1].min(),
                bboxes[:, 2].max(),
                bboxes[:, 3].max(),
            ],
            dtype=np.int32,
        )
        return EntityGroup(
            anchors=group_anchors,
            bbox=merged,
            confidence=float(max(a.confidence for a in group_anchors)),
        )
