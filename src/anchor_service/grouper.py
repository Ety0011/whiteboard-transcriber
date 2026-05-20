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
        # px to extend each line up/down before IoU test — bridges inter-line gaps
        vertical_expand_px: float = 20.0,
        # px to extend each line left/right before IoU test
        horizontal_expand_px: float = 0.0,
        # min IoU to merge two (expanded) anchors
        iou_threshold: float = 0.02,
    ) -> None:
        self._vertical_expand_px = vertical_expand_px
        self._horizontal_expand_px = horizontal_expand_px
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

        num_anchors = len(anchors)
        union_find = UnionFind(num_anchors)

        # Pairwise comparison to find connected components
        for i in range(num_anchors):
            for j in range(i + 1, num_anchors):
                if self._should_merge(anchors[i], anchors[j]):
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

    def _should_merge(self, a: Anchor, b: Anchor) -> bool:
        """Return True if two anchors should be merged into the same semantic group."""
        ax1, ay1, ax2, ay2 = a.bbox.tolist()
        bx1, by1, bx2, by2 = b.bbox.tolist()

        ax1e = ax1 - self._horizontal_expand_px
        ax2e = ax2 + self._horizontal_expand_px
        ay1e = ay1 - self._vertical_expand_px
        ay2e = ay2 + self._vertical_expand_px
        bx1e = bx1 - self._horizontal_expand_px
        bx2e = bx2 + self._horizontal_expand_px
        by1e = by1 - self._vertical_expand_px
        by2e = by2 + self._vertical_expand_px

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


# ---------------------------------------------------------------------------
# SOTA Visual Masking Helper
# ---------------------------------------------------------------------------


def get_masked_crop(
    group: EntityGroup,
    full_image: np.ndarray,
    bg_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Create a crop containing ONLY the pixels of this group's constituent anchors.

    Everything else inside the group's rectangular bounding box is guaranteed
    to be pure whiteboard background. No external anchors list is needed.
    """
    gx1, gy1, gx2, gy2 = group.bbox.tolist()
    h, w = full_image.shape[:2]

    # 1. Create a blank, pure-white canvas of the exact group bounding box size
    crop_h = max(1, int(gy2 - gy1))
    crop_w = max(1, int(gx2 - gx1))
    crop = np.full((crop_h, crop_w, 3), bg_color, dtype=np.uint8)

    # 2. Copy ONLY our own group's anchors onto the canvas
    for anchor in group.anchors:
        ax1, ay1, ax2, ay2 = anchor.bbox.tolist()

        # Clamp boundaries to the physical image limits
        ax1_c = max(0, min(w, int(ax1)))
        ay1_c = max(0, min(h, int(ay1)))
        ax2_c = max(0, min(w, int(ax2)))
        ay2_c = max(0, min(h, int(ay2)))

        if ax2_c <= ax1_c or ay2_c <= ay1_c:
            continue

        # Extract the precise ink pixels of this line anchor
        ink_slice = full_image[ay1_c:ay2_c, ax1_c:ax2_c]

        # Calculate relative coordinates on the crop canvas
        lx1 = int(ax1_c - gx1)
        ly1 = int(ay1_c - gy1)
        lx2 = lx1 + ink_slice.shape[1]
        ly2 = ly1 + ink_slice.shape[0]

        # Copy the ink directly onto the clean white canvas
        crop[ly1:ly2, lx1:lx2] = ink_slice

    return crop
