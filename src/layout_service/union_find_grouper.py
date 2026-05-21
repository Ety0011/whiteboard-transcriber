import numpy as np

from .grouper import EntityGroup, AnchorGrouper
from .text_line_detector import Anchor, UnionFind


class UnionFindGrouper(AnchorGrouper):
    """
    Hierarchical Union-Find grouping strategy.
    Adaptive spatial thresholds scaled to median line height.
    Multi-column gutter guard prevents cross-column merges.
    """

    def __init__(self, iou_threshold: float = 0.02):
        self.iou_threshold = iou_threshold

    def group(self, anchors: list[Anchor]) -> list[EntityGroup]:
        if not anchors:
            return []

        heights = [a.bbox[3] - a.bbox[1] for a in anchors]
        median_height = np.median(heights) if heights else 20.0
        v_expand = median_height * 0.65
        h_expand = median_height * 0.25

        n = len(anchors)
        uf = UnionFind(n)
        for i in range(n):
            for j in range(i + 1, n):
                if self._should_merge(anchors[i], anchors[j], v_expand, h_expand):
                    uf.union(i, j)

        sets: dict[int, list[Anchor]] = {}
        for i in range(n):
            root = uf.find(i)
            sets.setdefault(root, []).append(anchors[i])

        output_groups = []
        for constituent_anchors in sets.values():
            macro_box = self.compute_macro_bbox(constituent_anchors)
            max_conf = max(a.confidence for a in constituent_anchors)
            output_groups.append(
                EntityGroup(anchors=constituent_anchors, bbox=macro_box, confidence=max_conf)
            )

        return output_groups

    def _should_merge(self, a: Anchor, b: Anchor, v_expand: float, h_expand: float) -> bool:
        ax1, ay1, ax2, ay2 = a.bbox.tolist()
        bx1, by1, bx2, by2 = b.bbox.tolist()

        gap_x = max(0, bx1 - ax2, ax1 - bx2)
        if gap_x > max(ax2 - ax1, bx2 - bx1) * 0.35:
            return False

        ax1e, ax2e = ax1 - h_expand, ax2 + h_expand
        ay1e, ay2e = ay1 - v_expand, ay2 + v_expand
        bx1e, bx2e = bx1 - h_expand, bx2 + h_expand
        by1e, by2e = by1 - v_expand, by2 + v_expand

        ix1, iy1 = max(ax1e, bx1e), max(ay1e, by1e)
        ix2, iy2 = min(ax2e, bx2e), min(ay2e, by2e)

        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter == 0.0:
            return False

        area_a = (ax2e - ax1e) * (ay2e - ay1e)
        area_b = (bx2e - bx1e) * (by2e - by1e)
        return inter / (area_a + area_b - inter) > self.iou_threshold
