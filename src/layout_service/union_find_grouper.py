import numpy as np

from .grouper import AnchorGrouper, Block
from .text_line_detector import Anchor, UnionFind


class UnionFindGrouper(AnchorGrouper):
    """
    Hierarchical Union-Find grouping strategy based on standard IoU.
    Uses asymmetric, tunable dilation to bridge corner connections.
    """

    def __init__(
        self,
        v_expand_ratio: float = 0.5,
        h_expand_ratio: float = 0.0,
        max_width_ratio: float = 1.5,
    ):
        """
        Args:
            v_expand_ratio: Vertical dilation multiplier scaled to median line height.
            h_expand_ratio: Horizontal dilation multiplier scaled to median line height.
            max_width_ratio: Factor to isolate column-poisoning intruder lines.
        """
        self.max_width_ratio = max_width_ratio
        self.v_expand_ratio = v_expand_ratio
        self.h_expand_ratio = h_expand_ratio

    def group(self, anchors: list[Anchor]) -> list[Block]:
        if not anchors:
            return []

        n = len(anchors)

        # 1. Sort anchors vertically by top coordinate for sweep-line optimization
        sorted_indices = sorted(range(n), key=lambda idx: anchors[idx].bbox[1])

        heights = [a.bbox[3] - a.bbox[1] for a in anchors]
        median_height = np.median(heights) if heights else 20.0

        # Dynamic expansion based on exposed tuning ratios
        v_expand = median_height * self.v_expand_ratio
        h_expand = median_height * self.h_expand_ratio

        uf = UnionFind(n)

        # 2. Optimized sweep-line evaluation
        for i_idx, i in enumerate(sorted_indices):
            anchor_i = anchors[i]
            ax1, ay1, ax2, ay2 = anchor_i.bbox
            w_i = ax2 - ax1

            for j in sorted_indices[i_idx + 1 :]:
                anchor_j = anchors[j]
                bx1, by1, bx2, by2 = anchor_j.bbox

                # Performance Break: stop looking when lines are too far down
                if by1 - ay2 > v_expand * 2.0:
                    break

                w_j = bx2 - bx1

                # Anti-Engulfment Guard to preserve columns
                if max(w_i, w_j) > min(w_i, w_j) * self.max_width_ratio:
                    has_horiz_overlap = max(0.0, min(ax2, bx2) - max(ax1, bx1)) > 0
                    if has_horiz_overlap:
                        continue

                # Standard IoU check
                if self._should_merge(anchor_i, anchor_j, v_expand, h_expand):
                    uf.union(i, j)

        # 3. Aggregate clusters
        sets: dict[int, list[Anchor]] = {}
        for i in range(n):
            root = uf.find(i)
            sets.setdefault(root, []).append(anchors[i])

        blocks = []
        for constituent_anchors in sets.values():
            macro_box = self.compute_macro_bbox(constituent_anchors)
            macro_poly = self.compute_macro_poly(constituent_anchors)
            max_conf = max(a.confidence for a in constituent_anchors)
            blocks.append(
                Block(
                    poly=macro_poly,
                    bbox=macro_box,
                    label="TEXT",
                    confidence=max_conf,
                    anchors=constituent_anchors,
                )
            )

        return blocks

    def _should_merge(
        self, a: Anchor, b: Anchor, v_expand: float, h_expand: float
    ) -> bool:
        ax1, ay1, ax2, ay2 = a.bbox
        bx1, by1, bx2, by2 = b.bbox

        # Enforce column strictness: require some baseline horizontal closeness
        gap_x = max(0.0, bx1 - ax2, ax1 - bx2)
        if gap_x > h_expand:
            return False

        # Dilate bounding structures
        ax1e, ax2e = ax1 - h_expand, ax2 + h_expand
        ay1e, ay2e = ay1 - v_expand, ay2 + v_expand
        bx1e, bx2e = bx1 - h_expand, bx2 + h_expand
        by1e, by2e = by1 - v_expand, by2 + v_expand

        # Calculate intersection
        ix1, iy1 = max(ax1e, bx1e), max(ay1e, by1e)
        ix2, iy2 = min(ax2e, bx2e), min(ay2e, by2e)

        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        return inter > 0.0
