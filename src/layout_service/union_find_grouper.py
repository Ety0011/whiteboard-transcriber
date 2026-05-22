import numpy as np

from .grouper import TextLineGrouper, Block
from .text_line_detector import TextLine, UnionFind


class UnionFindGrouper(TextLineGrouper):
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

    def group(self, lines: list[TextLine]) -> list[Block]:
        if not lines:
            return []

        n = len(lines)

        sorted_indices = sorted(range(n), key=lambda idx: lines[idx].bbox[1])

        heights = [line.bbox[3] - line.bbox[1] for line in lines]
        median_height = np.median(heights) if heights else 20.0

        v_expand = median_height * self.v_expand_ratio
        h_expand = median_height * self.h_expand_ratio

        uf = UnionFind(n)

        for i_idx, i in enumerate(sorted_indices):
            line_i = lines[i]
            ax1, ay1, ax2, ay2 = line_i.bbox
            w_i = ax2 - ax1

            for j in sorted_indices[i_idx + 1:]:
                line_j = lines[j]
                bx1, by1, bx2, by2 = line_j.bbox

                if by1 - ay2 > v_expand * 2.0:
                    break

                w_j = bx2 - bx1

                if max(w_i, w_j) > min(w_i, w_j) * self.max_width_ratio:
                    has_horiz_overlap = max(0.0, min(ax2, bx2) - max(ax1, bx1)) > 0
                    if has_horiz_overlap:
                        continue

                if self._should_merge(line_i, line_j, v_expand, h_expand):
                    uf.union(i, j)

        sets: dict[int, list[TextLine]] = {}
        for i in range(n):
            root = uf.find(i)
            sets.setdefault(root, []).append(lines[i])

        blocks = []
        for constituent_lines in sets.values():
            bbox = self.compute_bbox(constituent_lines)
            max_conf = max(line.confidence for line in constituent_lines)
            blocks.append(Block(bbox=bbox, confidence=max_conf, lines=constituent_lines))

        return blocks

    def _should_merge(
        self, a: TextLine, b: TextLine, v_expand: float, h_expand: float
    ) -> bool:
        ax1, ay1, ax2, ay2 = a.bbox
        bx1, by1, bx2, by2 = b.bbox

        gap_x = max(0.0, bx1 - ax2, ax1 - bx2)
        if gap_x > h_expand:
            return False

        ax1e, ax2e = ax1 - h_expand, ax2 + h_expand
        ay1e, ay2e = ay1 - v_expand, ay2 + v_expand
        bx1e, bx2e = bx1 - h_expand, bx2 + h_expand
        by1e, by2e = by1 - v_expand, by2 + v_expand

        ix1, iy1 = max(ax1e, bx1e), max(ay1e, by1e)
        ix2, iy2 = min(ax2e, bx2e), min(ay2e, by2e)

        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        return inter > 0.0
