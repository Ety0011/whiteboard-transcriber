"""Union-Find text-block clusterer with asymmetric spatial dilation.

Lines are sorted top-to-bottom and compared pairwise.  Expansion radii are
derived from the median line height so the clusterer adapts to font size.
A width-ratio guard prevents wide header/title lines from absorbing narrow
body lines in adjacent columns.
"""

import numpy as np

from .block import Block, TextLineClusterer
from .text_detector import TextLine


class UnionFindClusterer(TextLineClusterer):
    """Union-Find clusterer with asymmetric vertical/horizontal dilation.

    Clusters text lines into blocks by unioning any pair whose expanded
    bounding boxes intersect.  Expansion is anchored to the median line
    height so the clusterer scales with font size.

    Args:
        v_expand_ratio: Vertical dilation as a multiple of median line height.
            Controls how much inter-line whitespace is tolerated within a block.
        h_expand_ratio: Horizontal dilation as a multiple of median line height.
            Set to 0 to disable horizontal bridging (default).
        max_width_ratio: Maximum ratio of widths (wider/narrower) above which
            two horizontally overlapping lines are blocked from merging — prevents
            a full-width title from absorbing narrow column text.
    """

    def __init__(
        self,
        v_expand_ratio: float = 0.5,
        h_expand_ratio: float = 0.0,
        max_width_ratio: float = 1.5,
    ):
        self.max_width_ratio = max_width_ratio
        self.v_expand_ratio = v_expand_ratio
        self.h_expand_ratio = h_expand_ratio

    def group(self, lines: list[TextLine]) -> list[Block]:
        """Cluster *lines* into Blocks using Union-Find over expanded-bbox intersection.

        Args:
            lines: Detected text lines from Stage 5.

        Returns:
            List of Blocks, each grouping spatially adjacent lines.
        """
        if not lines:
            return []

        n = len(lines)

        # Sort by y1 so the inner-loop early-break is valid.
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

            for j in sorted_indices[i_idx + 1 :]:
                line_j = lines[j]
                bx1, by1, bx2, by2 = line_j.bbox

                # Lines are sorted by y1; once the vertical gap exceeds 2×v_expand
                # all subsequent j values are further away and can be skipped.
                if by1 - ay2 > v_expand * 2.0:
                    break

                w_j = bx2 - bx1

                # Block merges between lines of very different widths that share
                # horizontal extent — a full-width header should not absorb body text.
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
            blocks.append(
                Block(bbox=bbox, confidence=max_conf, lines=constituent_lines)
            )

        return blocks

    def _should_merge(
        self, a: TextLine, b: TextLine, v_expand: float, h_expand: float
    ) -> bool:
        """Return True if expanded bboxes of *a* and *b* intersect.

        Each bbox is dilated by (h_expand, v_expand) before the intersection
        test.  A positive h_expand bridges small horizontal gaps; v_expand
        controls the maximum tolerated inter-line whitespace.

        Args:
            a: First text line.
            b: Second text line.
            v_expand: Vertical dilation in pixels.
            h_expand: Horizontal dilation in pixels.

        Returns:
            True if the dilated bboxes overlap.
        """
        ax1, ay1, ax2, ay2 = a.bbox
        bx1, by1, bx2, by2 = b.bbox

        # Reject immediately if raw horizontal gap already exceeds h_expand.
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


class UnionFind:
    """Path-compressed disjoint-set forest for O(α(n)) union and find operations.

    Used by UnionFindClusterer to cluster text lines into blocks without
    maintaining explicit per-cluster membership lists during traversal.
    """

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        """Return the root of the set containing *i*, with path compression."""
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i: int, j: int) -> bool:
        """Merge the sets containing *i* and *j*. Returns True if they were distinct."""
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j
            return True
        return False
