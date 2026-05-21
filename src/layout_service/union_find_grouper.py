import numpy as np

from .grouper import AnchorGrouper, Block
from .text_line_detector import Anchor, UnionFind


class UnionFindGrouper(AnchorGrouper):
    """
    Hierarchical Union-Find grouping strategy with deterministic
    anti-engulfment constraints via two-pass layout simulation.
    """

    def __init__(self, iou_threshold: float = 0.02):
        self.iou_threshold = iou_threshold

    def _boxes_intersect(self, boxA: np.ndarray, boxB: np.ndarray) -> bool:
        """Determines if two bounding boxes overlap or intersect even in part."""
        return not (
            boxA[2] <= boxB[0]  # A is completely left of B
            or boxA[0] >= boxB[2]  # A is completely right of B
            or boxA[3] <= boxB[1]  # A is completely above B
            or boxA[1] >= boxB[3]  # A is completely below B
        )

    def group(self, anchors: list[Anchor]) -> list[Block]:
        if not anchors:
            return []

        n = len(anchors)
        heights = [a.bbox[3] - a.bbox[1] for a in anchors]
        median_height = np.median(heights) if heights else 20.0
        v_expand = median_height * 0.65
        h_expand = median_height * 0.25

        # ==========================================
        # PASS 1: Generate Raw Initial Block States
        # ==========================================
        sim_uf = UnionFind(n)
        for i in range(n):
            for j in range(i + 1, n):
                if self._should_merge(anchors[i], anchors[j], v_expand, h_expand):
                    sim_uf.union(i, j)

        # Build map of root IDs to anchor lists
        initial_sets: dict[int, list[Anchor]] = {}
        for i in range(n):
            root = sim_uf.find(i)
            initial_sets.setdefault(root, []).append(anchors[i])

        # Convert sets into temporary Block structures with concrete bounding boxes
        temp_blocks: list[dict] = []
        for root_id, constituent_anchors in initial_sets.items():
            bbox = self.compute_macro_bbox(constituent_anchors)
            temp_blocks.append(
                {"root_id": root_id, "bbox": bbox, "anchors": constituent_anchors}
            )

        # ==========================================
        # PASS 2: Deterministic Overlap Validation
        # ==========================================
        final_uf = UnionFind(n)
        num_blocks = len(temp_blocks)

        for i in range(num_blocks):
            blockA = temp_blocks[i]
            for j in range(i + 1, num_blocks):
                blockB = temp_blocks[j]

                # Verify if any underlying anchors between these two blocks actually want to merge
                wants_merge = False
                for a_i in blockA["anchors"]:
                    for a_j in blockB["anchors"]:
                        if self._should_merge(a_i, a_j, v_expand, h_expand):
                            wants_merge = True
                            break
                    if wants_merge:
                        break

                if not wants_merge:
                    continue

                # Compute the exact hypothetical bounding box if these two blocks merge
                hypothetical_x1 = min(blockA["bbox"][0], blockB["bbox"][0])
                hypothetical_y1 = min(blockA["bbox"][1], blockB["bbox"][1])
                hypothetical_x2 = max(blockA["bbox"][2], blockB["bbox"][2])
                hypothetical_y2 = max(blockA["bbox"][3], blockB["bbox"][3])
                hypothetical_box = np.array(
                    [
                        hypothetical_x1,
                        hypothetical_y1,
                        hypothetical_x2,
                        hypothetical_y2,
                    ],
                    dtype=np.int32,
                )

                # Check if this expanded box engulfs or intersects ANY other block in the scene
                engulfment_detected = False
                for k in range(num_blocks):
                    if k == i or k == j:
                        continue

                    target_block = temp_blocks[k]
                    if self._boxes_intersect(hypothetical_box, target_block["bbox"]):
                        engulfment_detected = True
                        break

                # Deterministic Guard Action
                if engulfment_detected:
                    # Reject the union! This merge would swallow an independent text block.
                    continue

                # If no other blocks are intercepted, the merge is safe. Execute across all constituent anchors.
                for a_i in blockA["anchors"]:
                    idx_i = anchors.index(a_i)
                    for a_j in blockB["anchors"]:
                        idx_j = anchors.index(a_j)
                        if self._should_merge(a_i, a_j, v_expand, h_expand):
                            final_uf.union(idx_i, idx_j)

        # ==========================================
        # PASS 3: Construct Output Primitives
        # ==========================================
        final_sets: dict[int, list[Anchor]] = {}
        for i in range(n):
            root = final_uf.find(i)
            final_sets.setdefault(root, []).append(anchors[i])

        blocks = []
        for constituent_anchors in final_sets.values():
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
