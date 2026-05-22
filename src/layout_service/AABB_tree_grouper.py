import heapq

import numpy as np

from .grouper import Block, TextLineGrouper
from .text_line_detector import TextLine


class AABBNode:
    """Static spatial leaf/branch node for fast query pruning."""

    def __init__(self, block_ref: Block | None, bbox: np.ndarray):
        self.block: Block | None = block_ref
        self.bbox: np.ndarray = bbox.copy()
        self.left: AABBNode | None = None
        self.right: AABBNode | None = None

    def is_leaf(self) -> bool:
        return self.block is not None


class AABBTreeGrouper(TextLineGrouper):
    def __init__(self, max_distance_px: float = 30.0, x_penalty_factor: float = 3.0):
        self.max_distance_px = max_distance_px
        self.x_penalty = x_penalty_factor

    def _intersects(self, box_a: np.ndarray, box_b: np.ndarray) -> bool:
        return not (
            box_a[2] <= box_b[0]
            or box_a[0] >= box_b[2]
            or box_a[3] <= box_b[1]
            or box_a[1] >= box_b[3]
        )

    def _centroid_inside(self, box: np.ndarray, other: np.ndarray) -> bool:
        """True if centroid of other falls inside box."""
        cx = (other[0] + other[2]) / 2.0
        cy = (other[1] + other[3]) / 2.0
        return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]

    def _compute_anisotropic_cost(self, box_a: np.ndarray, box_b: np.ndarray) -> float:
        dx = max(0, box_a[0] - box_b[2], box_b[0] - box_a[2])
        dy = max(0, box_a[1] - box_b[3], box_b[1] - box_a[3])
        adjusted_dx = dx * self.x_penalty
        return float(np.sqrt(adjusted_dx * adjusted_dx + dy * dy))

    def _build_aabb_tree(self, blocks: list[Block]) -> AABBNode | None:
        if not blocks:
            return None
        if len(blocks) == 1:
            return AABBNode(blocks[0], blocks[0].bbox)

        bboxes = np.array([b.bbox for b in blocks])
        axis = 0 if np.var(bboxes[:, 0]) > np.var(bboxes[:, 1]) else 1
        blocks_sorted = sorted(blocks, key=lambda b: b.bbox[axis])

        mid = len(blocks_sorted) // 2
        left_child = self._build_aabb_tree(blocks_sorted[:mid])
        right_child = self._build_aabb_tree(blocks_sorted[mid:])

        parent_box = left_child.bbox.copy()
        parent_box[0] = min(parent_box[0], right_child.bbox[0])
        parent_box[1] = min(parent_box[1], right_child.bbox[1])
        parent_box[2] = max(parent_box[2], right_child.bbox[2])
        parent_box[3] = max(parent_box[3], right_child.bbox[3])

        parent_node = AABBNode(None, parent_box)
        parent_node.left = left_child
        parent_node.right = right_child
        return parent_node

    def _check_engulfment(
        self,
        root: AABBNode | None,
        proposed_box: np.ndarray,
        block_a: Block,
        block_b: Block,
    ) -> bool:
        if root is None:
            return False
        if not self._intersects(proposed_box, root.bbox):
            return False
        if root.is_leaf():
            if root.block is block_a or root.block is block_b:
                return False
            return self._centroid_inside(proposed_box, root.block.bbox)
        return self._check_engulfment(
            root.left, proposed_box, block_a, block_b
        ) or self._check_engulfment(root.right, proposed_box, block_a, block_b)

    def group(self, lines: list[TextLine]) -> list[Block]:
        if not lines:
            return []

        next_id = 0
        block_map: dict[int, Block] = {}
        priority_queue: list = []

        for line in lines:
            block_map[next_id] = Block(
                bbox=line.bbox.copy(), confidence=line.confidence, lines=[line]
            )
            next_id += 1

        active_ids = list(block_map.keys())
        for i in range(len(active_ids)):
            for j in range(i + 1, len(active_ids)):
                id_a, id_b = active_ids[i], active_ids[j]
                cost = self._compute_anisotropic_cost(
                    block_map[id_a].bbox, block_map[id_b].bbox
                )
                if cost <= self.max_distance_px * self.x_penalty:
                    heapq.heappush(priority_queue, (cost, id_a, id_b))

        spatial_tree = self._build_aabb_tree(list(block_map.values()))
        tree_dirty = False

        while priority_queue:
            cost, id_a, id_b = heapq.heappop(priority_queue)

            if id_a not in block_map or id_b not in block_map:
                continue

            if tree_dirty:
                spatial_tree = self._build_aabb_tree(list(block_map.values()))
                tree_dirty = False

            block_a = block_map[id_a]
            block_b = block_map[id_b]
            proposed_lines = block_a.lines + block_b.lines
            proposed_bbox = self.compute_bbox(proposed_lines)

            if self._check_engulfment(spatial_tree, proposed_bbox, block_a, block_b):
                continue

            del block_map[id_b]
            block_a.lines = proposed_lines
            block_a.bbox = proposed_bbox
            block_a.confidence = float(
                np.mean([line.confidence for line in proposed_lines])
            )
            tree_dirty = True

            for check_id, target_block in block_map.items():
                if check_id == id_a:
                    continue
                new_cost = self._compute_anisotropic_cost(
                    block_a.bbox, target_block.bbox
                )
                if new_cost <= self.max_distance_px * self.x_penalty:
                    heapq.heappush(priority_queue, (new_cost, id_a, check_id))

        return list(block_map.values())
