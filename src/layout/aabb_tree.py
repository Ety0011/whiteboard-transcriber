"""Greedy agglomerative text-block grouper backed by an AABB spatial tree.

Pairs of singleton blocks are evaluated in ascending cost order (cheapest merge
first).  A merge is vetoed if the proposed bounding box would engulf an unrelated
third block — detected in O(log N) via the AABB tree.  The tree is rebuilt lazily
after each merge so spatial queries remain accurate.
"""

import heapq

import numpy as np

from .block import Block, TextLineGrouper
from .text_detector import TextLine


class AABBNode:
    """Node in a binary AABB spatial tree.

    Leaf nodes (block is not None) wrap a single Block with a tight bbox.
    Branch nodes (block is None) store the union bbox of their two children
    and are used purely for spatial pruning during tree traversal.
    """

    def __init__(self, block_ref: Block | None, bbox: np.ndarray):
        self.block: Block | None = block_ref
        self.bbox: np.ndarray = bbox.copy()
        self.left: AABBNode | None = None
        self.right: AABBNode | None = None

    def is_leaf(self) -> bool:
        """Return True if this node wraps a Block (leaf), False for branch nodes."""
        return self.block is not None


class AABBTreeGrouper(TextLineGrouper):
    """Greedy agglomerative line grouper using anisotropic cost and AABB engulfment veto.

    Merges are processed cheapest-first via a min-heap.  The anisotropic cost
    penalises horizontal gaps more heavily than vertical gaps, preventing lines
    in adjacent columns from merging.  An engulfment check via the AABB tree
    vetoes any merge whose proposed bbox would absorb an unrelated third block.

    Args:
        max_distance_px: Maximum axis-aligned gap (in the penalised metric) for
            a candidate pair to be considered at all.
        x_penalty_factor: Multiplier applied to horizontal gap before computing
            Euclidean cost.  Values > 1 discourage cross-column merges.
    """

    def __init__(self, max_distance_px: float = 30.0, x_penalty_factor: float = 3.0):
        self.max_distance_px = max_distance_px
        self.x_penalty = x_penalty_factor

    def _intersects(self, box_a: np.ndarray, box_b: np.ndarray) -> bool:
        """Return True if two (x1,y1,x2,y2) bboxes overlap."""
        return not (
            box_a[2] <= box_b[0]
            or box_a[0] >= box_b[2]
            or box_a[3] <= box_b[1]
            or box_a[1] >= box_b[3]
        )

    def _centroid_inside(self, box: np.ndarray, other: np.ndarray) -> bool:
        """Return True if the centroid of *other* falls inside *box*."""
        cx = (other[0] + other[2]) / 2.0
        cy = (other[1] + other[3]) / 2.0
        return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]

    def _compute_anisotropic_cost(self, box_a: np.ndarray, box_b: np.ndarray) -> float:
        """Return merge cost: Euclidean gap with horizontal component scaled up.

        Horizontal gaps are multiplied by x_penalty before computing the
        Euclidean norm, making cross-column merges more expensive than
        same-column vertical merges of the same pixel distance.

        Args:
            box_a: (x1,y1,x2,y2) bbox of first block.
            box_b: (x1,y1,x2,y2) bbox of second block.

        Returns:
            Scalar cost (0.0 if boxes overlap).
        """
        dx = max(0, box_a[0] - box_b[2], box_b[0] - box_a[2])
        dy = max(0, box_a[1] - box_b[3], box_b[1] - box_a[3])
        adjusted_dx = dx * self.x_penalty
        return float(np.sqrt(adjusted_dx * adjusted_dx + dy * dy))

    def _build_aabb_tree(self, blocks: list[Block]) -> AABBNode | None:
        """Build a balanced binary AABB tree over *blocks*.

        Recursively splits on the axis with higher spatial variance (median-axis
        split), producing a balanced tree with O(log N) query depth.  Each
        branch node stores the union bbox of its subtree for fast pruning.

        Args:
            blocks: List of Block objects to index.

        Returns:
            Root AABBNode, or None if *blocks* is empty.
        """
        if not blocks:
            return None
        if len(blocks) == 1:
            return AABBNode(blocks[0], blocks[0].bbox)

        bboxes = np.array([b.bbox for b in blocks])
        # Split on the axis with higher variance for a more balanced tree.
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
        """Return True if merging block_a and block_b would engulf a third block.

        Traverses the AABB tree pruning branches whose bbox does not intersect
        *proposed_box*.  At leaf nodes, checks whether any third-party block's
        centroid falls inside *proposed_box* — if so, the merge would absorb an
        unrelated block and must be vetoed.

        Args:
            root: Current tree node (None terminates traversal).
            proposed_box: The merged bbox that would result from combining
                block_a and block_b.
            block_a: First operand of the candidate merge (excluded from check).
            block_b: Second operand of the candidate merge (excluded from check).

        Returns:
            True if at least one third-party block would be engulfed.
        """
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
        """Group detected text lines into blocks via greedy agglomerative merging.

        Algorithm:
          1. Each line starts as a singleton Block.
          2. All pairs within the cost threshold are pushed to a min-heap.
          3. Pop the cheapest candidate; skip stale entries (consumed blocks).
          4. Rebuild the AABB tree if dirty (invalidated by a previous merge).
          5. Veto the merge if the proposed bbox would engulf a third block.
          6. On merge: absorb id_b into id_a, mark tree dirty, enqueue new
             candidates between id_a and all remaining blocks.

        Args:
            lines: Detected text lines from Stage 5.

        Returns:
            List of Blocks, each grouping one or more spatially adjacent lines.
        """
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

            # Skip stale entries whose blocks were consumed by an earlier merge.
            if id_a not in block_map or id_b not in block_map:
                continue

            # Defer tree rebuild until the next merge attempt needs a valid tree.
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
