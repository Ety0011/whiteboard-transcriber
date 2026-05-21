import numpy as np

from .grouper import Block, AnchorGrouper
from .text_line_detector import Anchor


class XYCutGrouper(AnchorGrouper):
    """
    Classical top-down document layout structure parser.
    Projects spatial distributions horizontally and vertically to detect white space
    valleys, recursively slicing the coordinates into multi-level block hierarchies.
    """

    def __init__(self, x_threshold: int = 45, y_threshold: int = 20):
        self.x_threshold = x_threshold  # Minimum whitespace gap to separate columns
        self.y_threshold = y_threshold  # Minimum whitespace gap to separate paragraphs

    def _recursive_split(
        self,
        indices: list[int],
        bboxes: np.ndarray,
        split_horizontal: bool,
    ) -> list[list[int]]:
        """Recursively parses indices into partitioned lists based on profile gaps."""
        if len(indices) <= 1:
            return [indices]

        current_boxes = bboxes[indices]

        if split_horizontal:
            # Sort by x1 to analyze column projection profiles
            sorted_meta = sorted(
                zip(indices, current_boxes), key=lambda x: x[1][0]
            )
            sorted_indices = [x[0] for x in sorted_meta]

            partitions = []
            curr_partition = [sorted_indices[0]]
            max_edge = sorted_meta[0][1][2]  # tracked x2 max boundary

            for idx, box in sorted_meta[1:]:
                if box[0] - max_edge > self.x_threshold:
                    partitions.append(curr_partition)
                    curr_partition = [idx]
                else:
                    curr_partition.append(idx)
                max_edge = max(max_edge, box[2])
            partitions.append(curr_partition)

            # No horizontal split found → attempt vertical
            if len(partitions) == 1:
                return self._recursive_split(indices, bboxes, split_horizontal=False)

        else:
            # Sort by y1 to analyze paragraph row profiles
            sorted_meta = sorted(
                zip(indices, current_boxes), key=lambda x: x[1][1]
            )
            sorted_indices = [x[0] for x in sorted_meta]

            partitions = []
            curr_partition = [sorted_indices[0]]
            max_edge = sorted_meta[0][1][3]  # tracked y2 max boundary

            for idx, box in sorted_meta[1:]:
                if box[1] - max_edge > self.y_threshold:
                    partitions.append(curr_partition)
                    curr_partition = [idx]
                else:
                    curr_partition.append(idx)
                max_edge = max(max_edge, box[3])
            partitions.append(curr_partition)

            # No vertical split found → tree leaf reached
            if len(partitions) == 1:
                return [indices]

        # Recurse down the tree alternating orientation axis
        final_leaves = []
        for part in partitions:
            final_leaves.extend(
                self._recursive_split(part, bboxes, not split_horizontal)
            )
        return final_leaves

    def group(self, anchors: list[Anchor]) -> list[Block]:
        if not anchors:
            return []

        bboxes = np.array([a.bbox for a in anchors])
        initial_indices = list(range(len(anchors)))

        # Start from X axis to detect column zones first
        grouped_index_leaves = self._recursive_split(
            initial_indices, bboxes, split_horizontal=True
        )

        blocks = []
        for leaf_indices in grouped_index_leaves:
            constituent_anchors = [anchors[idx] for idx in leaf_indices]
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
