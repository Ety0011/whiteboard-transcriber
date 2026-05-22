import numpy as np

from .grouper import Block, TextLineGrouper
from .text_line_detector import TextLine


class XYCutGrouper(TextLineGrouper):
    """
    Classical top-down document layout structure parser.
    Projects spatial distributions horizontally and vertically to detect white space
    valleys, recursively slicing the coordinates into multi-level block hierarchies.
    """

    def __init__(self, x_threshold: int = 45, y_threshold: int = 20):
        self.x_threshold = x_threshold
        self.y_threshold = y_threshold

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
            sorted_meta = sorted(
                zip(indices, current_boxes), key=lambda x: x[1][0]
            )
            sorted_indices = [x[0] for x in sorted_meta]

            partitions = []
            curr_partition = [sorted_indices[0]]
            max_edge = sorted_meta[0][1][2]

            for idx, box in sorted_meta[1:]:
                if box[0] - max_edge > self.x_threshold:
                    partitions.append(curr_partition)
                    curr_partition = [idx]
                else:
                    curr_partition.append(idx)
                max_edge = max(max_edge, box[2])
            partitions.append(curr_partition)

            if len(partitions) == 1:
                return self._recursive_split(indices, bboxes, split_horizontal=False)

        else:
            sorted_meta = sorted(
                zip(indices, current_boxes), key=lambda x: x[1][1]
            )
            sorted_indices = [x[0] for x in sorted_meta]

            partitions = []
            curr_partition = [sorted_indices[0]]
            max_edge = sorted_meta[0][1][3]

            for idx, box in sorted_meta[1:]:
                if box[1] - max_edge > self.y_threshold:
                    partitions.append(curr_partition)
                    curr_partition = [idx]
                else:
                    curr_partition.append(idx)
                max_edge = max(max_edge, box[3])
            partitions.append(curr_partition)

            if len(partitions) == 1:
                return [indices]

        final_leaves = []
        for part in partitions:
            final_leaves.extend(
                self._recursive_split(part, bboxes, not split_horizontal)
            )
        return final_leaves

    def group(self, lines: list[TextLine]) -> list[Block]:
        if not lines:
            return []

        bboxes = np.array([line.bbox for line in lines])
        initial_indices = list(range(len(lines)))

        grouped_index_leaves = self._recursive_split(
            initial_indices, bboxes, split_horizontal=True
        )

        blocks = []
        for leaf_indices in grouped_index_leaves:
            constituent_lines = [lines[idx] for idx in leaf_indices]
            bbox = self.compute_bbox(constituent_lines)
            max_conf = max(line.confidence for line in constituent_lines)
            blocks.append(Block(bbox=bbox, confidence=max_conf, lines=constituent_lines))

        return blocks
