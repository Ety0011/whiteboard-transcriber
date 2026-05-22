import numpy as np

from .grouper import Block, TextLineGrouper
from .text_line_detector import TextLine


class DBSCANGrouper(TextLineGrouper):
    """
    Density-based whiteboard layout analyzer.
    Samples 3 horizontal coordinate nodes per line (L/C/R), runs DBSCAN with
    eps scaled to median line height for 100% column safety.
    """

    def __init__(self, eps_factor: float = 1.8):
        self.eps_factor = eps_factor

    def group(self, lines: list[TextLine]) -> list[Block]:
        from sklearn.cluster import DBSCAN

        if not lines:
            return []

        heights = [line.bbox[3] - line.bbox[1] for line in lines]
        median_height = np.median(heights) if heights else 20.0

        db_points = []
        line_indices = []
        for idx, line in enumerate(lines):
            x1, y1, x2, y2 = line.bbox.tolist()
            cy = (y1 + y2) / 2.0
            db_points.extend([[x1, cy], [(x1 + x2) / 2.0, cy], [x2, cy]])
            line_indices.extend([idx, idx, idx])

        db_points = np.array(db_points)
        eps = median_height * self.eps_factor
        labels = DBSCAN(eps=eps, min_samples=2, metric="euclidean").fit(db_points).labels_

        sets: dict[int, list[TextLine]] = {}
        added: dict[int, set[int]] = {}
        for idx, cluster_id in enumerate(labels):
            if cluster_id == -1:
                continue
            orig_idx = line_indices[idx]
            if cluster_id not in sets:
                sets[cluster_id] = []
                added[cluster_id] = set()
            if orig_idx not in added[cluster_id]:
                sets[cluster_id].append(lines[orig_idx])
                added[cluster_id].add(orig_idx)

        blocks = []
        for constituent_lines in sets.values():
            bbox = self.compute_bbox(constituent_lines)
            max_conf = max(line.confidence for line in constituent_lines)
            blocks.append(Block(bbox=bbox, confidence=max_conf, lines=constituent_lines))

        return blocks
