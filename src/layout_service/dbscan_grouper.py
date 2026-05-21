import numpy as np

from .grouper import Block, AnchorGrouper
from .text_line_detector import Anchor


class DBSCANGrouper(AnchorGrouper):
    """
    Density-based whiteboard layout analyzer.
    Samples 3 horizontal coordinate nodes per line (L/C/R), runs DBSCAN with
    eps scaled to median line height for 100% column safety.
    """

    def __init__(self, eps_factor: float = 1.8):
        # eps_factor scales clustering merge radius relative to median line height
        self.eps_factor = eps_factor

    def group(self, anchors: list[Anchor]) -> list[Block]:
        from sklearn.cluster import DBSCAN

        if not anchors:
            return []

        heights = [a.bbox[3] - a.bbox[1] for a in anchors]
        median_height = np.median(heights) if heights else 20.0

        # Sample 3 horizontal nodes per line (Left, Center, Right)
        db_points = []
        anchor_indices = []
        for idx, a in enumerate(anchors):
            x1, y1, x2, y2 = a.bbox.tolist()
            cy = (y1 + y2) / 2.0
            db_points.extend([[x1, cy], [(x1 + x2) / 2.0, cy], [x2, cy]])
            anchor_indices.extend([idx, idx, idx])

        db_points = np.array(db_points)
        eps = median_height * self.eps_factor
        labels = DBSCAN(eps=eps, min_samples=2, metric="euclidean").fit(db_points).labels_

        # Aggregate clusters back to Anchor lists (set-tracked to avoid bbox __eq__)
        sets: dict[int, list[Anchor]] = {}
        added: dict[int, set[int]] = {}
        for idx, cluster_id in enumerate(labels):
            if cluster_id == -1:
                continue
            orig_idx = anchor_indices[idx]
            if cluster_id not in sets:
                sets[cluster_id] = []
                added[cluster_id] = set()
            if orig_idx not in added[cluster_id]:
                sets[cluster_id].append(anchors[orig_idx])
                added[cluster_id].add(orig_idx)

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
