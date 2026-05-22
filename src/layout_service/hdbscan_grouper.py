import numpy as np

from .grouper import Block, TextLineGrouper
from .text_line_detector import TextLine


class HDBSCANGrouper(TextLineGrouper):
    """
    Anisotropic spatial density clusterer utilizing HDBSCAN.
    Penalizes vertical distances tightly while allowing wide horizontal tracking
    scaled dynamically against localized line heights.
    """

    def __init__(self, min_cluster_size: int = 2, horizontal_scale: float = 3.5):
        self.min_cluster_size = min_cluster_size
        self.horizontal_scale = horizontal_scale

    def _custom_pairwise_distance(self, centroids: np.ndarray) -> np.ndarray:
        """Scale-invariant anisotropic distance matrix. Input shape: (N, 3) [cx, cy, h]."""
        cx = centroids[:, 0:1]
        cy = centroids[:, 1:2]
        h = centroids[:, 2:3]
        dx = np.abs(cx - cx.T)
        dy = np.abs(cy - cy.T)
        scale = np.maximum((h + h.T) / 2.0, 1e-5)
        norm_dx = dx / (self.horizontal_scale * scale)
        norm_dy = dy / scale
        return np.sqrt(norm_dx**2 + norm_dy**2).astype(np.float32)

    def group(self, lines: list[TextLine]) -> list[Block]:
        if not lines:
            return []
        if len(lines) == 1:
            return [Block(bbox=lines[0].bbox, confidence=lines[0].confidence, lines=lines)]

        from hdbscan import HDBSCAN

        centroids = []
        for line in lines:
            cx = (line.bbox[0] + line.bbox[2]) / 2.0
            cy = (line.bbox[1] + line.bbox[3]) / 2.0
            h = line.bbox[3] - line.bbox[1]
            centroids.append([cx, cy, h])

        X = np.array(centroids, dtype=np.float32)
        dist_matrix = self._custom_pairwise_distance(X)

        clusterer = HDBSCAN(
            metric="precomputed",
            min_cluster_size=self.min_cluster_size,
            min_samples=1,
        )
        labels = clusterer.fit_predict(dist_matrix.astype(np.float64))

        groups_dict = {}
        for idx, label in enumerate(labels):
            if label == -1:
                groups_dict[f"noise_{idx}"] = [lines[idx]]
            else:
                if label not in groups_dict:
                    groups_dict[label] = []
                groups_dict[label].append(lines[idx])

        blocks = []
        for constituent_lines in groups_dict.values():
            bbox = self.compute_bbox(constituent_lines)
            max_conf = max(line.confidence for line in constituent_lines)
            blocks.append(Block(bbox=bbox, confidence=max_conf, lines=constituent_lines))

        return blocks
