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
        """
        Calculates a scale-invariant, anisotropic distance matrix.
        X shape: (N, 3) -> [cx, cy, line_height]
        """
        n_samples = centroids.shape[0]
        dist_matrix = np.zeros((n_samples, n_samples), dtype=np.float32)

        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dx = abs(centroids[i, 0] - centroids[j, 0])
                dy = abs(centroids[i, 1] - centroids[j, 1])
                avg_h = (centroids[i, 2] + centroids[j, 2]) / 2.0

                scale = max(avg_h, 1e-5)

                norm_dx = dx / (self.horizontal_scale * scale)
                norm_dy = dy / scale

                dist = np.sqrt(norm_dx**2 + norm_dy**2)
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist

        return dist_matrix

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
