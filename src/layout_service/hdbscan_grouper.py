import numpy as np

from .grouper import Block, AnchorGrouper
from .text_line_detector import Anchor


class HDBSCANGrouper(AnchorGrouper):
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

    def group(self, anchors: list[Anchor]) -> list[Block]:
        if not anchors:
            return []
        if len(anchors) == 1:
            bbox = anchors[0].bbox
            x1, y1, x2, y2 = bbox.tolist()
            poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
            return [
                Block(
                    poly=poly,
                    bbox=bbox,
                    label="TEXT",
                    confidence=anchors[0].confidence,
                    anchors=anchors,
                )
            ]

        from hdbscan import HDBSCAN

        centroids = []
        for a in anchors:
            cx = (a.bbox[0] + a.bbox[2]) / 2.0
            cy = (a.bbox[1] + a.bbox[3]) / 2.0
            h = a.bbox[3] - a.bbox[1]
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
                # Noise → independent structural entity
                groups_dict[f"noise_{idx}"] = [anchors[idx]]
            else:
                if label not in groups_dict:
                    groups_dict[label] = []
                groups_dict[label].append(anchors[idx])

        blocks = []
        for constituent_anchors in groups_dict.values():
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
