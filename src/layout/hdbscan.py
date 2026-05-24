"""HDBSCAN-based text-block clusterer with scale-invariant anisotropic distance.

Distances between line centroids are normalised by local line height so the
clusterer adapts to font size.  Horizontal distances are relaxed relative to
vertical distances, allowing lines in the same column to cluster even when
they share limited x-range.
"""

import numpy as np

from .block import Block, TextLineClusterer
from .text_detector import TextLine


class HDBSCANClusterer(TextLineClusterer):
    """Density-based text-block clusterer using HDBSCAN with a custom distance metric.

    Args:
        min_cluster_size: Minimum number of lines to form a cluster.  Lines
            that do not reach this threshold are treated as noise and returned
            as singleton blocks (whiteboard content is never truly noise).
        horizontal_scale: Divisor applied to normalised horizontal distance.
            Higher values allow wider horizontal spread within one cluster,
            useful for multi-column layouts.
    """

    def __init__(self, min_cluster_size: int = 2, horizontal_scale: float = 3.5):
        self.min_cluster_size = min_cluster_size
        self.horizontal_scale = horizontal_scale

    def _custom_pairwise_distance(self, centroids: np.ndarray) -> np.ndarray:
        """Compute a scale-invariant anisotropic (N, N) distance matrix.

        Distances are normalised by the average line height of each pair so
        the metric is independent of font size.  Horizontal gaps are divided
        by an additional `horizontal_scale` factor, making vertical proximity
        the primary clustering signal while still permitting lines in the same
        column (wide horizontal spread, tight vertical spread) to merge.

        Args:
            centroids: (N, 3) array of [cx, cy, line_height] per line.

        Returns:
            Symmetric (N, N) float64 distance matrix.
        """
        cx = centroids[:, 0:1]
        cy = centroids[:, 1:2]
        h = centroids[:, 2:3]
        dx = np.abs(cx - cx.T)
        dy = np.abs(cy - cy.T)
        scale = np.maximum((h + h.T) / 2.0, 1e-5)
        norm_dx = dx / (self.horizontal_scale * scale)
        norm_dy = dy / scale
        return np.sqrt(norm_dx**2 + norm_dy**2)

    def group(self, lines: list[TextLine]) -> list[Block]:
        """Cluster *lines* into Blocks using HDBSCAN on the custom distance matrix.

        Lines assigned label -1 (HDBSCAN noise) are returned as singleton
        blocks rather than discarded — on a whiteboard every detected line
        represents real content.

        Args:
            lines: Detected text lines from Stage 5.

        Returns:
            List of Blocks, each grouping one or more lines.
        """
        if not lines:
            return []
        if len(lines) == 1:
            return [
                Block(bbox=lines[0].bbox, confidence=lines[0].confidence, lines=lines)
            ]

        from hdbscan import HDBSCAN

        centroids = []
        for line in lines:
            cx = (line.bbox[0] + line.bbox[2]) / 2.0
            cy = (line.bbox[1] + line.bbox[3]) / 2.0
            h = line.bbox[3] - line.bbox[1]
            centroids.append([cx, cy, h])

        X = np.array(centroids, dtype=np.float64)
        dist_matrix = self._custom_pairwise_distance(X)

        clusterer = HDBSCAN(
            metric="precomputed",
            min_cluster_size=self.min_cluster_size,
            min_samples=1,
        )
        labels = clusterer.fit_predict(dist_matrix)

        groups_dict = {}
        for idx, label in enumerate(labels):
            # Noise lines (-1) become singleton blocks rather than being dropped.
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
            blocks.append(
                Block(bbox=bbox, confidence=max_conf, lines=constituent_lines)
            )

        return blocks
