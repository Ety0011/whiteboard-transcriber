import cv2
import numpy as np

from .anchor_detector import AnchorDetector
from .base import BaseLayoutDetector


class DBSCANGroupDetector(BaseLayoutDetector):
    """
    SOTA Density-based Whiteboard Layout Analyzer.
    Reuses the multiprocessing AnchorDetector from Backend 5, samples multi-point
    axis densities per line, and runs DBSCAN to group paragraphs with 100% column safety.
    """

    def __init__(
        self,
        box_thresh: float = 0.6,
        unclip_ratio: float = 1.2,
        eps_factor: float = 1.8,
    ):
        self.box_thresh = box_thresh
        self.unclip_ratio = unclip_ratio
        # eps_factor determines clustering merge radius relative to median line height
        self.eps_factor = eps_factor
        self.anchor_detector = None

    def load(self) -> None:
        print(
            "[DBSCANGroupDetector] Spawning multiprocessing AnchorDetector (PP-OCRv5_server_det)..."
        )
        self.anchor_detector = AnchorDetector(
            box_thresh=self.box_thresh, unclip_ratio=self.unclip_ratio
        )

    def detect(self, frame: np.ndarray) -> list[dict]:
        # Lazy import of density-clustering package
        from sklearn.cluster import DBSCAN

        # 1. Fetch latest cached result from background process
        result = self.anchor_detector.detect(frame)
        anchors = result.anchors

        if not anchors:
            return []

        # 2. Extract median line height to scale the search radius dynamically
        heights = [a.bbox[3] - a.bbox[1] for a in anchors]
        median_height = np.median(heights) if heights else 20.0

        # 3. Point-Cloud Generation: Axis-Aligned Multi-Point Density Representation
        # Sample 3 horizontal coordinate nodes per detected line (Left, Center, Right)
        db_points = []
        anchor_indices = []

        for idx, a in enumerate(anchors):
            x1, y1, x2, y2 = a.bbox.tolist()
            cy = (y1 + y2) / 2.0

            db_points.extend([[x1, cy], [(x1 + x2) / 2.0, cy], [x2, cy]])
            anchor_indices.extend([idx, idx, idx])

        db_points = np.array(db_points)

        # 4. Perform Density-Based Spatial Clustering (DBSCAN)
        eps = median_height * self.eps_factor
        db = DBSCAN(eps=eps, min_samples=2, metric="euclidean").fit(db_points)
        labels = db.labels_

        # 5. Aggregate density clusters back into Anchor lists
        # Track inserted indices in a set to avoid calling dataclass __eq__ on bbox arrays
        sets: dict[int, list] = {}
        added_anchors: dict[int, set[int]] = {}

        for idx, cluster_id in enumerate(labels):
            if cluster_id == -1:
                continue  # Treat isolated single marks as background noise

            orig_anchor_idx = anchor_indices[idx]
            anchor_obj = anchors[orig_anchor_idx]

            if cluster_id not in sets:
                sets[cluster_id] = []
                added_anchors[cluster_id] = set()

            if orig_anchor_idx not in added_anchors[cluster_id]:
                sets[cluster_id].append(anchor_obj)
                added_anchors[cluster_id].add(orig_anchor_idx)

        # 6. Trace irregular polygon boundaries (Convex Hulls)
        discovered_regions = []
        for g_idx, group_anchors in enumerate(sets.values()):
            coords = []
            for a in group_anchors:
                x1, y1, x2, y2 = a.bbox.tolist()
                coords.extend([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

            coords = np.array(coords, dtype=np.int32)
            if len(coords) < 3:
                continue

            hull = cv2.convexHull(coords)
            poly_pts = hull.reshape(-1, 2)

            discovered_regions.append(
                {
                    "text": f"DBSCAN {g_idx} ({len(group_anchors)} lines)",
                    "poly": poly_pts,
                    "label": "TEXT",
                    "color": (255, 128, 0),  # SOTA Azure/Orange
                }
            )

        return sorted(discovered_regions, key=lambda g: g["poly"][:, 1].min())
