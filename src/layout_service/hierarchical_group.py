import numpy as np

from .anchor_detector import Anchor, AnchorDetector, UnionFind
from .base import BaseLayoutDetector


class HierarchicalGroupDetector(BaseLayoutDetector):
    """
    Combines multiprocessing AnchorDetector (PP-OCRv5_server_det)
    with hierarchical Union-Find grouping to discover unified semantic blocks.
    """

    def __init__(
        self,
        box_thresh: float = 0.6,
        unclip_ratio: float = 1.2,
        iou_threshold: float = 0.02,
    ):
        self.box_thresh = box_thresh
        self.unclip_ratio = unclip_ratio
        self.iou_threshold = iou_threshold
        self.anchor_detector = None

    def load(self) -> None:
        print(
            "[HierarchicalGroupDetector] Spawning multiprocessing AnchorDetector (PP-OCRv5_server_det)..."
        )
        self.anchor_detector = AnchorDetector(
            box_thresh=self.box_thresh, unclip_ratio=self.unclip_ratio
        )

    def detect(self, frame: np.ndarray) -> list[dict]:
        # 1. Fetch latest cached result from the non-blocking worker
        result = self.anchor_detector.detect(frame)
        anchors = result.anchors

        if not anchors:
            return []

        # 2. Adaptive Spatial Thresholds scaled dynamically to the median line height
        heights = [a.bbox[3] - a.bbox[1] for a in anchors]
        median_height = np.median(heights) if heights else 20.0

        vertical_expand = median_height * 0.65
        horizontal_expand = median_height * 0.25

        # 3. Disjoint-Set Clustering
        num_anchors = len(anchors)
        uf = UnionFind(num_anchors)

        for i in range(num_anchors):
            for j in range(i + 1, num_anchors):
                if self._should_merge(
                    anchors[i], anchors[j], vertical_expand, horizontal_expand
                ):
                    uf.union(i, j)

        # Assemble grouping clusters
        sets: dict[int, list[Anchor]] = {}
        for i in range(num_anchors):
            root = uf.find(i)
            if root not in sets:
                sets[root] = []
            sets[root].append(anchors[i])

        # 4. Extract clustered blocks and return tight bounding boxes
        discovered_regions = []
        for g_idx, group_anchors in enumerate(sets.values()):
            bboxes = np.stack([a.bbox for a in group_anchors])
            merged_bbox = np.array(
                [
                    bboxes[:, 0].min(),
                    bboxes[:, 1].min(),
                    bboxes[:, 2].max(),
                    bboxes[:, 3].max(),
                ],
                dtype=np.int32,
            )

            x1, y1, x2, y2 = merged_bbox
            poly_pts = np.array(
                [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32
            )

            discovered_regions.append(
                {
                    "text": f"Block {g_idx} ({len(group_anchors)} lines)",
                    "poly": poly_pts,
                    "label": "TEXT",
                    "color": (255, 0, 255),  # Pink Group Overlay
                }
            )

        return sorted(discovered_regions, key=lambda g: g["poly"][:, 1].min())

    def _should_merge(
        self, a: Anchor, b: Anchor, v_expand: float, h_expand: float
    ) -> bool:
        ax1, ay1, ax2, ay2 = a.bbox.tolist()
        bx1, by1, bx2, by2 = b.bbox.tolist()

        # Multi-Column Gutter Guard Check:
        gap_x = max(0, bx1 - ax2, ax1 - bx2)
        if gap_x > max(ax2 - ax1, bx2 - bx1) * 0.35:
            return False

        ax1e, ax2e = ax1 - h_expand, ax2 + h_expand
        ay1e, ay2e = ay1 - v_expand, ay2 + v_expand
        bx1e, bx2e = bx1 - h_expand, bx2 + h_expand
        by1e, by2e = by1 - v_expand, by2 + v_expand

        ix1, iy1 = max(ax1e, bx1e), max(ay1e, by1e)
        ix2, iy2 = min(ax2e, bx2e), min(ay2e, by2e)

        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter == 0.0:
            return False

        area_a = (ax2e - ax1e) * (ay2e - ay1e)
        area_b = (bx2e - bx1e) * (by2e - by1e)
        return inter / (area_a + area_b - inter) > self.iou_threshold
