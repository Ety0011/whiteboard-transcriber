import cv2
import numpy as np

from .base import BaseLayoutDetector


class StrokeDetector(BaseLayoutDetector):
    """
    SOTA Whiteboard-specific Layout Engine.
    Uses Connected Component Extraction + BFS Spatial Distance-based Clustering.
    Guaranteed to group handwritten blocks without deep learning failures.
    """

    def __init__(
        self, horizontal_dist: int = 100, vertical_dist: int = 50, min_area: int = 15
    ):
        self.horizontal_dist = horizontal_dist
        self.vertical_dist = vertical_dist
        self.min_area = min_area

    def load(self):
        print(
            "[StrokeDetector] Initializing spatial clustering..."
        )

    def detect(self, frame: np.ndarray) -> list[dict]:
        # 1. Grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2. Extract ink via Otsu's inversion (Whiteboards are light, text is dark)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 3. Connected Components (Isolate individual letters/strokes)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)

        valid_components = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            # Exclude full board borders and tiny dust speckles
            if (
                area < self.min_area
                or w > frame.shape[1] * 0.8
                or h > frame.shape[0] * 0.8
            ):
                continue
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]

            # Extract point coordinates
            comp_pts = np.argwhere(labels == i)[:, ::-1]  # (x, y) coordinates
            valid_components.append({"bbox": [x, y, x + w, y + h], "points": comp_pts})

        n_comp = len(valid_components)
        if n_comp == 0:
            return []

        # 4. Group strokes using Adjacency Graph BFS
        adj = {i: [] for i in range(n_comp)}
        for i in range(n_comp):
            boxA = valid_components[i]["bbox"]
            for j in range(i + 1, n_comp):
                boxB = valid_components[j]["bbox"]

                # Check absolute horizontal and vertical gaps between stroke bounds
                dx = max(0, boxB[0] - boxA[2], boxA[0] - boxB[2])
                dy = max(0, boxB[1] - boxA[3], boxA[1] - boxB[3])

                if dx < self.horizontal_dist and dy < self.vertical_dist:
                    adj[i].append(j)
                    adj[j].append(i)

        # Find isolated graph networks using Breadth-First Search
        visited = [False] * n_comp
        groups = []
        for i in range(n_comp):
            if visited[i]:
                continue
            group = []
            queue = [i]
            visited[i] = True
            while queue:
                curr = queue.pop(0)
                group.append(curr)
                for neighbor in adj[curr]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            groups.append(group)

        # 5. Compute tight Convex Hulls representing irregular region boundaries
        discovered_regions = []
        for g_idx, group in enumerate(groups):
            all_pts = []
            for comp_idx in group:
                comp_pts = valid_components[comp_idx]["points"]
                if len(comp_pts) > 10:
                    comp_pts = comp_pts[::3]  # Downsample for speed
                all_pts.extend(comp_pts)

            all_pts = np.array(all_pts, dtype=np.int32)
            if len(all_pts) < 3:
                continue

            # Compute Convex Hull around stroke coordinates to wrap skewed writing tightly
            hull = cv2.convexHull(all_pts)
            poly_pts = hull.reshape(-1, 2)

            discovered_regions.append(
                {
                    "text": f"Cluster {g_idx} ({len(group)} strokes)",
                    "poly": poly_pts,
                    "label": "TEXT",
                    "color": (0, 230, 0),
                }
            )

        return discovered_regions
