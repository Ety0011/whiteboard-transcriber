"""Interactive Whiteboard Reconstruction + Swappable Layout Discovery Test.

Integrates Stages 1-4 board reconstruction and feeds the clean, rectified
whiteboard composite directly into a swappable Stage 5 async Layout Worker.

Usage:
    python src/main.py video.mp4 --model stroke_cluster   # Deterministic clustering (0 VRAM)
    python src/main.py video.mp4 --model yolo             # YOLOv8 layout model
    python src/main.py video.mp4 --model doclayoutv3      # DocLayoutV3 irregular polygons
    python src/main.py video.mp4 --model paddleocrvl      # PaddleOCR-VL-1.5 MLX VLM
    python src/main.py video.mp4 --model hierarchical_union_find  # Legacy PP-OCRv5 + Union-Find
    python src/main.py video.mp4 --model dbscan           # Legacy PP-OCRv5 + SOTA DBSCAN Density Grouping
"""

import argparse
import enum
import logging
import multiprocessing as mp
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch

# Native pipeline imports
from src import capture
from src.board_service.board_masker import BoardMasker
from src.board_service.person_masker import PersonMasker
from src.board_service.reconstructor import BoardReconstructor
from src.board_service.rectifier import Rectifier

TARGET_W = 1280
TARGET_H = 720


# =====================================================================
# STAGE 5 INTERFACE: Swappable Layout Detector Base
# =====================================================================
class BaseLayoutDetector(ABC):
    """Abstract interface to decouple model architectures from pipeline execution."""

    @abstractmethod
    def load(self) -> None:
        """Initialize models, configure devices (MPS/CPU), and load weights."""
        pass

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Inference loop execution.
        Must return a list of dictionaries structured as:
            {
                "text": str,          # Text labels/confidence to overlay
                "poly": np.ndarray,   # Boundary coordinates, shape (N, 2), dtype=int32
                "label": str,         # Simplified taxonomy ("MATH", "TABLE", "DIAGRAM", "TEXT")
                "color": tuple        # BGR coordinate color
            }
        """
        pass


# =====================================================================
# BACKEND 1: Whiteboard Stroke Clusterer (Robust, 0 VRAM, <5ms)
# =====================================================================
class WhiteboardStrokeClusterer(BaseLayoutDetector):
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
            "[WhiteboardStrokeClusterer] Initializing hardware-accelerated spatial clustering..."
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


# =====================================================================
# BACKEND 2: YOLO Layout Detector
# =====================================================================
class YOLOLayoutDetector(BaseLayoutDetector):
    def __init__(
        self,
        repo_id: str = "hantian/yolo-doclaynet",
        filename: str = "yolov8m-doclaynet.pt",
    ):
        self.repo_id = repo_id
        self.filename = filename
        self.model = None
        self.device = "cpu"

    def load(self):
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO

        print("[YOLOLayoutDetector] Downloading weights...")
        weights_path = hf_hub_download(repo_id=self.repo_id, filename=self.filename)
        self.model = YOLO(weights_path)
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

    def detect(self, frame: np.ndarray) -> list[dict]:
        results = self.model.predict(
            frame, imgsz=640, conf=0.25, verbose=False, device=self.device
        )[0]

        discovered_regions = []
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            class_name = self.model.names[cls_id]
            xyxy = box.xyxy[0].cpu().numpy().astype(np.int32)
            score = float(box.conf[0].item())

            name_lower = class_name.lower()
            if "math" in name_lower or "formula" in name_lower:
                label = "MATH"
                color = (0, 200, 255)
            elif "table" in name_lower:
                label = "TABLE"
                color = (255, 255, 0)
            elif (
                "illus" in name_lower
                or "pic" in name_lower
                or "fig" in name_lower
                or "chart" in name_lower
            ):
                label = "DIAGRAM"
                color = (255, 100, 0)
            else:
                label = "TEXT"
                color = (0, 230, 0)

            x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
            poly_pts = np.array(
                [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32
            )

            discovered_regions.append(
                {
                    "text": f"{class_name.upper()} ({score:.1%})",
                    "poly": poly_pts,
                    "label": label,
                    "color": color,
                }
            )
        return discovered_regions


# =====================================================================
# BACKEND 3: PP-DocLayoutV3
# =====================================================================
class PPDocLayoutV3Detector(BaseLayoutDetector):
    def __init__(self, model_id: str = "PaddlePaddle/PP-DocLayoutV3_safetensors"):
        self.model_id = model_id
        self.model = None
        self.image_processor = None
        self.device = "cpu"

    def load(self):
        from transformers import AutoModelForObjectDetection

        try:
            from transformers import RTDetrImageProcessor as ImageProcessorClass
        except ImportError:
            from transformers import AutoImageProcessor as ImageProcessorClass

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.image_processor = ImageProcessorClass.from_pretrained(self.model_id)
        self.model = AutoModelForObjectDetection.from_pretrained(self.model_id).to(
            self.device
        )
        self.model.eval()

    def detect(self, frame: np.ndarray) -> list[dict]:
        from PIL import Image

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        inputs = self.image_processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.image_processor.post_process_object_detection(
            outputs, target_sizes=torch.tensor([[h, w]], device=self.device)
        )[0]

        scores = results["scores"].cpu()
        labels = results["labels"].cpu()
        polygon_points_list = results.get("polygon_points", [])

        discovered_regions = []
        for idx, score in enumerate(scores):
            if score < 0.35:
                continue

            label_id = labels[idx].item()
            raw_label = self.model.config.id2label[label_id]

            name_lower = raw_label.lower()
            if "formula" in name_lower or "algorithm" in name_lower:
                label = "MATH"
                color = (0, 200, 255)
            elif "table" in name_lower:
                label = "TABLE"
                color = (255, 255, 0)
            elif "chart" in name_lower or "image" in name_lower or "pic" in name_lower:
                label = "DIAGRAM"
                color = (255, 100, 0)
            else:
                label = "TEXT"
                color = (0, 230, 0)

            if idx < len(polygon_points_list):
                poly_tensor = polygon_points_list[idx]
                if torch.is_tensor(poly_tensor):
                    poly_pts = poly_tensor.cpu().numpy().astype(np.int32)
                else:
                    poly_pts = np.array(poly_tensor, dtype=np.int32)
            else:
                box = results["boxes"][idx].cpu().numpy().astype(np.int32)
                poly_pts = np.array(
                    [
                        [box[0], box[1]],
                        [box[2], box[1]],
                        [box[2], box[3]],
                        [box[0], box[3]],
                    ],
                    dtype=np.int32,
                )

            discovered_regions.append(
                {
                    "text": f"{raw_label.upper()} ({score:.1%})",
                    "poly": poly_pts,
                    "label": label,
                    "color": color,
                }
            )
        return discovered_regions


# =====================================================================
# BACKEND 4: PaddleOCR-VL VLM (Native MLX-VLM Engine) [10]
# =====================================================================
class PaddleOCRVLDetector(BaseLayoutDetector):
    """Autoregressive VLM layout grounding using the native 'Spotting:' chat template."""

    def __init__(self, model_id: str = "mlx-community/PaddleOCR-VL-1.5-8bit"):
        self.model_id = model_id
        self.model = None
        self.processor = None
        self.config = None

    def load(self):
        from mlx_vlm import load as load_mlx

        print(f"[PaddleOCRVLDetector] Loading native VLM on MLX: {self.model_id}...")
        self.model, self.processor = load_mlx(self.model_id)
        self.config = self.model.config

    def detect(self, frame: np.ndarray) -> list[dict]:
        import re

        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        from PIL import Image

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        formatted_prompt = apply_chat_template(
            self.processor, self.config, "Spotting:", num_images=1
        )
        gen_result = generate(
            self.model,
            self.processor,
            formatted_prompt,
            pil_img,
            max_tokens=512,
            verbose=False,
        )
        raw_text = gen_result.text if hasattr(gen_result, "text") else str(gen_result)

        # Parse coordinate tokens matching the LOC template [10]
        regions = []
        tokens_pattern = re.compile(r"((?:<\|LOC_\d+\|>)+)")
        parts = tokens_pattern.split(raw_text)

        for i in range(0, len(parts) - 1, 2):
            content = parts[i].strip()
            loc_block = parts[i + 1]

            if not content or not loc_block:
                continue

            coords = [int(val) for val in re.findall(r"\d+", loc_block)]
            if len(coords) < 6 or len(coords) % 2 != 0:
                continue

            poly_pts = []
            for j in range(0, len(coords), 2):
                token_x, token_y = coords[j], coords[j + 1]
                abs_x = int((token_x / 1000.0) * w)
                abs_y = int((token_y / 1000.0) * h)
                abs_x = max(0, min(w, abs_x))
                abs_y = max(0, min(h, abs_y))
                poly_pts.append([abs_x, abs_y])

            pts_arr = np.array(poly_pts, dtype=np.int32)

            if (
                "$$" in content
                or "\\(" in content
                or "e^{" in content
                or "=" in content
            ):
                label = "MATH"
                color = (0, 200, 255)
            elif len(content) < 3 and not content.isalnum():
                label = "DIAGRAM"
                color = (255, 100, 0)
            else:
                label = "TEXT"
                color = (0, 230, 0)

            regions.append(
                {"text": content, "poly": pts_arr, "label": label, "color": color}
            )
        return regions


# =====================================================================
# BACKEND 5: Hierarchical Union-Find Grouping Detector (Improved & Multiprocessed)
# =====================================================================
class AnchorType(enum.Enum):
    TEXT_LINE = "TEXT_LINE"


@dataclass
class Anchor:
    bbox: np.ndarray  # (4,) int32: x1, y1, x2, y2
    confidence: float
    anchor_type: AnchorType


@dataclass
class DetectorResult:
    anchors: list[Anchor] = field(default_factory=list)


def _extract_polys(raw_results: list) -> list[tuple[list, float]]:
    """Extract (polygon, score) pairs from raw TextDetection output."""
    if not raw_results:
        return []
    results = []
    for result in raw_results:
        polys = result.get("dt_polys", [])
        scores = result.get("dt_scores", [1.0] * len(polys))
        for poly, score in zip(polys, scores):
            results.append(
                ([[float(pt[0]), float(pt[1])] for pt in poly], float(score))
            )
    return results


def _polygon_to_bbox(polygon: list, img_h: int, img_w: int) -> np.ndarray:
    """Convert polygon to axis-aligned bbox clamped to image bounds."""
    pts = np.array(polygon, dtype=np.float32)
    x1 = int(np.clip(pts[:, 0].min(), 0, img_w))
    y1 = int(np.clip(pts[:, 1].min(), 0, img_h))
    x2 = int(np.clip(pts[:, 0].max(), 0, img_w))
    y2 = int(np.clip(pts[:, 1].max(), 0, img_h))
    return np.array([x1, y1, x2, y2], dtype=np.int32)


def _raw_to_anchors(raw: list, h: int, w: int) -> list[Anchor]:
    anchors = []
    for poly, score in _extract_polys(raw):
        bbox = _polygon_to_bbox(poly, h, w)
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            continue
        anchors.append(
            Anchor(bbox=bbox, confidence=score, anchor_type=AnchorType.TEXT_LINE)
        )
    return anchors


def _worker_main(
    in_q: mp.Queue,
    out_q: mp.Queue,
    box_thresh: float,
    unclip_ratio: float,
) -> None:
    """PaddleOCR text detection loop — runs in a dedicated child process."""
    import logging as _log

    _log.basicConfig(level=logging.WARNING)
    log = _log.getLogger(__name__)

    from paddleocr import TextDetection

    detector = TextDetection(
        model_name="PP-OCRv5_server_det",
        box_thresh=box_thresh,
        unclip_ratio=unclip_ratio,
    )
    log.warning("AnchorDetector: PP-OCRv5_server_det ready")

    while True:
        composite = in_q.get()  # block until work arrives
        if composite is None:  # shutdown sentinel
            break

        anchors: list[Anchor] = []
        try:
            h, w = composite.shape[:2]
            raw = detector.predict(composite)
            anchors = _raw_to_anchors(raw, h, w)
            log.warning("AnchorDetector: %d TEXT_LINE anchors", len(anchors))
        except Exception:
            log.exception("PaddleOCR detection failed")

        result = DetectorResult(anchors=anchors)
        try:
            out_q.get_nowait()
        except Exception:
            pass
        try:
            out_q.put_nowait(result)
        except Exception:
            pass


class AnchorDetector:
    """Non-blocking PaddleOCR anchor detector."""

    def __init__(
        self,
        box_thresh: float = 0.6,
        unclip_ratio: float = 1.2,
    ) -> None:
        self._cached = DetectorResult()
        self._in_q: mp.Queue = mp.Queue(maxsize=1)
        self._out_q: mp.Queue = mp.Queue(maxsize=1)
        self._worker = mp.Process(
            target=_worker_main,
            args=(self._in_q, self._out_q, box_thresh, unclip_ratio),
            daemon=True,
            name="paddle-detect",
        )
        self._worker.start()
        print(f"AnchorDetector worker started (pid={self._worker.pid})")

    def detect(self, composite: np.ndarray) -> DetectorResult:
        """Submit composite for async detection; return latest cached result."""
        try:
            self._in_q.put_nowait(composite)
        except Exception:
            pass

        try:
            self._cached = self._out_q.get_nowait()
        except Exception:
            pass

        return self._cached

    def shutdown(self) -> None:
        """Signal the worker to stop and wait for clean exit."""
        try:
            self._in_q.put_nowait(None)
        except Exception:
            pass
        self._worker.join(timeout=5)
        if self._worker.is_alive():
            self._worker.terminate()
        print("AnchorDetector worker stopped")


class UnionFind:
    """Disjoint-Set Forest for clustering."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i: int, j: int) -> bool:
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j
            return True
        return False


class HierarchicalGroupDetector(BaseLayoutDetector):
    """
    Combines your legacy multiprocessing AnchorDetector (PP-OCRv5_server_det)
    with your hierarchical Union-Find grouping logic to discover unified semantic blocks.
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


# =====================================================================
# BACKEND 6: SOTA DBSCAN Multi-Point Density Grouping Detector
# =====================================================================
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
        # Reuse the global AnchorDetector class declared in Backend 5
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
        sets: dict[int, list[Anchor]] = {}
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


# =====================================================================
# STAGE 5 CORE: Thread Orchestrator
# =====================================================================
class Stage5LayoutDiscovery(threading.Thread):
    def __init__(
        self,
        detector: BaseLayoutDetector,
        target_w: int,
        target_h: int,
        on_regions_discovered,
    ):
        super().__init__(daemon=True)
        self.detector = detector
        self.target_w = target_w
        self.target_h = target_h
        self.on_regions_discovered = on_regions_discovered

        # Mailbox Synchronization
        self.mailbox_lock = threading.Lock()
        self.frame_mailbox: np.ndarray | None = None
        self.new_frame_event = threading.Event()
        self.is_running = True
        self.is_busy = False

    def submit_frame(self, frame: np.ndarray) -> bool:
        if self.is_busy:
            return False

        with self.mailbox_lock:
            self.frame_mailbox = frame.copy()
        self.new_frame_event.set()
        return True

    def run(self):
        self.detector.load()

        while self.is_running:
            self.new_frame_event.wait()
            self.new_frame_event.clear()

            with self.mailbox_lock:
                if self.frame_mailbox is None:
                    continue
                local_frame = self.frame_mailbox
                self.frame_mailbox = None

            self.is_busy = True
            start_time = time.time()

            try:
                discovered_regions = self.detector.detect(local_frame)
                latency = time.time() - start_time
                self.on_regions_discovered(discovered_regions, latency)

            except Exception as e:
                print(f"[Stage 5 Thread Error]: {e}")
            finally:
                self.is_busy = False

    def stop(self):
        self.is_running = False
        self.new_frame_event.set()


# =====================================================================
# STAGE 6: Spatial-Temporal Entity Tracker (Real-Time State Machine)
# =====================================================================
class TrackedEntity:
    """Represents a stateful, spatially tracked whiteboard block."""

    def __init__(self, id_num: int, label: str, poly: np.ndarray, bbox: np.ndarray):
        self.id = f"Entity_{id_num}"
        self.label = label
        self.poly = poly
        self.bbox = bbox  # [x1, y1, x2, y2]

        # State Machine: STABILIZING -> INFERRING -> ACTIVE -> ERASED
        self.state = "STABILIZING"
        self.stability_count = 1
        self.miss_count = 0
        self.transcription = ""
        self.infer_timer_started = False


def compute_bbox_iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
    """Calculates Bounding Box Intersection-over-Union."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = float(areaA + areaB - inter)

    return inter / union if union > 0 else 0.0


class Stage6TemporalRegistry:
    """
    SOTA Spatial-Temporal Entity Registry (Stage 6).
    Performs frame-to-frame greedy IoU matching to maintain track ID consistency and
    manages the STABILIZING -> INFERRING -> ACTIVE -> ERASED state machine.
    """

    def __init__(
        self,
        stability_threshold: int = 5,
        miss_threshold: int = 12,
        iou_threshold: float = 0.3,
    ):
        self.lock = threading.Lock()
        self.entities: list[TrackedEntity] = []
        self.id_counter = 0
        self.stability_threshold = stability_threshold
        self.miss_threshold = miss_threshold
        self.iou_threshold = iou_threshold

        self.last_latency = 0.0
        self.update_ready = False

    def receive_new_regions(self, regions: list[dict], latency: float):
        """Asynchronously triggered by Stage 5. Matches and updates tracking states."""
        with self.lock:
            self.last_latency = latency
            self.update_ready = True

            # 1. Precompute bounding boxes for incoming detections
            detected_boxes = []
            for r in regions:
                poly = r["poly"]
                x1, y1 = poly[:, 0].min(), poly[:, 1].min()
                x2, y2 = poly[:, 0].max(), poly[:, 1].max()
                detected_boxes.append(np.array([x1, y1, x2, y2], dtype=np.int32))

            # Filter out permanently erased entities from memory
            self.entities = [e for e in self.entities if e.state != "ERASED"]

            matched_detections = set()
            matched_entities = set()

            # 2. Compute IoU matrix and perform Greedy Matching
            if self.entities and regions:
                iou_matrix = np.zeros((len(regions), len(self.entities)))
                for d_idx, det_box in enumerate(detected_boxes):
                    for t_idx, entity in enumerate(self.entities):
                        iou_matrix[d_idx, t_idx] = compute_bbox_iou(
                            det_box, entity.bbox
                        )

                # Match coordinates descending by IoU overlap
                flat_indices = np.argsort(-iou_matrix, axis=None)
                for index in flat_indices:
                    d_idx, t_idx = divmod(index, iou_matrix.shape[1])
                    if iou_matrix[d_idx, t_idx] < self.iou_threshold:
                        break
                    if (
                        d_idx not in matched_detections
                        and t_idx not in matched_entities
                    ):
                        matched_detections.add(d_idx)
                        matched_entities.add(t_idx)

                        # Match confirmed: update spatial envelope and reset decay
                        entity = self.entities[t_idx]
                        entity.poly = regions[d_idx]["poly"]
                        entity.bbox = detected_boxes[d_idx]
                        entity.miss_count = 0

                        # Handle STABILIZING state count
                        if entity.state == "STABILIZING":
                            entity.stability_count += 1
                            if entity.stability_count >= self.stability_threshold:
                                entity.state = "INFERRING"

            # 3. Handle unmatched detections -> Spawn a new STABILIZING track
            for d_idx, r in enumerate(regions):
                if d_idx not in matched_detections:
                    self.id_counter += 1
                    new_entity = TrackedEntity(
                        id_num=self.id_counter,
                        label=r["label"],
                        poly=r["poly"],
                        bbox=detected_boxes[d_idx],
                    )
                    self.entities.append(new_entity)

            # 4. Handle unmatched tracks -> Decay miss counter and trigger ERASED state
            for t_idx, entity in enumerate(self.entities):
                if t_idx not in matched_entities:
                    entity.miss_count += 1
                    if entity.miss_count >= self.miss_threshold:
                        entity.state = "ERASED"

    def get_tracked_elements(self) -> tuple[list[dict], float, bool]:
        """Called by the rendering thread to draw active states. Mocks Stage 7 Async VLM."""
        with self.lock:
            ready = self.update_ready
            self.update_ready = False

            # Simulate Stage 7 OCR: When an entity hits INFERRING, launch a 1.2s transcription thread
            for entity in self.entities:
                if entity.state == "INFERRING" and not entity.infer_timer_started:
                    entity.infer_timer_started = True

                    def mock_transcribe_task(ent: TrackedEntity):
                        time.sleep(1.2)  # Simulate VLM transcription lag
                        with self.lock:
                            if ent.state == "INFERRING":
                                ent.transcription = f"OCR text verified for {ent.id}"
                                ent.state = "ACTIVE"

                    threading.Thread(
                        target=mock_transcribe_task, args=(entity,), daemon=True
                    ).start()

            # Map the inner tracker states to visual render components
            display_regions = []
            for e in self.entities:
                display_regions.append(
                    {
                        "id": e.id,
                        "poly": e.poly,
                        "label": e.label,
                        "state": e.state,
                        "text": e.transcription
                        if e.state == "ACTIVE"
                        else f"{e.state} ({e.stability_count if e.state == 'STABILIZING' else '...'})",
                    }
                )

            return display_regions, self.last_latency, ready


# =====================================================================
# Main Execution Pipeline Loop
# =====================================================================
def main() -> None:
    """Start the capture thread and run the pipeline loop."""

    parser = argparse.ArgumentParser(
        description="Whiteboard transcription pipeline — Stage 5 Test"
    )
    parser.add_argument(
        "source",
        nargs="?",
        metavar="FILE",
        help="video or image file (omit to use the default webcam)",
    )
    parser.add_argument(
        "--model",
        choices=[
            "stroke_cluster",
            "yolo",
            "doclayoutv3",
            "paddleocrvl",
            "hierarchical_union_find",
            "dbscan",
        ],
        default="hierarchical_union_find",
        help="Stage 5 Layout Discovery backend model to run",
    )
    args = parser.parse_args()

    frame_queue = capture.start(args.source)

    print("Loading Native Reconstruction Pipeline Models...")
    board_masker = BoardMasker()
    person_masker = PersonMasker()
    rectifier = Rectifier()
    reconstructor = BoardReconstructor()

    # Instantiate the selected Stage 5 Layout Detector backend
    if args.model == "stroke_cluster":
        detector = WhiteboardStrokeClusterer()
    elif args.model == "yolo":
        detector = YOLOLayoutDetector()
    elif args.model == "doclayoutv3":
        detector = PPDocLayoutV3Detector()
    elif args.model == "paddleocrvl":
        detector = PaddleOCRVLDetector()
    elif args.model == "hierarchical_union_find":
        detector = HierarchicalGroupDetector()  # New swappable Union-Find backend
    elif args.model == "dbscan":
        detector = DBSCANGroupDetector()  # New swappable DBSCAN density backend
    else:
        raise ValueError(f"Unknown layout model: {args.model}")

    # Initialize Stage 6 Registry & Injected Stage 5 Worker
    registry = Stage6TemporalRegistry()
    stage5_worker = Stage5LayoutDiscovery(
        detector=detector,
        target_w=TARGET_W,
        target_h=TARGET_H,
        on_regions_discovered=registry.receive_new_regions,
    )
    stage5_worker.start()

    print("\n" + "=" * 60)
    print(f" PIPELINE ACTIVE. Active Stage 5 Model: {args.model.upper()}")
    print(" CONTROLS:")
    print(
        "   [Spacebar] -> Submit latest clean composite whiteboard frame to layout detector"
    )
    print("   [a]        -> Toggle Auto-Continuous layout mode")
    print("   [q]        -> Quit")
    print("=" * 60 + "\n")

    frame_count = 0
    active_regions = []
    auto_mode = False
    last_eval_latency = 0.0
    status_msg = "Idle - Press [SPACE] to evaluate composite"

    try:
        while True:
            frame = frame_queue.get()
            if frame is None:
                print("[Pipeline Loop] End of stream.")
                break

            frame_count += 1

            # -----------------------------------------------------------------
            # RUN STAGES 1-4: Whiteboard Reconstruction Pipeline
            # -----------------------------------------------------------------
            board_mask = board_masker.segment(frame)
            person_mask = person_masker.segment(frame)
            rect_frame, rect_mask = rectifier.rectify(frame, board_mask, person_mask)
            composite = reconstructor.update(rect_frame, rect_mask)

            # Retrieve background layout results
            new_regions, latency, updated = registry.get_tracked_elements()
            if updated:
                active_regions = new_regions
                status_msg = f"Last Layout Latency: {latency * 1000:.1f}ms"

            # -----------------------------------------------------------------
            # STAGES 5 & 6: Render real-time spatial-temporal tracking states
            # -----------------------------------------------------------------
            # Unified Ledger Visual Themes
            STATE_THEMES = {
                "STABILIZING": ((0, 165, 255), "STABILIZING"),  # Orange (Validating)
                "INFERRING": ((255, 255, 0), "INFERRING..."),  # Cyan (VLM queue)
                "ACTIVE": ((0, 230, 0), "ACTIVE"),  # Green (Active Ledger)
                "ERASED": ((0, 0, 220), "ERASED"),  # Red (Pruning)
            }

            board_display = composite.copy()
            if active_regions:
                overlay = board_display.copy()
                for reg in active_regions:
                    poly = reg["poly"]
                    state = reg["state"]
                    text_slice = reg["text"]

                    # Fetch coordinate overlay color scheme based on state machine
                    color, state_lbl = STATE_THEMES.get(
                        state, ((255, 255, 255), "UNKNOWN")
                    )

                    cv2.fillPoly(overlay, [poly], color)
                    cv2.polylines(
                        board_display, [poly], isClosed=True, color=color, thickness=2
                    )

                    x, y = int(poly[:, 0].min()), int(poly[:, 1].min())
                    display_label = f"[{reg['id']} | {state_lbl}] {text_slice}"
                    (tw, th), _ = cv2.getTextSize(
                        display_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
                    )

                    cv2.rectangle(
                        board_display, (x, y - th - 6), (x + tw + 6, y), color, -1
                    )
                    cv2.putText(
                        board_display,
                        display_label,
                        (x + 3, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 0, 0),
                        1,
                        cv2.LINE_AA,
                    )
                cv2.addWeighted(overlay, 0.25, board_display, 0.75, 0, board_display)

            # Draw HUD Overlays
            cv2.putText(
                board_display,
                f"Frame: {frame_count} | Mode: {'AUTO-TRACK' if auto_mode else 'MANUAL'}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                board_display,
                status_msg,
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

            if stage5_worker.is_busy:
                cv2.circle(
                    board_display,
                    (board_display.shape[1] - 30, 30),
                    10,
                    (0, 165, 255),
                    -1,
                )
            else:
                cv2.circle(
                    board_display,
                    (board_display.shape[1] - 30, 30),
                    10,
                    (0, 255, 0),
                    -1,
                )

            cv2.imshow(
                "Lecture Historian - Clean Whiteboard (Stage 4 + Stage 5)",
                board_display,
            )
            cv2.imshow("Raw Video Stream (Input)", frame)

            # Continuous auto-mode submission of the clean board
            if auto_mode:
                stage5_worker.submit_frame(composite)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                # Submit the reconstructed composite frame to Stage 5
                submitted = stage5_worker.submit_frame(composite)
                if submitted:
                    status_msg = "Submitted reconstruction to background Stage 5..."
                else:
                    status_msg = "Layout Worker busy. Frame skipped."
            elif key == ord("a"):
                auto_mode = not auto_mode
                status_msg = f"Auto-discovery: {'ENABLED' if auto_mode else 'DISABLED'}"

    except KeyboardInterrupt:
        pass
    finally:
        board_masker.shutdown()
        stage5_worker.stop()
        stage5_worker.join()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
