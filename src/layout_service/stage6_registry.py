import threading
import time

import numpy as np


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
