import numpy as np
import torch

from .base import BaseLayoutDetector
from .grouper import Block


class YOLODetector(BaseLayoutDetector):
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

        print("[YOLODetector] Downloading weights...")
        weights_path = hf_hub_download(repo_id=self.repo_id, filename=self.filename)
        self.model = YOLO(weights_path)
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

    def detect(self, frame: np.ndarray) -> list[Block]:
        results = self.model.predict(
            frame, imgsz=640, conf=0.25, verbose=False, device=self.device
        )[0]

        blocks = []
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            class_name = self.model.names[cls_id]
            xyxy = box.xyxy[0].cpu().numpy().astype(np.int32)
            score = float(box.conf[0].item())

            name_lower = class_name.lower()
            if "math" in name_lower or "formula" in name_lower:
                label = "MATH"
            elif "table" in name_lower:
                label = "TABLE"
            elif (
                "illus" in name_lower
                or "pic" in name_lower
                or "fig" in name_lower
                or "chart" in name_lower
            ):
                label = "DIAGRAM"
            else:
                label = "TEXT"

            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
            bbox = np.array([x1, y1, x2, y2], dtype=np.int32)

            blocks.append(Block(poly=poly, bbox=bbox, label=label, confidence=score, anchors=[]))

        return blocks
