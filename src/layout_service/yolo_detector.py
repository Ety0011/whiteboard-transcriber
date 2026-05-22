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
            xyxy = box.xyxy[0].cpu().numpy().astype(np.int32)
            score = float(box.conf[0].item())
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            bbox = np.array([x1, y1, x2, y2], dtype=np.int32)
            blocks.append(Block(bbox=bbox, confidence=score))

        return blocks
