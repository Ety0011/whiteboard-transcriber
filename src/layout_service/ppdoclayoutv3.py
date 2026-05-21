import numpy as np
import torch

from .base import BaseLayoutDetector


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
        import cv2
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
