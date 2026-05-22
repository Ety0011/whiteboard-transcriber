import numpy as np

from .base import BaseLayoutDetector
from .grouper import Block


class PaddleVLDetector(BaseLayoutDetector):
    """Autoregressive VLM layout grounding using the native 'Spotting:' chat template."""

    def __init__(self, model_id: str = "mlx-community/PaddleOCR-VL-1.5-8bit"):
        self.model_id = model_id
        self.model = None
        self.processor = None
        self.config = None

    def load(self):
        from mlx_vlm import load as load_mlx

        print(f"[PaddleVLDetector] Loading native VLM on MLX: {self.model_id}...")
        self.model, self.processor = load_mlx(self.model_id)
        self.config = self.model.config

    def detect(self, frame: np.ndarray) -> list[Block]:
        import re

        import cv2
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

        blocks = []
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
                abs_x = max(0, min(w, int((token_x / 1000.0) * w)))
                abs_y = max(0, min(h, int((token_y / 1000.0) * h)))
                poly_pts.append([abs_x, abs_y])

            poly = np.array(poly_pts, dtype=np.int32)
            x1 = int(poly[:, 0].min())
            y1 = int(poly[:, 1].min())
            x2 = int(poly[:, 0].max())
            y2 = int(poly[:, 1].max())
            bbox = np.array([x1, y1, x2, y2], dtype=np.int32)
            blocks.append(Block(bbox=bbox, confidence=1.0))

        return blocks
