"""Stage 9 — PaddleVLTranscriber — PaddleOCR-VL-1.5 OCR backend.

Uses the MLX-quantised model with an "OCR:" prompt — returns plain text
without location tokens. Device: Apple Silicon MLX (native quantised 8-bit).
"""

from __future__ import annotations

import cv2
import numpy as np

_MODEL_ID = "mlx-community/PaddleOCR-VL-1.5-8bit"


class PaddleVLTranscriber:
    """PaddleOCR-VL-1.5 OCR — reads text from whiteboard entity crops via MLX."""

    def __init__(self, model_id: str = _MODEL_ID) -> None:
        self.model_id = model_id
        self._model = None
        self._processor = None
        self._config = None
        self._generate = None       # mlx_vlm.generate, cached at load time
        self._apply_chat = None     # mlx_vlm.prompt_utils.apply_chat_template
        self._pil_image = None      # PIL.Image module, cached at load time

    def load(self) -> None:
        """Load model weights inside the subprocess."""
        from mlx_vlm import generate, load as load_mlx
        from mlx_vlm.prompt_utils import apply_chat_template
        from PIL import Image

        self._model, self._processor = load_mlx(self.model_id)
        self._config = self._model.config
        self._generate = generate
        self._apply_chat = apply_chat_template
        self._pil_image = Image

    def transcribe(self, crop: np.ndarray) -> str:
        """Transcribe text from *crop* using PaddleOCR-VL-1.5.

        Args:
            crop: BGR uint8 image of a whiteboard note region.

        Returns:
            Recognised text string (may include LaTeX).
        """
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = self._pil_image.fromarray(rgb)

        prompt = self._apply_chat(self._processor, self._config, "OCR:", num_images=1)
        result = self._generate(
            self._model,
            self._processor,
            prompt,
            pil_img,
            max_tokens=512,
            verbose=False,
        )
        raw = result.text if hasattr(result, "text") else str(result)
        return raw.strip()
