"""GotOcrTranscriber — GOT-OCR 2.0 backend.

Model: stepfun-ai/GOT-OCR-2.0-hf (HF-native, float16).
Device: MPS (Apple Silicon) if available, otherwise CPU.
Preprocessing: CLAHE contrast enhancement on L channel before inference.
"""

from __future__ import annotations

import cv2
import numpy as np

from .base import BaseTranscriber

_MODEL_ID = "stepfun-ai/GOT-OCR-2.0-hf"


def _preprocess_crop(bgr: np.ndarray) -> np.ndarray:
    """CLAHE contrast enhancement on L channel to maximise legibility for VLM."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


class GotTranscriber(BaseTranscriber):
    """GOT-OCR 2.0 — high-fidelity OCR/LaTeX on whiteboard entity crops."""

    def __init__(self, model_id: str = _MODEL_ID) -> None:
        self.model_id = model_id
        self._processor = None
        self._model = None
        self._device = "cpu"

    def load(self) -> None:
        """Load GOT-OCR 2.0 in float16 on MPS if available, otherwise CPU."""
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self._device = "mps" if self._mps_available() else "cpu"
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = (
            AutoModelForCausalLM.from_pretrained(
                self.model_id,
                dtype=torch.float16,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )
            .to(self._device)
            .eval()
        )

    def transcribe(self, crop: np.ndarray) -> str:
        """Run GOT-OCR 2.0 on *crop* and return recognised text/LaTeX.

        Applies CLAHE contrast enhancement before inference. Decodes only the
        newly generated tokens (strips the input prompt from the output).

        Args:
            crop: BGR uint8 entity crop from the clean board composite.

        Returns:
            Recognised text string, potentially containing LaTeX markup.
        """
        import torch
        from PIL import Image

        enhanced = _preprocess_crop(crop)
        rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        inputs = self._processor(pil_img, return_tensors="pt", format=True).to(
            self._device
        )
        with torch.inference_mode():
            generate_ids = self._model.generate(
                **inputs,
                do_sample=False,
                tokenizer=self._processor.tokenizer,
                stop_strings="<|im_end|>",
                max_new_tokens=512,
            )
        prompt_len = inputs["input_ids"].shape[1]
        return self._processor.decode(
            generate_ids[0, prompt_len:],
            skip_special_tokens=True,
        ).strip()

    @staticmethod
    def _mps_available() -> bool:
        """Return True if PyTorch MPS backend is available on this machine."""
        try:
            import torch

            return torch.backends.mps.is_available()
        except Exception:
            return False
