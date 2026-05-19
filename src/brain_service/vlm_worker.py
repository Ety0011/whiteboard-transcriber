"""Stage 7 — GOT-OCR 2.0 VLM Brain (async multiprocessing worker).

Receives Semantic Entity crops from Stage 6 and returns structured Markdown
text. Runs as a dedicated multiprocessing.Process so the main CV pipeline is
never blocked by VLM inference.

Model: stepfun-ai/GOT-OCR-2.0-hf (HF-native, no trust_remote_code, no verovio).
Device: MPS (Apple Silicon) if available, otherwise CPU.
Dtype: float16 to fit within the 11GB VLM memory budget.

Queue design:
  in_q  (maxsize=10): (region_id: int, crop: np.ndarray) — accepts multiple
        newly-stable regions from a single frame without dropping them.
  out_q (maxsize=30): VLMResult objects — drained by get_results() each frame.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

_MODEL_ID = "stepfun-ai/GOT-OCR-2.0-hf"


@dataclass
class VLMResult:
    region_id: int
    text: str


# ---------------------------------------------------------------------------
# Worker process
# ---------------------------------------------------------------------------

def _worker_main(in_q: mp.Queue, out_q: mp.Queue, model_id: str) -> None:
    """GOT-OCR inference loop — runs in a dedicated child process."""
    import logging as _log
    _log.basicConfig(level=logging.WARNING)
    log = _log.getLogger(__name__)

    import cv2
    import torch
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor

    from brain_service.preprocessor import preprocess_crop

    device = "mps" if _mps_available() else "cpu"
    log.warning("VLMWorker: loading %s on %s …", model_id, device)

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(device).eval()

    log.warning("VLMWorker: model ready on %s", device)

    while True:
        item = in_q.get()   # block until work arrives
        if item is None:    # shutdown sentinel
            break

        region_id, crop = item
        text = ""
        try:
            enhanced = preprocess_crop(crop)
            rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            inputs = processor(pil_img, return_tensors="pt", format=True).to(device)
            with torch.inference_mode():
                generate_ids = model.generate(
                    **inputs,
                    do_sample=False,
                    tokenizer=processor.tokenizer,
                    stop_strings="<|im_end|>",
                    max_new_tokens=512,
                )
            prompt_len = inputs["input_ids"].shape[1]
            text = processor.decode(
                generate_ids[0, prompt_len:],
                skip_special_tokens=True,
            ).strip()

            log.warning(
                "VLMWorker: region %d → %d chars: %r",
                region_id, len(text), text[:60],
            )
        except Exception:
            log.exception("VLMWorker: inference failed for region %d", region_id)

        try:
            out_q.put_nowait(VLMResult(region_id=region_id, text=text))
        except Exception:
            log.warning("VLMWorker: output queue full, result for region %d dropped", region_id)


def _mps_available() -> bool:
    try:
        import torch
        return torch.backends.mps.is_available()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class VLMWorker:
    """Non-blocking GOT-OCR 2.0 worker.

    submit() enqueues a crop for inference. get_results() drains all
    completed inferences available this frame. Both are non-blocking.
    """

    def __init__(self, model_id: str = _MODEL_ID) -> None:
        self._in_q: mp.Queue = mp.Queue(maxsize=10)
        self._out_q: mp.Queue = mp.Queue(maxsize=30)
        self._worker = mp.Process(
            target=_worker_main,
            args=(self._in_q, self._out_q, model_id),
            daemon=True,
            name="got-ocr",
        )
        self._worker.start()
        logger.info("VLMWorker started (pid=%d)", self._worker.pid)

    def submit(self, region_id: int, crop: np.ndarray) -> None:
        """Enqueue *crop* for VLM inference. Non-blocking; logs if queue full."""
        try:
            self._in_q.put_nowait((region_id, crop))
        except Exception:
            logger.warning(
                "VLMWorker input queue full — region %d dropped. "
                "VLM may be behind; will process on next stable cycle.",
                region_id,
            )

    def get_results(self) -> list[VLMResult]:
        """Drain all completed inferences available right now. Non-blocking."""
        results = []
        while True:
            try:
                results.append(self._out_q.get_nowait())
            except Exception:
                break
        return results

    def shutdown(self) -> None:
        """Signal the worker to stop and wait for clean exit."""
        try:
            self._in_q.put_nowait(None)
        except Exception:
            pass
        self._worker.join(timeout=10)
        if self._worker.is_alive():
            self._worker.terminate()
        logger.info("VLMWorker stopped")
