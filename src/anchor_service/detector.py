"""Stage 5 — Layout Discovery (PaddleOCR-VL-1.5 / MLX Native).

Detects all text regions on the clean board composite via the native
"Spotting:" task of PaddlePaddle/PaddleOCR-VL-1.5. Each detected polygon
is wrapped into a tight axis-aligned LayoutRegion (x1,y1,x2,y2) in the
rectified 1920×1080 coordinate space.

Model: mlx-community/PaddleOCR-VL-1.5-8bit (~1.1GB, bfloat16, Apple Silicon).

Runs as a dedicated multiprocessing.Process (non-blocking). detect() submits
the latest composite via a maxsize=1 queue and returns the most recent
DetectorResult immediately. Stale composites are dropped automatically.
"""

from __future__ import annotations

import enum
import logging
import multiprocessing as mp
import re
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

_MODEL_ID = "mlx-community/PaddleOCR-VL-1.5-8bit"
_IMG_H, _IMG_W = 1080, 1920


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class AnchorType(enum.Enum):
    TEXT_CLUSTER = "TEXT_CLUSTER"
    MATH_BLOCK = "MATH_BLOCK"
    DIAGRAM = "DIAGRAM"


@dataclass
class LayoutRegion:
    bbox: np.ndarray  # (4,) int32: x1, y1, x2, y2 — rectified 1920×1080
    raw_polygon: np.ndarray | None  # (N, 2) int32: absolute polygon vertices
    confidence: float
    anchor_type: AnchorType
    label: str  # canonical label ("TEXT_CLUSTER" etc.)
    text: str  # raw layout text content string


@dataclass
class DetectorResult:
    regions: list[LayoutRegion] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_spotting_output(raw_text: str, h: int, w: int) -> list[LayoutRegion]:
    """Parse coordinate tokens from PaddleOCR-VL-1.5 matching [X, Y] format."""
    regions: list[LayoutRegion] = []
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

            # Direct linear percentage mapping to high-resolution model workspace coordinates
            abs_x = int((token_x / 1000.0) * w)
            abs_y = int((token_y / 1000.0) * h)

            abs_x = max(0, min(w, abs_x))
            abs_y = max(0, min(h, abs_y))
            poly_pts.append([abs_x, abs_y])

        raw_polygon = np.array(poly_pts, dtype=np.int32)

        x1 = int(np.clip(raw_polygon[:, 0].min(), 0, w))
        y1 = int(np.clip(raw_polygon[:, 1].min(), 0, h))
        x2 = int(np.clip(raw_polygon[:, 0].max(), 0, w))
        y2 = int(np.clip(raw_polygon[:, 1].max(), 0, h))

        if x2 <= x1 or y2 <= y1:
            continue

        bbox = np.array([x1, y1, x2, y2], dtype=np.int32)

        if "$$" in content or "\\(" in content or "e^{" in content or "=" in content:
            anchor_type = AnchorType.MATH_BLOCK
        elif len(content) < 3 and not content.isalnum():
            anchor_type = AnchorType.DIAGRAM
        else:
            anchor_type = AnchorType.TEXT_CLUSTER

        regions.append(
            LayoutRegion(
                bbox=bbox,
                raw_polygon=raw_polygon,
                confidence=1.0,
                anchor_type=anchor_type,
                label=anchor_type.value,
                text=content,
            )
        )

    return regions


# ---------------------------------------------------------------------------
# Worker process
# ---------------------------------------------------------------------------


def _worker_main(in_q: mp.Queue, out_q: mp.Queue) -> None:
    """PaddleOCR-VL-1.5 spotting loop — runs natively on Apple Silicon via MLX."""
    import logging as _log

    _log.basicConfig(level=logging.WARNING)
    log = _log.getLogger(__name__)

    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    from PIL import Image

    log.warning("AnchorDetector: Loading native MLX component: %s", _MODEL_ID)
    model, processor = load(_MODEL_ID)
    config = model.config
    log.warning(
        "AnchorDetector: MLX pipeline initialized successfully on Apple Silicon."
    )

    while True:
        composite = in_q.get()
        if composite is None:
            break

        regions: list[LayoutRegion] = []
        try:
            import cv2 as _cv2

            h, w = composite.shape[:2]
            rgb = _cv2.cvtColor(composite, _cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            prompt = "Spotting:"
            formatted_prompt = apply_chat_template(
                processor, config, prompt, num_images=1
            )

            gen_result = generate(
                model,
                processor,
                formatted_prompt,
                pil_img,
                max_tokens=512,
                verbose=False,
            )

            raw_text = (
                gen_result.text if hasattr(gen_result, "text") else str(gen_result)
            )
            raw_text = raw_text.strip()

            regions = parse_spotting_output(raw_text, h, w)
        except Exception:
            log.exception(
                "AnchorDetector: MLX inference pass dropped frame due to failure"
            )

        result = DetectorResult(regions=regions)
        try:
            out_q.get_nowait()
        except Exception:
            pass
        try:
            out_q.put_nowait(result)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class AnchorDetector:
    """Non-blocking PaddleOCR-VL-1.5 visual grounding detector.

    Spawns a single child process. detect() always returns immediately with
    the latest DetectorResult (empty until the first inference completes).
    """

    def __init__(self) -> None:
        self._cached = DetectorResult()
        self._in_q: mp.Queue = mp.Queue(maxsize=1)
        self._out_q: mp.Queue = mp.Queue(maxsize=1)
        self._worker = mp.Process(
            target=_worker_main,
            args=(self._in_q, self._out_q),
            daemon=True,
            name="paddleocr-vl",
        )
        self._worker.start()
        logger.info("AnchorDetector worker started (pid=%d)", self._worker.pid)

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
        logger.info("AnchorDetector worker stopped")
