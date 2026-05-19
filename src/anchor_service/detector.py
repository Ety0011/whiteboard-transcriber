"""Stage 5 — Anchor Discovery (Grounded-DINO).

Detects line-level Spatial Anchors on the clean board composite produced by
Stage 4. Each anchor is classified as TEXT_LINE or MATH_UNIT.

Model: IDEA-Research/grounding-dino-tiny via HuggingFace transformers.
Prompt: "text line . math equation ."

Runs as a dedicated multiprocessing.Process (non-blocking). process() submits
the latest composite via a maxsize=1 queue and returns the most recent
DetectorResult immediately. Stale composites are dropped automatically.
"""

from __future__ import annotations

import enum
import logging
import multiprocessing as mp
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
_TEXT_PROMPT = "text line . math equation ."
_DEFAULT_BOX_THRESH = 0.35
_DEFAULT_TEXT_THRESH = 0.25


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

class AnchorType(enum.Enum):
    TEXT_LINE = "TEXT_LINE"
    MATH_UNIT = "MATH_UNIT"


@dataclass
class Anchor:
    bbox: np.ndarray   # (4,) int32: x1, y1, x2, y2 in rectified 1920×1080 space
    confidence: float
    anchor_type: AnchorType


@dataclass
class DetectorResult:
    anchors: list[Anchor] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Worker process
# ---------------------------------------------------------------------------

def _worker_main(
    in_q: mp.Queue,
    out_q: mp.Queue,
    model_id: str,
    box_thresh: float,
    text_thresh: float,
) -> None:
    """Grounded-DINO inference loop — runs in a dedicated child process."""
    import logging as _log
    _log.basicConfig(level=logging.WARNING)
    log = _log.getLogger(__name__)

    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    device = "mps" if _mps_available() else "cpu"
    log.warning("AnchorDetector: loading %s on %s …", model_id, device)

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    model.eval()
    log.warning("AnchorDetector: model ready")

    while True:
        composite = in_q.get()   # block until work arrives
        if composite is None:    # shutdown sentinel
            break

        anchors: list[Anchor] = []
        try:
            rgb = cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            h, w = composite.shape[:2]

            inputs = processor(
                images=pil_img,
                text=_TEXT_PROMPT,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=box_thresh,
                text_threshold=text_thresh,
                target_sizes=[(h, w)],
            )[0]

            for box, score, label in zip(
                results["boxes"], results["scores"], results["text_labels"]
            ):
                anchor_type = _label_to_type(label)
                if anchor_type is None:
                    continue
                x1, y1, x2, y2 = box.cpu().tolist()
                bbox = np.array(
                    [int(x1), int(y1), int(x2), int(y2)], dtype=np.int32
                )
                anchors.append(Anchor(bbox=bbox, confidence=float(score), anchor_type=anchor_type))

            log.warning(
                "AnchorDetector: %d anchors (%d TEXT, %d MATH)",
                len(anchors),
                sum(1 for a in anchors if a.anchor_type == AnchorType.TEXT_LINE),
                sum(1 for a in anchors if a.anchor_type == AnchorType.MATH_UNIT),
            )

        except Exception:
            log.exception("Grounded-DINO inference failed")

        result = DetectorResult(anchors=anchors)
        try:
            out_q.get_nowait()
        except Exception:
            pass
        try:
            out_q.put_nowait(result)
        except Exception:
            pass


def _label_to_type(label: str) -> AnchorType | None:
    label = label.lower()
    if "math" in label:
        return AnchorType.MATH_UNIT
    if "text" in label:
        return AnchorType.TEXT_LINE
    return None


def _mps_available() -> bool:
    try:
        import torch
        return torch.backends.mps.is_available()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class AnchorDetector:
    """Non-blocking Grounded-DINO anchor detector.

    Spawns a single child process. process() always returns immediately with
    the latest DetectorResult (empty until the first inference completes).
    """

    def __init__(
        self,
        model_id: str = _MODEL_ID,
        box_thresh: float = _DEFAULT_BOX_THRESH,
        text_thresh: float = _DEFAULT_TEXT_THRESH,
    ) -> None:
        self._cached = DetectorResult()
        self._in_q: mp.Queue = mp.Queue(maxsize=1)
        self._out_q: mp.Queue = mp.Queue(maxsize=1)
        self._worker = mp.Process(
            target=_worker_main,
            args=(self._in_q, self._out_q, model_id, box_thresh, text_thresh),
            daemon=True,
            name="grounding-dino",
        )
        self._worker.start()
        logger.info("AnchorDetector worker started (pid=%d)", self._worker.pid)

    def process(self, composite: np.ndarray) -> DetectorResult:
        """Submit composite for async detection; return latest cached result."""
        try:
            self._in_q.put_nowait(composite)
        except Exception:
            pass  # worker busy — drop this composite

        try:
            self._cached = self._out_q.get_nowait()
        except Exception:
            pass  # no new result yet

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
