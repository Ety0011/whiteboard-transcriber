"""Stage 6 — Content Recognition.

Extracts the textual or structural content from each classified region.
Sub-pipelines run sequentially by default; they can be parallelised with
concurrent.futures.ThreadPoolExecutor for I/O-bound OCR calls.

Sub-pipelines:
    6a  Text OCR (primary):    EasyOCR — handles handwriting and scene text.
                               Reader is created once at startup (3–5 s init).
    6b  Handwriting fallback:  TrOCR-small-handwritten via HuggingFace
                               Transformers. Invoked only when EasyOCR
                               confidence < 0.65 (~10–20 % of lines).
    6c  Diagram vectorization: OpenCV Canny + HoughLinesP + contour analysis
                               → Mermaid flowchart syntax or shape description.
    6d  Table recognition:     HoughLinesP for grid, EasyOCR per cell
                               → Markdown table syntax.

Key libraries: easyocr, transformers (TrOCR), Pillow, OpenCV.
"""

from __future__ import annotations

import numpy as np


def process(composite: np.ndarray, regions: list[dict]) -> list[dict]:
    """Recognise content in each classified region.

    Args:
        composite: BGR uint8 board composite (clean surface from Stage 3).
        regions:   List of region dicts from Stage 5
                   (keys: ``x``, ``y``, ``w``, ``h``, ``hash``, ``label``).

    Returns:
        List of content-block dicts, each with keys:
            ``x``, ``y``, ``w``, ``h`` — bounding box,
            ``label``                  — region type,
            ``content``                — recognised text / Mermaid / Markdown.
    """
    raise NotImplementedError
