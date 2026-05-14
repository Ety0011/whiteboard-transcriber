"""Stage 6 — Text Recognition.

OCRs newly-stable regions using PaddleOCR PP-OCRv5_rec_server. Diffs text on
re-stabilization and patches the persistent WhiteboardDoc. Upstream stages decide
which regions are text; this module OCRs all regions it receives unconditionally.
"""

from __future__ import annotations

import dataclasses
import difflib
import logging
import time

import numpy as np
from paddleocr import TextRecognition

from src.tracker import Region, TrackerResult

log = logging.getLogger(__name__)


@dataclasses.dataclass
class WhiteboardDoc:
    """Persistent Markdown document for the whiteboard session.

    blocks maps region_id to the current Markdown text for that region.
    Erased regions are wrapped in Markdown strikethrough to preserve history.
    """

    blocks: dict[int, str] = dataclasses.field(default_factory=dict)


class Recognizer:
    """Loads OCR model once and processes newly-stable regions each frame."""

    def __init__(self) -> None:
        """Load PP-OCRv5_rec_server. Blocks for a few seconds on first run."""
        log.info("Loading PP-OCRv5_rec_server …")
        self._recognizer = TextRecognition(model_name="PP-OCRv5_rec_server")
        log.info("PP-OCRv5_rec_server loaded.")

    def _extract_line_crops(self, region: Region) -> list[np.ndarray]:
        """Return individual line crops from the region's stable crop.

        Uses stored line_bboxes (in board coordinate space) to slice
        sub-images from last_stable_crop. Falls back to the full crop
        when no line bboxes are available.

        Args:
            region: A Region with last_stable_crop set.

        Returns:
            List of BGR uint8 line images, sorted top-to-bottom.
        """
        if region.last_stable_crop is None:
            return []

        if not region.line_bboxes:
            return [region.last_stable_crop]

        ox, oy = int(region.bbox[0]), int(region.bbox[1])
        h, w = region.last_stable_crop.shape[:2]

        crops: list[tuple[int, np.ndarray]] = []
        for bbox in region.line_bboxes:
            x1 = max(0, int(bbox[0]) - ox)
            y1 = max(0, int(bbox[1]) - oy)
            x2 = min(w, int(bbox[2]) - ox)
            y2 = min(h, int(bbox[3]) - oy)
            if x2 > x1 and y2 > y1:
                crops.append((y1, region.last_stable_crop[y1:y2, x1:x2]))

        if not crops:
            return [region.last_stable_crop]

        crops.sort(key=lambda t: t[0])
        return [c for _, c in crops]

    def _ocr_lines(self, line_crops: list[np.ndarray]) -> tuple[str, float]:
        """Run TextRecognition on a list of line images.

        Args:
            line_crops: BGR uint8 images, one per text line.

        Returns:
            Tuple of (joined text, mean confidence score).
        """
        results = self._recognizer.predict(line_crops)
        texts: list[str] = []
        scores: list[float] = []
        for item in results:
            text = item.get("rec_text", "").strip()
            score = float(item.get("rec_score", 0.0))
            if text:
                texts.append(text)
                scores.append(score)
        joined = "\n".join(texts)
        mean_score = sum(scores) / len(scores) if scores else 0.0
        return joined, mean_score

    def _ocr_region(self, region: Region) -> tuple[str, float]:
        """Extract text from a stable region crop.

        Args:
            region: A Region in STABLE state with last_stable_crop set.

        Returns:
            Tuple of (recognized text, mean confidence). Empty string and 0.0
            when the crop is missing or yields no text lines.
        """
        if region.last_stable_crop is None:
            return "", 0.0
        line_crops = self._extract_line_crops(region)
        if not line_crops:
            return "", 0.0
        return self._ocr_lines(line_crops)

    def process(
        self,
        tracker_result: TrackerResult,
        doc: WhiteboardDoc,
    ) -> WhiteboardDoc:
        """OCR newly-stable regions and patch the WhiteboardDoc.

        For first stabilization: inserts a new Markdown block.
        For re-stabilization: diffs against cached text; skips update when
        content is identical, otherwise replaces the block.
        For erased regions: wraps the existing block in strikethrough.

        Mutates region.ocr_text, region.ocr_confidence, region.last_modified
        directly on the Region objects (shared references with the tracker).

        Args:
            tracker_result: Output of RegionTracker.process() for this frame.
            doc:            Persistent document to patch in-place.

        Returns:
            The same doc object, mutated.
        """
        for region in tracker_result.newly_erased:
            if region.id in doc.blocks:
                doc.blocks[region.id] = f"~~{doc.blocks[region.id]}~~"
                log.debug("Region %d erased from doc.", region.id)

        for region in tracker_result.newly_stable:
            new_text, confidence = self._ocr_region(region)

            if region.ocr_text is not None:
                diff = list(
                    difflib.unified_diff(
                        region.ocr_text.splitlines(),
                        new_text.splitlines(),
                        lineterm="",
                    )
                )
                if not diff:
                    log.debug("Region %d re-stabilized, content unchanged.", region.id)
                    continue
                log.debug(
                    "Region %d re-stabilized, diff:\n%s",
                    region.id,
                    "\n".join(diff),
                )

            doc.blocks[region.id] = new_text
            region.ocr_text = new_text
            region.ocr_confidence = confidence
            region.last_modified = time.monotonic()
            log.debug(
                "Region %d OCR'd (conf=%.2f): %r",
                region.id,
                confidence,
                new_text[:60],
            )

        return doc
