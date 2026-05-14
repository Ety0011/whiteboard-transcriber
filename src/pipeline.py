"""Pipeline orchestrator — Stages 1–7.

Owns all long-lived model state. Chains stages sequentially on each frame
and writes the Markdown output to disk atomically after OCR updates.

Typical usage::

    pipeline = Pipeline(Path("output/whiteboard.md"))
    # each frame (called from the processing thread):
    pipeline.process(frame)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from board_detector import BoardDetector
from board_reconstructor import BoardReconstructor
from document import WhiteboardDoc
from layout import LayoutRegion
from person_masker import PersonMasker
from rectifier import Rectifier
from text_detector import TextDetector
from text_recognizer import TextRecognizer
from tracker import Detection, RegionTracker

log = logging.getLogger(__name__)


class Pipeline:
    """Chains Stages 1–7 and maintains all stateful model instances.

    Create once at startup — model loading blocks here. Call ``process()``
    in the processing thread for each new frame.
    """

    def __init__(self, output_path: Path) -> None:
        """Instantiate all models and state. Blocks until models are loaded.

        Args:
            output_path: Destination path for the Markdown file. Parent
                directory must exist. The file is written atomically via a
                sibling ``.tmp`` file.
        """
        self._output_path = output_path

        log.info("Loading pipeline models …")
        self._board_detector = BoardDetector()
        self._person_masker = PersonMasker()
        self._rectifier = Rectifier()
        self._reconstructor = BoardReconstructor()
        self._text_detector = TextDetector()
        self._tracker = RegionTracker()
        self._recognizer = TextRecognizer()
        self._doc = WhiteboardDoc()
        log.info("Pipeline ready.")

    def process(self, frame: np.ndarray) -> None:
        """Run one full pipeline cycle on a single BGR uint8 frame.

        Stages 1–6 run every call. Stage 7 (OCR) runs only when the tracker
        reports newly-stable regions. The Markdown file is written atomically
        after each OCR update.

        Args:
            frame: Latest BGR uint8 frame from the camera queue.
        """
        # Stage 1: board detection (non-blocking, returns cached corners)
        corners = self._board_detector.process(frame)

        # Stage 2: person masking on raw frame
        mask = self._person_masker.process(frame)

        # Stage 3: perspective warp of frame + mask
        warped, warped_mask = self._rectifier.process(frame, mask, corners)

        # Stage 4: distance-weighted EMA board model
        composite = self._reconstructor.process(warped, warped_mask)

        # Stage 5: text detection — treat the full board as one text region
        h, w = composite.shape[:2]
        full_region = LayoutRegion(
            bbox=np.array([0, 0, w, h], dtype=np.int32),
            label="text",
            confidence=1.0,
            crop=composite,
        )
        regions_with_lines = self._text_detector.process([full_region])

        detections = [
            Detection(
                bbox=line.bbox,
                confidence=line.confidence,
                line_bboxes=[line.bbox],
            )
            for region in regions_with_lines
            for line in region.lines
        ]

        # Stage 6: region tracker
        tracker_result = self._tracker.process(detections, composite)

        if not tracker_result.newly_stable:
            return

        # Stage 7: OCR on newly-stable regions
        self._recognizer.process(tracker_result, self._tracker, self._doc)

        # Atomic write: .tmp → output_path
        tmp = self._output_path.with_suffix(".tmp")
        tmp.write_text(self._doc.to_markdown(), encoding="utf-8")
        tmp.rename(self._output_path)
        log.debug("Markdown written to %s", self._output_path)
