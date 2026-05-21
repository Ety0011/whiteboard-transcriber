"""Whiteboard transcription pipeline — entry point.

Usage::

    python src/main.py                           # live webcam, hierarchical_union_find
    python src/main.py video.mp4                 # video file
    python src/main.py --detector yolo video.mp4    # YOLO backend

Keyboard controls:
    q      — quit
    a      — toggle auto/manual Stage 5 detection
    Space  — manual Stage 5 submit (manual mode only)
    w      — toggle Stage 1/2 corner overlay
    p      — toggle Stage 1/2 body-mask overlay
    t      — toggle Stage 5 block overlay
    r      — toggle Stage 6 entity overlay
"""

from __future__ import annotations

import argparse
import logging
from functools import partial
from pathlib import Path

import cv2

import capture
from board_service.board_masker import BoardMasker
from board_service.person_masker import PersonMasker
from board_service.reconstructor import BoardReconstructor
from board_service.rectifier import Rectifier
from discovery import Discovery
from layout_service import (
    Block,
    DBSCANGrouper,
    DocLayoutDetector,
    HDBSCANGrouper,
    PaddleVLDetector,
    StrokeDetector,
    TextBlockDetector,
    UnionFindGrouper,
    XYCutGrouper,
    YOLODetector,
)
from ledger_service import assembly
from ledger_service.registry import LedgerRegistry
from registry import Registry
from renderer import Renderer
from transcriber_service.transcriber import MockTranscriber

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Whiteboard transcription pipeline")
    parser.add_argument(
        "source",
        nargs="?",
        metavar="FILE",
        help="video or image file (omit to use the default webcam)",
    )
    parser.add_argument(
        "--detector",
        choices=[
            "stroke_cluster",
            "yolo",
            "doclayoutv3",
            "paddleocrvl",
            "hierarchical_union_find",
            "dbscan",
            "hdbscan",
            "xycut",
        ],
        default="hierarchical_union_find",
        help="Stage 5 layout detection backend (default: hierarchical_union_find)",
    )
    args = parser.parse_args()

    output_path = Path("output/whiteboard.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame_queue = capture.start(args.source)

    log.info("Loading models …")
    board_masker = BoardMasker()
    person_masker = PersonMasker()
    rectifier = Rectifier()
    reconstructor = BoardReconstructor()

    factories = {
        "stroke_cluster": StrokeDetector,
        "yolo": YOLODetector,
        "doclayoutv3": DocLayoutDetector,
        "paddleocrvl": PaddleVLDetector,
        "hierarchical_union_find": partial(
            TextBlockDetector, strategy=UnionFindGrouper()
        ),
        "dbscan": partial(TextBlockDetector, strategy=DBSCANGrouper()),
        "hdbscan": partial(TextBlockDetector, strategy=HDBSCANGrouper()),
        "xycut": partial(TextBlockDetector, strategy=XYCutGrouper()),
    }

    discovery = Discovery(factory=factories[args.detector])
    registry = Registry()
    transcriber = MockTranscriber()
    ledger = LedgerRegistry()
    renderer = Renderer()
    pending_ocr: dict[int, object] = {}
    log.info("Ready. Model: %s | Press q or Ctrl-C to stop.", args.detector)

    frame_count = 0
    auto_mode = True
    status_msg = f"Detector: {args.detector.upper()} | AUTO"
    last_blocks: list[Block] = []

    try:
        while True:
            frame = frame_queue.get()
            if frame is None:
                log.info("End of stream.")
                break

            frame_count += 1

            # Stage 1 — board mask (SAM, async, ~10s cadence)
            board_mask = board_masker.segment(frame)
            # Stage 2 — person mask (MediaPipe, sync, per-frame)
            person_mask = person_masker.segment(frame)
            # Stage 3+4 — rectify every frame using cached homography
            rect_frame, rect_mask = rectifier.rectify(frame, board_mask, person_mask)
            composite = reconstructor.update(rect_frame, rect_mask)

            # Stage 5 — layout detection (async, non-blocking)
            if auto_mode:
                blocks, latency = discovery.detect(composite)
                if latency:
                    status_msg = (
                        f"Detector: {args.detector.upper()} | AUTO | {latency * 1000:.1f}ms"
                    )
            else:
                blocks, latency = discovery.poll()

            if blocks:
                last_blocks = blocks

            # Stage 6 — entity lifecycle (cross-frame persistence)
            entity_update = registry.tick(blocks, composite)

            # Stage 7 — submit newly dispatched entities to VLM (non-blocking)
            for entity in entity_update.newly_inferring:
                x1, y1, x2, y2 = entity.bbox
                crop = composite[y1:y2, x1:x2]
                if crop.size > 0:
                    pending_ocr[entity.id] = entity
                    transcriber.submit(entity.id, crop)

            # Poll VLM results — update ledger and synthesise output files
            for result in transcriber.get_results():
                entity = pending_ocr.pop(result.entity_id, None)
                if entity is not None:
                    registry.mark_active(entity, result.text, confidence=1.0)
                    ledger.update(entity.id, entity.bbox, result.text)
                    assembly.synthesize(ledger, output_path.parent)
                    log.debug("Ledger written for entity %d", entity.id)

            # Stage 8 — erasure events
            for entity in entity_update.newly_erased:
                ledger.mark_erased(entity.id)
                assembly.synthesize(ledger, output_path.parent)

            # Render
            renderer.render_board(
                composite,
                last_blocks,
                entity_update.entities,
                frame_count,
                auto_mode,
                status_msg,
                discovery.is_busy,
            )
            renderer.render_raw(frame, person_mask, rectifier.cached_corners)

            # Keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                log.info("[q] Quit")
                break
            elif key == ord("a"):
                auto_mode = not auto_mode
                mode = "AUTO" if auto_mode else "MANUAL"
                status_msg = f"Detector: {args.detector.upper()} | {mode}"
                log.info("[a] Mode → %s", mode)
            elif key == ord(" ") and not auto_mode:
                blocks, latency = discovery.detect(composite)
                status_msg = (
                    f"Detector: {args.detector.upper()} | MANUAL | {latency * 1000:.0f}ms"
                )
                log.info("[space] Manual submit | latency=%.0fms", latency * 1000)
            else:
                renderer.handle_key(key)

    except KeyboardInterrupt:
        pass
    finally:
        board_masker.shutdown()
        discovery.shutdown()
        transcriber.shutdown()
        cv2.destroyAllWindows()

    log.info("Shutting down.")


if __name__ == "__main__":
    main()
