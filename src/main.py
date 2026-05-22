"""Whiteboard transcription pipeline — entry point.

Usage::

    python src/main.py                                        # live webcam
    python src/main.py video.mp4                             # video file
    python src/main.py --detector hdbscan video.mp4
    python src/main.py --transcriber got video.mp4
    python src/main.py --output-dir /tmp/lecture video.mp4
    python src/main.py --debug                               # verbose logging

Keyboard controls:
    q  — quit
    w  — toggle Stage 1/2 corner overlay
    p  — toggle Stage 1/2 body-mask overlay
    t  — toggle Stage 5 block overlay
    r  — toggle Stage 6 entity overlay
"""

from __future__ import annotations

import argparse
import logging
import os
from functools import partial
from pathlib import Path

import cv2
import numpy as np

import capture
from board_service.board_masker import BoardMasker
from board_service.person_masker import PersonMasker
from board_service.reconstructor import BoardReconstructor
from board_service.rectifier import Rectifier
from discovery import Discovery
from layout_service import (
    AABBTreeGrouper,
    HDBSCANGrouper,
    TextBlockDetector,
    UnionFindGrouper,
)
from ledger import Ledger
from logging_config import suppress_noise
from registry import EntityState, Registry, SemanticEntity
from renderer import Renderer
from transcriber import Transcriber
from transcriber_service import (
    GotTranscriber,
    MockTranscriber,
    PaddleVLTranscriber,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    suppress_noise()
    parser = argparse.ArgumentParser(description="Whiteboard transcription pipeline")
    parser.add_argument(
        "source",
        nargs="?",
        metavar="FILE",
        help="video or image file (omit to use the default webcam)",
    )
    parser.add_argument(
        "--detector",
        choices=["unionfind", "hdbscan", "aabbtree"],
        default="unionfind",
        help="Stage 5 layout detection backend",
    )
    parser.add_argument(
        "--transcriber",
        choices=["mock", "got", "paddlevl"],
        default="paddlevl",
        help="Stage 7 OCR backend",
    )
    parser.add_argument(
        "--display-width",
        type=int,
        default=960,  # half of 1920x1080
        help="Display window width in pixels (default: 1280)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        metavar="DIR",
        help="Directory for live.md and lecture_history.md (default: output/)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Set log level to DEBUG (propagates to all worker subprocesses)",
    )
    args = parser.parse_args()

    if args.debug:
        os.environ["LOG_LEVEL"] = "DEBUG"
        logging.getLogger().setLevel(logging.DEBUG)

    frame_queue = capture.start(args.source)

    log.info("Loading models …")
    board_masker = BoardMasker()
    person_masker = PersonMasker()
    rectifier = Rectifier()
    reconstructor = BoardReconstructor()

    detector_factories = {
        "unionfind": partial(TextBlockDetector, strategy=UnionFindGrouper()),
        "hdbscan": partial(TextBlockDetector, strategy=HDBSCANGrouper()),
        "aabbtree": partial(TextBlockDetector, strategy=AABBTreeGrouper()),
    }

    transcriber_factories = {
        "mock": MockTranscriber,
        "got": GotTranscriber,
        "paddlevl": PaddleVLTranscriber,
    }

    discovery = Discovery(factory=detector_factories[args.detector])
    registry = Registry()
    transcriber = Transcriber(factory=transcriber_factories[args.transcriber])
    ledger = Ledger(output_dir=args.output_dir)
    renderer = Renderer()
    pending_ocr: dict[int, SemanticEntity] = {}
    log.info("Ready. Model: %s | Press q or Ctrl-C to stop.", args.detector)

    try:
        while True:
            frame = frame_queue.get()
            if frame is None:
                log.info("End of stream.")
                break

            # Stage 1 — board mask (SAM, async, ~10s cadence)
            board_mask = board_masker.segment(frame)
            # Stage 2 — person mask (MediaPipe, sync, per-frame)
            person_mask = person_masker.segment(frame)
            # Stage 3+4 — rectify every frame using cached homography
            rect_frame, rect_mask = rectifier.rectify(frame, board_mask, person_mask)
            composite = reconstructor.update(rect_frame, rect_mask)

            # Stage 5 — layout detection (async, non-blocking)
            blocks = discovery.detect(composite)

            # Stage 6 — entity lifecycle (cross-frame persistence)
            entity_update = registry.tick(blocks, composite.shape[:2])

            # Stage 7 — submit newly dispatched entities to VLM (non-blocking)
            for entity in entity_update.newly_inferring:
                x1, y1, x2, y2 = entity.bbox
                crop = composite[y1:y2, x1:x2]
                if crop.size > 0:
                    pending_ocr[entity.id] = entity
                    transcriber.submit(entity.id, crop)
                else:
                    registry.reset_to_stabilizing(entity)

            # Poll VLM results — update ledger and synthesise output files
            for result in transcriber.get_results():
                entity = pending_ocr.pop(result.entity_id, None)
                if entity is not None and entity.state == EntityState.INFERRING:
                    registry.mark_active(entity, result.text)
                    ledger.update(entity.id, entity.bbox, result.text)

            # Stage 8 — erasure events
            for entity in entity_update.newly_erased:
                pending_ocr.pop(entity.id, None)
                ledger.mark_erased(entity.id)

            # Render — stack raw (top) above composite (bottom) in one window
            board = renderer.render_board(
                composite,
                blocks,
                entity_update.entities,
                discovery.is_busy,
            )
            raw = renderer.render_raw(
                frame,
                person_mask,
                rectifier.cached_corners,
                board_masker.is_busy,
            )
            if raw.shape[1] != board.shape[1]:
                scale = board.shape[1] / raw.shape[1]
                raw = cv2.resize(raw, (board.shape[1], int(raw.shape[0] * scale)))
            combined = np.vstack([raw, board])
            h, w = combined.shape[:2]
            target_w = args.display_width
            target_h = int(h * target_w / w)
            combined = cv2.resize(combined, (target_w, target_h))
            cv2.imshow("Lecture Historian", combined)

            # Keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                log.info("[q] Quit")
                break
            else:
                renderer.handle_key(key)

    except KeyboardInterrupt:
        pass
    finally:
        board_masker.shutdown()
        discovery.shutdown()
        transcriber.shutdown()
        cv2.destroyAllWindows()

    all_entries = ledger.get_all()
    n_total = len(all_entries)
    n_erased = sum(1 for e in all_entries if e.erased_at is not None)
    log.info(
        "Session complete — %d entities tracked (%d active, %d erased). Output: %s",
        n_total,
        n_total - n_erased,
        n_erased,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
