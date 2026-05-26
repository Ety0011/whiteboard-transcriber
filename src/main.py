"""Whiteboard transcription pipeline — entry point.

Usage::

    python src/main.py                                        # live webcam
    python src/main.py video.mp4                             # video file
    python src/main.py --detector hdbscan video.mp4
    python src/main.py --transcriber got video.mp4
    python src/main.py --output-dir /tmp/lecture video.mp4
    python src/main.py --debug                               # verbose logging

Keyboard controls:
    q      — quit
    space  — pause / resume
    w      — toggle Stage 4 corner overlay
    p      — toggle Stage 3 body-mask overlay
    t      — toggle Stage 7 block overlay
    r      — toggle Stage 8 note overlay
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from functools import partial
from pathlib import Path

from board.board_masker import BoardMasker
from board.person_masker import PersonMasker
from board.reconstructor import BoardReconstructor
from board.rectifier import Rectifier
from capture import Capture
from layout import (
    AABBTreeClusterer,
    BlockDetector,
    HDBSCANClusterer,
    SingleLinkageClusterer,
    UnionFindClusterer,
)
from layout.worker import LayoutWorker
from ledger import Ledger
from logging_config import suppress_noise
from ocr import GotTranscriber, MockTranscriber, PaddleVLTranscriber
from ocr.worker import TranscriptionWorker
from renderer import Renderer
from tracker import NoteTracker

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

_DETECTOR_FACTORIES = {
    "unionfind": partial(BlockDetector, strategy=UnionFindClusterer()),
    "hdbscan": partial(BlockDetector, strategy=HDBSCANClusterer()),
    "aabbtree": partial(BlockDetector, strategy=AABBTreeClusterer()),
    "singlelinkage": partial(BlockDetector, strategy=SingleLinkageClusterer()),
}

_TRANSCRIBER_FACTORIES = {
    "mock": MockTranscriber,
    "got": GotTranscriber,
    "paddlevl": PaddleVLTranscriber,
}


# TODO: add revisions label "pill" in video
# TODO: make all stages async
def main() -> None:
    suppress_noise()  # sets env vars inherited by all worker subprocesses
    import pygame  # after suppress_noise — env vars in place before pygame loads

    args = _parse_args()

    if args.debug:
        os.environ["LOG_LEVEL"] = "DEBUG"
        logging.getLogger().setLevel(logging.DEBUG)

    cap = Capture(args.source).start()

    log.info("Loading models …")
    board_masker = BoardMasker()
    person_masker = PersonMasker()
    rectifier = Rectifier()
    reconstructor = BoardReconstructor()
    layout_worker = LayoutWorker(factory=_DETECTOR_FACTORIES[args.detector])
    tracker = NoteTracker()
    transcriber = TranscriptionWorker(factory=_TRANSCRIBER_FACTORIES[args.transcriber])
    ledger = Ledger(output_dir=args.output_dir)
    renderer = Renderer(display_width=args.display_width)

    for worker in (board_masker, layout_worker, transcriber):
        worker.wait_ready()
        log.info("%s ready", type(worker).__name__)
    log.info("All workers ready.")

    pygame.init()
    screen = pygame.display.set_mode(
        _display_size(cap, args.display_width), pygame.RESIZABLE
    )
    pygame.display.set_caption("Lecture Historian")
    clock = pygame.time.Clock()
    paused = False
    fps = 0.0
    last_t = time.monotonic()

    log.info("Ready. Detector: %s | Press q or Ctrl-C to stop.", args.detector)

    try:
        while True:
            # --- events ------------------------------------------------------
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        log.info("[q] Quit")
                        raise KeyboardInterrupt
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                        cap.pause() if paused else cap.resume()
                        log.info("[space] %s", "Paused" if paused else "Resumed")
                    else:
                        renderer.handle_key(event.key)

            if paused:
                clock.tick(30)
                continue

            # --- frame -------------------------------------------------------
            frame = cap.read()
            if frame is None:
                log.info("End of stream.")
                break

            now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(now - last_t, 1e-6)
            last_t = now

            # --- pipeline ----------------------------------------------------
            board_mask = board_masker.segment(frame)           # Stage 2
            person_mask = person_masker.segment(frame)         # Stage 3
            rect_frame, rect_mask = rectifier.rectify(         # Stage 4
                frame, board_mask, person_mask
            )
            composite = reconstructor.reconstruct(             # Stage 5
                rect_frame, rect_mask
            )
            blocks = layout_worker.detect(composite)           # Stage 6 + 7
            newly_inferring, newly_erased, newly_active = tracker.update(  # Stage 8
                blocks, composite, transcriber.collect()       # Stage 9
            )
            transcriber.submit(newly_inferring)
            ledger.sync(newly_erased, newly_active)            # Stage 10

            # --- display -----------------------------------------------------
            display_frame = renderer.render(
                composite, blocks, tracker.notes, layout_worker.is_busy,
                frame, person_mask, rectifier.cached_corners, board_masker.is_busy,
                fps,
            )
            # numpy → pygame pixel buffer
            surface = pygame.surfarray.make_surface(display_frame)
            screen.blit(surface, (0, 0))  # paint onto back buffer
            pygame.display.flip()         # swap back→front (show)

    except KeyboardInterrupt:
        pass
    finally:
        cap.stop()
        board_masker.shutdown()
        layout_worker.shutdown()
        transcriber.shutdown()
        pygame.quit()

    all_entries = ledger.get_all()
    n_total = len(all_entries)
    n_erased = sum(1 for e in all_entries if e.erased_at is not None)
    log.info(
        "Session complete — %d notes tracked (%d active, %d erased). Output: %s",
        n_total,
        n_total - n_erased,
        n_erased,
        args.output_dir,
    )


def _display_size(cap: Capture, display_width: int) -> tuple[int, int]:
    """Compute the pygame window size from source metadata and display width.

    The renderer stacks the raw frame (rescaled to board width) above the
    1920×1080 board composite, then scales the combined image to display_width.
    """
    board_w, board_h = 1920, 1080
    raw_w, raw_h = cap.frame_size or (board_w, board_h)
    scaled_raw_h = int(raw_h * board_w / raw_w)
    display_h = int((scaled_raw_h + board_h) * display_width / board_w)
    return (display_width, display_h)


def _parse_args() -> argparse.Namespace:
    """Parse and return CLI arguments."""
    parser = argparse.ArgumentParser(description="Whiteboard transcription pipeline")
    parser.add_argument(
        "source",
        nargs="?",
        metavar="FILE",
        help="video or image file (omit to use the default webcam)",
    )
    parser.add_argument(
        "--detector",
        choices=list(_DETECTOR_FACTORIES),
        default="singlelinkage",
        help="Stage 7 block grouping strategy (default: singlelinkage)",
    )
    parser.add_argument(
        "--transcriber",
        choices=list(_TRANSCRIBER_FACTORIES),
        default="paddlevl",
        help="Stage 9 OCR backend (default: paddlevl)",
    )
    parser.add_argument(
        "--display-width",
        type=int,
        default=960,
        help="Display window width in pixels (default: 960)",
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
    return parser.parse_args()


if __name__ == "__main__":
    main()
