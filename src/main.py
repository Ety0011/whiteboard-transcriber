"""Whiteboard transcription pipeline — entry point (Tier 1: UI thread).

Pure pygame event loop at ~30 FPS. Reads frames non-blocking, drops them
into frame_queue for the PipelineOrchestrator, polls render_queue for the
latest overlay matrix, and handles all keyboard/mouse input.

Usage::

    python src/main.py                      # live webcam
    python src/main.py video.mp4            # video file
    python src/main.py --output-dir /tmp/lecture video.mp4
    python src/main.py --debug              # verbose logging
    python src/main.py --demo               # mouse-drawable canvas

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
import queue
from pathlib import Path

import numpy as np

from board import (
    BoardCompositor,
    BoardSegmenter,
    NullBoardCompositor,
    NullBoardSegmenter,
    NullPersonSegmenter,
    PersonSegmenterWorker,
    Rectifier,
)
from capture import CanvasCapture, Capture
from layout import LayoutWorker
from ledger import Ledger
from logging_config import suppress_noise
from ocr import TranscriptionWorker
from orchestrator import PipelineOrchestrator, PipelineResult, _drop_put
from renderer import Renderer
from tracker import NoteTracker

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tier 1 — UI thread
# ---------------------------------------------------------------------------


def main() -> None:
    suppress_noise()  # sets env vars inherited by all worker subprocesses
    import pygame  # after suppress_noise — env vars in place before pygame loads

    args = _parse_args()

    if args.debug:
        os.environ["LOG_LEVEL"] = "DEBUG"
        logging.getLogger().setLevel(logging.DEBUG)

    if args.demo:
        cap: Capture | CanvasCapture = CanvasCapture().start()
        board_segmenter: BoardSegmenter | NullBoardSegmenter = NullBoardSegmenter()
        person_segmenter: PersonSegmenterWorker | NullPersonSegmenter = (
            NullPersonSegmenter()
        )
        rectifier = Rectifier()
        compositor: BoardCompositor | NullBoardCompositor = NullBoardCompositor()
    else:
        cap = Capture(args.source).start()
        board_segmenter = BoardSegmenter()
        person_segmenter = PersonSegmenterWorker()
        rectifier = Rectifier()
        compositor = BoardCompositor()

    log.info("Loading models …")
    layout_worker = LayoutWorker()
    tracker = NoteTracker()
    transcriber = TranscriptionWorker()
    ledger = Ledger(output_dir=args.output_dir)
    renderer = Renderer(display_width=args.display_width, stack=not args.demo)

    for worker in (board_segmenter, person_segmenter, layout_worker, transcriber):
        worker.wait_ready()
        log.info("%s ready", type(worker).__name__)
    log.info("All workers ready.")

    frame_queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=1)
    render_queue: queue.Queue[PipelineResult | None] = queue.Queue(maxsize=1)

    orchestrator = PipelineOrchestrator(
        frame_queue=frame_queue,
        render_queue=render_queue,
        board_segmenter=board_segmenter,
        person_segmenter=person_segmenter,
        rectifier=rectifier,
        compositor=compositor,
        layout_worker=layout_worker,
        tracker=tracker,
        transcriber=transcriber,
        ledger=ledger,
    )
    orchestrator.start()

    pygame.init()
    init_size = _display_size(cap, args.display_width, stack=not args.demo)
    aspect_ratio = init_size[0] / init_size[1]
    screen = pygame.display.set_mode(init_size, pygame.RESIZABLE)
    pygame.display.set_caption("Lecture Historian")
    clock = pygame.time.Clock()
    fps_font = pygame.font.SysFont("monospace", 18)
    paused = False
    last_raw_surface: pygame.Surface | None = None
    last_board_surface: pygame.Surface | None = None

    log.info("Ready. Press q or Ctrl-C to stop.")

    try:
        while True:
            # --- events --------------------------------------------------
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
                elif event.type == pygame.VIDEORESIZE:
                    new_h = round(event.w / aspect_ratio)
                    screen = pygame.display.set_mode(
                        (event.w, new_h), pygame.RESIZABLE
                    )
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        log.info("[q] Quit")
                        raise KeyboardInterrupt
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                        cap.pause() if paused else cap.resume()
                        log.info("[space] %s", "Paused" if paused else "Resumed")
                    elif event.key == ord("c"):
                        cap.clear()
                        log.info("[c] Canvas cleared")
                    else:
                        renderer.handle_key(event.key)
                sz = screen.get_size()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        cap.on_mouse_down(event.pos, sz)
                    elif event.button == 3:
                        cap.on_eraser_down(event.pos, sz)
                elif event.type == pygame.MOUSEMOTION:
                    if event.buttons[0]:
                        cap.on_mouse_move(event.pos, sz)
                    elif event.buttons[2]:
                        cap.on_eraser_move(event.pos, sz)
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        cap.on_mouse_up()
                    elif event.button == 3:
                        cap.on_eraser_up()

            # --- raw frame: display at source FPS, also feed orchestrator --
            if not paused:
                try:
                    frame = cap.try_read()
                    if frame is None:
                        log.info("End of stream.")
                        _drop_put(frame_queue, None)
                        break
                    last_raw_surface = pygame.surfarray.make_surface(
                        renderer.render_raw_panel(frame)
                    )
                    _drop_put(frame_queue, frame)
                except queue.Empty:
                    pass

            # --- board panel: async update from orchestrator -------------
            try:
                result = render_queue.get_nowait()
                last_board_surface = pygame.surfarray.make_surface(
                    renderer.render_board_panel(
                        result.composite, result.blocks, result.notes, result.layout_busy
                    )
                )
            except queue.Empty:
                pass

            # --- display: stack raw (live) above board (async) -----------
            w = screen.get_width()
            if not args.demo and last_raw_surface is not None:
                raw_aspect = (
                    last_raw_surface.get_height() / last_raw_surface.get_width()
                )
                raw_h = round(w * raw_aspect)
                screen.blit(
                    pygame.transform.smoothscale(last_raw_surface, (w, raw_h)),
                    (0, 0),
                )
                board_y = raw_h
            else:
                board_y = 0
            if last_board_surface is not None:
                board_aspect = (
                    last_board_surface.get_height() / last_board_surface.get_width()
                )
                board_h = round(w * board_aspect)
                screen.blit(
                    pygame.transform.smoothscale(last_board_surface, (w, board_h)),
                    (0, board_y),
                )
            if last_raw_surface is not None or last_board_surface is not None:
                fps_surf = fps_font.render(
                    f"{clock.get_fps():.1f} fps", True, (0, 255, 0)
                )
                screen.blit(fps_surf, (10, 10))
                pygame.display.flip()

            clock.tick(60)

    except KeyboardInterrupt:
        pass
    finally:
        orchestrator.stop()
        orchestrator.join(timeout=5.0)
        cap.stop()
        board_segmenter.shutdown()
        person_segmenter.shutdown()
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
    ledger.synthesize_timelapse()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _display_size(
    cap: Capture | CanvasCapture, display_width: int, stack: bool = True
) -> tuple[int, int]:
    """Compute the pygame window size from source metadata and display width.

    When stack=False only the 1920×1080 board panel is shown. When stack=True
    the raw frame is stacked above the composite.
    """
    board_w, board_h = 1920, 1080
    if not stack:
        return (display_width, display_width * board_h // board_w)
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
        "--demo",
        action="store_true",
        help="Demo mode: mouse-drawable canvas, no camera/video required",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Set log level to DEBUG (propagates to all worker subprocesses)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
