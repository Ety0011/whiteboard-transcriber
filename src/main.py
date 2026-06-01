"""Whiteboard transcription pipeline — entry point (Tier 1: UI thread).

Pure pygame event loop at ~30 FPS. Reads frames non-blocking, drops them
into frame_queue for the PipelineOrchestrator, polls render_queue for the
latest overlay surface, and handles all keyboard/mouse input.

Usage::

    python src/main.py                      # live webcam
    python src/main.py video.mp4            # video file
    python src/main.py --output-dir /tmp/lecture video.mp4
    python src/main.py --debug              # verbose logging
    python src/main.py --canvas             # mouse-drawable canvas

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
import pygame

from board import (
    BoardCompositor,
    BoardSegmenter,
    Compositor,
    NullBoardCompositor,
    NullBoardSegmenter,
    NullPersonSegmenter,
    PersonSegmenter,
    Rectifier,
    Segmenter,
)
from capture import CanvasCapture, Capture, FrameSource
from layout import LayoutWorker
from ledger import Ledger
from logging_config import suppress_noise
from ocr.worker import TranscriptionWorker
from orchestrator import PipelineOrchestrator, PipelineResult
from renderer import Renderer
from stage import replace
from tracker import NoteTracker

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tier 1 — UI thread
# ---------------------------------------------------------------------------


def main() -> None:
    suppress_noise()  # sets env vars inherited by all worker subprocesses
    args = _parse_args()

    if args.debug:
        os.environ["LOG_LEVEL"] = "DEBUG"
        logging.getLogger().setLevel(logging.DEBUG)

    cap: FrameSource
    board_segmenter: Segmenter
    person_segmenter: Segmenter

    if args.canvas:
        cap = CanvasCapture().start()
        board_segmenter = NullBoardSegmenter()
        person_segmenter = NullPersonSegmenter()
        rectifier = Rectifier()
        compositor: Compositor = NullBoardCompositor()
    else:
        cap = Capture(args.source).start()
        board_segmenter = BoardSegmenter()
        person_segmenter = PersonSegmenter()
        person_segmenter.load()  # synchronous; runs here before workers start
        rectifier = Rectifier()
        compositor = BoardCompositor()

    log.info("Loading models …")
    layout_worker = LayoutWorker()
    tracker = NoteTracker()
    transcriber = TranscriptionWorker()
    ledger = Ledger(output_dir=args.output_dir)
    renderer = Renderer(display_width=args.display_width)

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
    init_size = _display_size(cap, args.display_width, stack=not args.canvas)
    aspect_ratio = init_size[0] / init_size[1]
    screen = pygame.display.set_mode(init_size, pygame.RESIZABLE)
    pygame.display.set_caption("Lecture Historian")
    clock = pygame.time.Clock()
    fps_font = pygame.font.SysFont("monospace", 18)
    paused = False
    raw_surf: pygame.Surface | None = None
    board_surf: pygame.Surface | None = None

    log.info("Ready. Press q or Ctrl-C to stop.")

    try:
        while True:
            # --- events --------------------------------------------------
            screen, paused = _handle_events(screen, aspect_ratio, cap, renderer, paused)

            # --- raw frame: display at source FPS, also feed orchestrator --
            if not paused:
                frame = cap.try_read()
                if frame is not None:
                    raw_surf = renderer.raw_surface(
                        frame,
                        person_segmenter.cached_mask,
                        rectifier.cached_corners,
                        board_segmenter.is_busy,
                    )
                    replace(frame_queue, frame)
                elif not cap.is_active:
                    log.info("End of stream.")
                    replace(frame_queue, None)
                    break

            # --- board panel: async update from orchestrator -------------
            try:
                result = render_queue.get_nowait()
                board_surf = renderer.board_surface(
                    result.composite,
                    result.blocks,
                    result.notes,
                    layout_worker.is_busy,
                )
            except queue.Empty:
                pass

            # --- display: stack raw (live) above board (async) -----------
            if raw_surf is not None or board_surf is not None:
                w = screen.get_width()
                y = 0
                if not args.canvas and raw_surf is not None:
                    y = _blit_panel(screen, raw_surf, y, w)
                if board_surf is not None:
                    _blit_panel(screen, board_surf, y, w)
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


def _handle_events(
    screen: pygame.Surface,
    aspect_ratio: float,
    cap: FrameSource,
    renderer: Renderer,
    paused: bool,
) -> tuple[pygame.Surface, bool]:
    """Process all pending pygame events for one loop tick.

    Args:
        screen: Current pygame display surface.
        aspect_ratio: Window width/height ratio used to constrain resizes.
        cap: Frame source — receives mouse and pause/resume events.
        renderer: Receives overlay toggle-key events.
        paused: Current pause state.

    Returns:
        Updated (screen, paused) after processing all queued events.

    Raises:
        KeyboardInterrupt: On QUIT event or q keypress.
    """
    sz = screen.get_size()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            raise KeyboardInterrupt
        elif event.type == pygame.VIDEORESIZE:
            new_h = round(event.w / aspect_ratio)
            screen = pygame.display.set_mode((event.w, new_h), pygame.RESIZABLE)
            sz = screen.get_size()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                log.info("[q] Quit")
                raise KeyboardInterrupt
            elif event.key == pygame.K_SPACE:
                paused = not paused
                cap.pause() if paused else cap.resume()
                log.info("[space] %s", "Paused" if paused else "Resumed")
            elif event.key == pygame.K_c:
                cap.clear()
                log.info("[c] Canvas cleared")
            elif event.key == pygame.K_w:
                renderer.show_corners = not renderer.show_corners
                log.info("[w] Corners → %s", "ON" if renderer.show_corners else "OFF")
            elif event.key == pygame.K_p:
                renderer.show_mask = not renderer.show_mask
                log.info("[p] Mask → %s", "ON" if renderer.show_mask else "OFF")
            elif event.key == pygame.K_t:
                renderer.show_blocks = not renderer.show_blocks
                log.info("[t] Blocks → %s", "ON" if renderer.show_blocks else "OFF")
            elif event.key == pygame.K_r:
                renderer.show_tracker = not renderer.show_tracker
                log.info("[r] Entities → %s", "ON" if renderer.show_tracker else "OFF")
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
    return screen, paused


def _blit_panel(
    screen: pygame.Surface,
    surf: pygame.Surface,
    y: int,
    width: int,
) -> int:
    """Blit *surf* scaled to *width* at vertical offset *y*.

    Skips smoothscale when the surface is already the right size (common case —
    renderer produces surfaces at display_width).

    Args:
        screen: Destination display surface.
        surf: Source surface to blit.
        y: Vertical pixel offset.
        width: Target width in pixels.

    Returns:
        y + height of the blitted area (vertical offset for the next panel).
    """
    height = round(surf.get_height() * width / surf.get_width())
    scaled = (
        surf if surf.get_size() == (width, height)
        else pygame.transform.smoothscale(surf, (width, height))
    )
    screen.blit(scaled, (0, y))
    return y + height


def _display_size(
    cap: FrameSource, display_width: int, stack: bool = True
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
    display_height = int((scaled_raw_h + board_h) * display_width / board_w)
    return (display_width, display_height)


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
        "--canvas",
        action="store_true",
        help="Mouse-drawable canvas mode; skips camera, SAM, and EMA",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Set log level to DEBUG (propagates to all worker subprocesses)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
