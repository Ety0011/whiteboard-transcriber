"""Tier 2 — Pipeline Orchestrator.

Coordinates all CV and tracking work in a dedicated thread, keeping the
UI thread free for pure event handling and display at ~30 FPS.

Owns the complete model lifecycle: construction, wait_ready, shutdown, and
post-session finalization. Callers only interact via frame_queue/render_queue
and the three UI-facing read-only properties.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from board import (
    BoardCompositor,
    BoardSegmenter,
    PersonSegmenter,
    Rectifier,
)
from layout import Block, LayoutWorker
from ledger import Ledger
from ocr.worker import TranscriptionWorker
from stage import replace
from tracker import Note, NoteTracker


@dataclass
class PipelineResult:
    """Board-side pipeline output, ready for the UI thread to render.

    All fields are in 1920×1080 rectified space. Runtime state (busy flags,
    person mask, corners) is exposed via orchestrator properties.
    """

    composite: np.ndarray
    blocks: list[Block]
    notes: list[Note]


log = logging.getLogger(__name__)


class PipelineOrchestrator(threading.Thread):
    """Central coordination thread — owns all CV stages and their lifecycle.

    Constructs, starts, and tears down every model worker. The UI thread
    only needs to call wait_ready(), start(), stop(), join(), shutdown(),
    and finalize(). Model internals are invisible to main.py.

    Args:
        frame_queue:  Drop-old queue supplying raw camera frames from Tier 1.
        render_queue: Drop-old queue delivering PipelineResult to Tier 1.
        canvas:       True for mouse-drawable canvas mode — skips Stages 2–5.
        output_dir:   Directory for live.md and lecture_history.md.
    """

    def __init__(
        self,
        frame_queue: queue.Queue[np.ndarray | None],
        render_queue: queue.Queue[PipelineResult | None],
        canvas: bool = False,
        output_dir: Path = Path("output"),
    ) -> None:
        super().__init__(daemon=True, name="pipeline-orchestrator")
        self._frame_queue = frame_queue
        self._render_queue = render_queue
        self._canvas = canvas
        self._output_dir = output_dir
        self._stop = threading.Event()
        self._zero_mask: np.ndarray | None = None

        if not canvas:
            self._board_segmenter: BoardSegmenter | None = BoardSegmenter()
            self._person_segmenter: PersonSegmenter | None = PersonSegmenter()
            self._rectifier: Rectifier | None = Rectifier()
            self._compositor: BoardCompositor | None = BoardCompositor()
        else:
            self._board_segmenter = None
            self._person_segmenter = None
            self._rectifier = None
            self._compositor = None

        self._layout_worker = LayoutWorker()
        self._tracker = NoteTracker()
        self._transcriber = TranscriptionWorker()
        self._ledger = Ledger(output_dir=output_dir)

    # ------------------------------------------------------------------
    # Lifecycle — called from the UI thread
    # ------------------------------------------------------------------

    def wait_ready(self, timeout: float | None = None) -> bool:
        """Load synchronous models and block until all worker subprocesses are ready.

        PersonSegmenter is loaded here (synchronous, ~200ms). Board segmenter,
        layout worker, and transcription worker are waited on in parallel via
        their internal ready events.

        Args:
            timeout: Per-worker timeout in seconds. None waits indefinitely.

        Returns:
            True if all workers became ready within timeout.

        Raises:
            RuntimeError: If any worker subprocess failed to load.
        """
        if self._person_segmenter is not None:
            self._person_segmenter.load()
        workers = [w for w in (
            self._board_segmenter,
            self._person_segmenter,
            self._layout_worker,
            self._transcriber,
        ) if w is not None]
        for worker in workers:
            if not worker.wait_ready(timeout=timeout):
                return False
            log.info("%s ready", type(worker).__name__)
        return True

    def shutdown(self) -> None:
        """Shut down all worker subprocesses. Call after stop() + join()."""
        for worker in (
            self._board_segmenter,
            self._person_segmenter,
            self._layout_worker,
            self._transcriber,
        ):
            if worker is not None:
                worker.shutdown()

    def finalize(self) -> None:
        """Synthesize timelapse output and log session summary.

        Call once after stop() + join() + shutdown(). Writes lecture_history.md
        and logs the final note count.
        """
        all_entries = self._ledger.get_all()
        n_total = len(all_entries)
        n_erased = sum(1 for e in all_entries if e.erased_at is not None)
        log.info(
            "Session complete — %d notes tracked (%d active, %d erased). Output: %s",
            n_total,
            n_total - n_erased,
            n_erased,
            self._output_dir,
        )
        self._ledger.synthesize_timelapse()

    def stop(self) -> None:
        """Signal the orchestrator to exit its run loop."""
        self._stop.set()

    # ------------------------------------------------------------------
    # UI-facing read-only state (read from UI thread, GIL-safe)
    # ------------------------------------------------------------------

    @property
    def person_mask(self) -> np.ndarray | None:
        """Latest person segmentation mask. None in canvas mode or before first run."""
        return self._person_segmenter.cached_mask if self._person_segmenter else None

    @property
    def board_corners(self) -> np.ndarray | None:
        """Cached board quad corners in camera space, or None in canvas mode."""
        return self._rectifier.cached_corners if self._rectifier else None

    @property
    def board_busy(self) -> bool:
        """True while SAM inference is in flight. Always False in canvas mode."""
        return self._board_segmenter.is_busy if self._board_segmenter else False

    @property
    def layout_busy(self) -> bool:
        """True while layout detector subprocess has unprocessed work in flight."""
        return self._layout_worker.is_busy

    # ------------------------------------------------------------------
    # Pipeline loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Pipeline loop — runs until stop() is called or EOS frame received."""
        while not self._stop.is_set():
            try:
                frame = self._frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if frame is None:
                log.info("Orchestrator received end-of-stream signal.")
                break

            if self._canvas:
                # Canvas IS the clean board — skip Stages 2–5 entirely.
                composite = frame
            else:
                # Stage 2: board segmentation (async subprocess, ~5s cadence)
                board_mask = self._board_segmenter.segment(frame)

                # Stage 3: person segmentation (inline ~5ms, ~10 Hz throttled)
                person_mask = self._person_segmenter.segment(frame)
                if person_mask is None:
                    if self._zero_mask is None:
                        self._zero_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    person_mask = self._zero_mask

                # Stage 4: perspective rectification
                rect_frame, rect_mask = self._rectifier.rectify(
                    frame, board_mask, person_mask
                )

                # Stage 5: distance-weighted EMA compositing
                composite = self._compositor.composite(rect_frame, rect_mask)

            # Stages 6+7: text detection + clustering (async subprocess)
            blocks = self._layout_worker.detect(composite)

            # Stage 8: note tracker state machine
            newly_inferring, newly_erased, newly_active = self._tracker.update(
                blocks, composite, self._transcriber.collect()
            )
            self._transcriber.submit(newly_inferring)

            # Stage 10: append-only ledger
            self._ledger.sync(newly_erased, newly_active, composite)

            replace(
                self._render_queue,
                PipelineResult(
                    composite=composite,
                    blocks=blocks,
                    notes=self._tracker.all_notes,
                ),
            )
