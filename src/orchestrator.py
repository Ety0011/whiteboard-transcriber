"""Tier 2 — Pipeline Orchestrator.

Coordinates all CV and tracking work in a dedicated thread, keeping the
UI thread free for pure event handling and display at ~30 FPS.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass

import numpy as np

from board import (
    Compositor,
    Rectifier,
    Segmenter,
)
from layout import Block, LayoutWorker
from ledger import Ledger
from ocr.worker import TranscriptionWorker
from stage import drop_put
from tracker import Note, NoteTracker


@dataclass
class PipelineResult:
    """Board-side pipeline output, ready for the UI thread to render.

    composite, blocks, and notes are in 1920×1080 rectified space.
    person_mask and cached_corners are in raw-camera space — used for
    the raw-panel overlays in the UI thread.
    """

    composite: np.ndarray
    blocks: list[Block]
    notes: list[Note]
    layout_busy: bool
    person_mask: np.ndarray          # raw-camera-space; zeros when no person detected
    cached_corners: np.ndarray | None  # None before first SAM result
    sam_busy: bool

log = logging.getLogger(__name__)


class PipelineOrchestrator(threading.Thread):
    """Central coordination thread for all CV and tracking work.

    Receives raw frames from the UI thread via frame_queue, runs Stages 2–10,
    and pushes a PipelineResult to render_queue for the UI thread to render.
    All blocking model inference happens inside isolated subprocesses; this
    thread only coordinates non-blocking queue exchanges and fast synchronous
    transforms (rectification, compositing, clustering).

    Args:
        frame_queue:      Drop-old queue supplying raw camera frames from Tier 1.
        render_queue:     Drop-old queue delivering PipelineResult to Tier 1.
        board_segmenter:  Stage 2 — SAM 3.1 async subprocess.
        person_segmenter: Stage 3 — MediaPipe inline, ~10 Hz (or null for demo).
        rectifier:        Stage 4 — perspective warp (synchronous, fast).
        compositor:       Stage 5 — distance-weighted EMA (synchronous).
        layout_worker:    Stages 6+7 worker (async subprocess).
        tracker:          Stage 8 — note lifecycle state machine (synchronous).
        transcriber:      Stage 9 worker (async subprocess).
        ledger:           Stage 10 — append-only record (synchronous).
    """

    def __init__(
        self,
        frame_queue: queue.Queue[np.ndarray | None],
        render_queue: queue.Queue[PipelineResult | None],
        board_segmenter: Segmenter,
        person_segmenter: Segmenter,
        rectifier: Rectifier,
        compositor: Compositor,
        layout_worker: LayoutWorker,
        tracker: NoteTracker,
        transcriber: TranscriptionWorker,
        ledger: Ledger,
    ) -> None:
        super().__init__(daemon=True, name="pipeline-orchestrator")
        self._frame_queue = frame_queue
        self._render_queue = render_queue
        self._board_segmenter = board_segmenter
        self._person_segmenter = person_segmenter
        self._rectifier = rectifier
        self._compositor = compositor
        self._layout_worker = layout_worker
        self._tracker = tracker
        self._transcriber = transcriber
        self._ledger = ledger

        self._stop = threading.Event()
        self._zero_mask: np.ndarray | None = None

    def stop(self) -> None:
        """Signal the orchestrator to exit its run loop."""
        self._stop.set()

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

            # Stage 2: board segmentation (async subprocess, ~5s cadence)
            board_mask = self._board_segmenter.segment(frame)

            # Stage 3: person segmentation (inline ~5ms, ~10 Hz throttled)
            # PersonSegmenter.segment() self-caches: returns cached mask on
            # sub-interval ticks, new mask when due. None only before first run.
            person_mask = self._person_segmenter.segment(frame)
            if person_mask is None:
                if self._zero_mask is None:
                    self._zero_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                person_mask = self._zero_mask

            # Stage 4: perspective rectification (fast sync, uses cached homography)
            rect_frame, rect_mask = self._rectifier.rectify(frame, board_mask, person_mask)

            # Stage 5: distance-weighted EMA compositing (sync)
            composite = self._compositor.composite(rect_frame, rect_mask)

            # Stages 6+7: text detection + clustering (async subprocess)
            blocks = self._layout_worker.detect(composite)

            # Stage 8: note tracker state machine (sync)
            newly_inferring, newly_erased, newly_active = self._tracker.update(
                blocks, composite, self._transcriber.collect()
            )
            self._transcriber.submit(newly_inferring)

            # Stage 10: append-only ledger (sync, atomic file writes)
            self._ledger.sync(newly_erased, newly_active, composite)

            # Push board result to UI thread for rendering
            drop_put(
                self._render_queue,
                PipelineResult(
                    composite=composite,
                    blocks=blocks,
                    notes=self._tracker.all_notes,
                    layout_busy=self._layout_worker.is_busy,
                    person_mask=person_mask,
                    cached_corners=self._rectifier.cached_corners,
                    sam_busy=self._board_segmenter.is_busy,
                ),
            )


