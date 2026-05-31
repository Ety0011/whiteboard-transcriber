"""Demo canvas — mouse-drawable whiteboard with the same API as Capture."""

from __future__ import annotations

import queue
import threading
import time

import cv2
import numpy as np

from .source import FrameSource


class CanvasCapture(FrameSource):
    """Drawable 1920×1080 white canvas that publishes frames at ~30fps.

    Provides the same interface as :class:`~capture.Capture` so it can replace
    it in the main loop without pipeline changes.

    Use on_mouse_down/on_mouse_move/on_mouse_up to paint pen strokes;
    on_eraser_down/on_eraser_move/on_eraser_up to erase; clear() to reset.
    """

    _PEN_COLOR: tuple[int, int, int] = (0, 0, 0)
    _PEN_RADIUS: int = 8
    _ERASER_RADIUS: int = 20

    def __init__(self) -> None:
        self.fps: float | None = 30.0
        self.frame_size: tuple[int, int] = (1920, 1080)

        self._canvas = np.full((1080, 1920, 3), 255, dtype=np.uint8)
        self._lock = threading.Lock()
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=1)
        self._active = False
        self._thread: threading.Thread | None = None
        self._last_pos: tuple[int, int] | None = None
        self._last_eraser_pos: tuple[int, int] | None = None

    # ------------------------------------------------------------------
    # FrameSource implementation
    # ------------------------------------------------------------------

    def start(self) -> CanvasCapture:
        """Spawn the frame-publish thread.

        Returns:
            self for fluent chaining.
        """
        self._active = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="canvas-capture"
        )
        self._thread.start()
        return self

    def try_read(self) -> np.ndarray | None:
        """Non-blocking read.

        Returns:
            BGR uint8 frame, or None if no frame is ready this tick or
            on end-of-stream. Never raises.
        """
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self) -> None:
        """Signal the publish thread to exit and unblock any pending reads."""
        self._active = False
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

    def pause(self) -> None:
        """No-op — canvas is always live."""

    def resume(self) -> None:
        """No-op — canvas is always live."""

    # ------------------------------------------------------------------
    # Drawing API
    # ------------------------------------------------------------------

    def on_mouse_down(
        self, display_pos: tuple[int, int], display_size: tuple[int, int]
    ) -> None:
        """Begin a pen stroke at *display_pos*."""
        self._last_pos = None  # discard stale endpoint from any previous stroke
        self._paint(display_pos, display_size, self._PEN_COLOR, self._PEN_RADIUS, pen=True)

    def on_mouse_move(
        self, display_pos: tuple[int, int], display_size: tuple[int, int]
    ) -> None:
        """Continue pen stroke to *display_pos* (call only while LMB is held)."""
        self._paint(display_pos, display_size, self._PEN_COLOR, self._PEN_RADIUS, pen=True)

    def on_mouse_up(self) -> None:
        """End the current pen stroke."""
        self._last_pos = None

    def on_eraser_down(
        self, display_pos: tuple[int, int], display_size: tuple[int, int]
    ) -> None:
        """Begin an eraser stroke at *display_pos* (RMB)."""
        self._last_eraser_pos = None  # discard stale endpoint from any previous stroke
        self._paint(
            display_pos, display_size, (255, 255, 255), self._ERASER_RADIUS, pen=False
        )

    def on_eraser_move(
        self, display_pos: tuple[int, int], display_size: tuple[int, int]
    ) -> None:
        """Continue eraser stroke (call only while RMB is held)."""
        self._paint(
            display_pos, display_size, (255, 255, 255), self._ERASER_RADIUS, pen=False
        )

    def on_eraser_up(self) -> None:
        """End the current eraser stroke."""
        self._last_eraser_pos = None

    def clear(self) -> None:
        """Reset canvas to white and end any in-progress stroke."""
        with self._lock:
            self._canvas[:] = 255
        self._last_pos = None
        self._last_eraser_pos = None

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _paint(
        self,
        display_pos: tuple[int, int],
        display_size: tuple[int, int],
        color: tuple[int, int, int],
        radius: int,
        *,
        pen: bool,
    ) -> None:
        """Draw a dot and optionally a line from the previous stroke position."""
        pos = self._to_canvas(display_pos, display_size)
        prev = self._last_pos if pen else self._last_eraser_pos
        with self._lock:
            if prev is not None:
                cv2.line(self._canvas, prev, pos, color, radius * 2)
            cv2.circle(self._canvas, pos, radius, color, -1)
        if pen:
            self._last_pos = pos
        else:
            self._last_eraser_pos = pos

    def _to_canvas(
        self, display_pos: tuple[int, int], display_size: tuple[int, int]
    ) -> tuple[int, int]:
        """Map display pixel coordinates to 1920×1080 canvas coordinates."""
        dw, dh = display_size
        if dw == 0 or dh == 0:
            return (0, 0)
        cx = max(0, min(1919, int(display_pos[0] * 1920 / dw)))
        cy = max(0, min(1079, int(display_pos[1] * 1080 / dh)))
        return (cx, cy)

    def _loop(self) -> None:
        """Publish canvas frames at ~30fps using monotonic deadline tracking."""
        interval = 1.0 / 30
        next_deadline = time.monotonic()
        while self._active:
            with self._lock:
                frame = self._canvas.copy()
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                pass
            next_deadline += interval
            remaining = next_deadline - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            else:
                next_deadline = time.monotonic()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
