"""Demo canvas — mouse-drawable whiteboard with the same API as Capture."""
from __future__ import annotations

import queue
import threading
import time

import cv2
import numpy as np


class CanvasCapture:
    """Drawable 1920×1080 white canvas that publishes frames at ~30fps.

    Provides start()/read()/pause()/resume()/stop() so it can replace
    Capture in the main loop without any pipeline changes.

    Use on_mouse_down/on_mouse_move/on_mouse_up to paint strokes onto the
    canvas; clear() resets it to white.
    """

    fps: float | None = 30.0
    frame_size: tuple[int, int] = (1920, 1080)

    _PEN_COLOR: tuple[int, int, int] = (0, 0, 0)
    _PEN_RADIUS: int = 8
    _ERASER_RADIUS: int = 20

    def __init__(self) -> None:
        self._canvas = np.full((1080, 1920, 3), 255, dtype=np.uint8)
        self._lock = threading.Lock()
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=1)
        self._active = False
        self._thread: threading.Thread | None = None
        self._last_pos: tuple[int, int] | None = None
        self._last_eraser_pos: tuple[int, int] | None = None

    # ------------------------------------------------------------------
    # Public API (matches Capture)
    # ------------------------------------------------------------------

    def start(self) -> CanvasCapture:
        """Spawn the frame-publish thread. Returns self for fluent chaining."""
        self._active = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="canvas-capture"
        )
        self._thread.start()
        return self

    def read(self) -> np.ndarray | None:
        """Block until the next frame is available."""
        return self._queue.get()

    def pause(self) -> None:
        """No-op — canvas is always live."""

    def resume(self) -> None:
        """No-op — canvas is always live."""

    def stop(self) -> None:
        """Signal the publish thread to exit and unblock any pending read()."""
        self._active = False
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

    # ------------------------------------------------------------------
    # Drawing API
    # ------------------------------------------------------------------

    def on_mouse_down(
        self, display_pos: tuple[int, int], display_size: tuple[int, int]
    ) -> None:
        """Begin a pen stroke at display_pos."""
        self._paint(
            display_pos, display_size, self._PEN_COLOR, self._PEN_RADIUS, pen=True
        )

    def on_mouse_move(
        self, display_pos: tuple[int, int], display_size: tuple[int, int]
    ) -> None:
        """Continue pen stroke to display_pos (call only while LMB is held)."""
        self._paint(
            display_pos, display_size, self._PEN_COLOR, self._PEN_RADIUS, pen=True
        )

    def on_mouse_up(self) -> None:
        """End the current pen stroke."""
        self._last_pos = None

    def on_eraser_down(
        self, display_pos: tuple[int, int], display_size: tuple[int, int]
    ) -> None:
        """Begin an eraser stroke at display_pos (RMB)."""
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
        """Draw a dot and optionally a line from the previous position."""
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
        cx = max(0, min(1919, int(display_pos[0] * 1920 / dw)))
        cy = max(0, min(1079, int(display_pos[1] * 1080 / dh)))
        return (cx, cy)

    def _loop(self) -> None:
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
            time.sleep(1 / 30)
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
