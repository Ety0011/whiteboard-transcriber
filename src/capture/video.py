"""Stage 1 — Video Feed.

Reads frames from a camera or file. File sources read directly per caller tick
(same pattern as replay.py — no background thread, no sleep, caller paces via
clock.tick). Camera sources read in a background thread so cap.read() blocking
does not stall the UI loop.

Usage::

    cap = Capture("recording.mp4").start()
    frame = cap.read()          # blocks until next frame (or None at EOF)
    cap.stop()

    # Or as a context manager:
    with Capture("recording.mp4") as cap:
        frame = cap.read()
"""

from __future__ import annotations

import logging
import queue
import threading
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"})


class Capture:
    """Video/image source with pause/resume support.

    For file sources, frames are decoded directly on each ``try_read()`` call —
    the caller's loop rate (e.g. ``clock.tick(fps)``) determines playback speed.
    For camera sources, a background thread drains the camera so the UI thread
    is never blocked by a hardware read.

    Metadata (fps, frame_size) is read synchronously from the source header
    in ``__init__`` so callers can inspect it before the stream starts.
    """

    def __init__(self, source: str | int | None = None) -> None:
        self._source: int | str = 0 if source is None else source
        self._is_camera = isinstance(self._source, int)
        self._is_image_src = _is_image(self._source)

        self._cap: cv2.VideoCapture | None = None
        self._cached_image: np.ndarray | None = None
        self._active = False
        self._paused = False

        # Camera path only — background thread + drop-old queue.
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=1)
        self._paused_event = threading.Event()
        self._paused_event.set()
        self._thread: threading.Thread | None = None

        self.fps: float | None = None
        """Source fps from file header. ``None`` for live cameras."""

        self.frame_size: tuple[int, int] | None = None
        """``(width, height)`` from source metadata. ``None`` if unknown."""

        self._probe_metadata()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> Capture:
        """Open the source and, for cameras, spawn the capture thread.

        Returns:
            *self* for fluent chaining.
        """
        if self._is_image_src:
            self._cached_image = cv2.imread(str(self._source))
            if self._cached_image is None:
                raise RuntimeError(f"Cannot read image: {self._source!r}")
            h, w = self._cached_image.shape[:2]
            self.frame_size = (w, h)
        else:
            self._cap = cv2.VideoCapture(self._source)
            if not self._cap.isOpened():
                raise RuntimeError(f"Cannot open source: {self._source!r}")
            if self._is_camera:
                self._thread = threading.Thread(
                    target=self._camera_loop, daemon=True, name="capture"
                )
                self._thread.start()

        self._active = True
        log.info(
            "Capture started: %r  %s",
            self._source,
            f"{self.frame_size[0]}×{self.frame_size[1]} @ {self.fps:.0f} fps"
            if self.frame_size and self.fps
            else str(self.frame_size or ""),
        )
        return self

    def read(self) -> np.ndarray | None:
        """Return the next frame, blocking if necessary.

        Returns:
            BGR uint8 frame, or ``None`` at end-of-stream.
        """
        if self._is_image_src:
            return self._cached_image
        if self._is_camera:
            return self._queue.get()
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        return frame if ok else None

    def try_read(self) -> np.ndarray | None:
        """Non-blocking read.

        For file sources, decodes and returns the next frame immediately.
        For camera sources, returns the latest buffered frame or raises
        ``queue.Empty`` if the background thread has not produced one yet.

        Returns:
            BGR uint8 frame, or ``None`` on end-of-stream.

        Raises:
            queue.Empty: Camera source has no new frame buffered this tick.
        """
        if self._paused:
            raise queue.Empty
        if self._is_image_src:
            return self._cached_image
        if self._is_camera:
            return self._queue.get_nowait()
        if self._cap is None or not self._active:
            raise queue.Empty
        ok, frame = self._cap.read()
        return frame if ok else None

    def pause(self) -> None:
        """Freeze playback before the next frame."""
        self._paused = True
        self._paused_event.clear()

    def resume(self) -> None:
        """Unfreeze playback."""
        self._paused = False
        self._paused_event.set()

    def stop(self) -> None:
        """Release the source and unblock any pending read()."""
        self._active = False
        self._paused_event.set()  # unblock camera thread if paused
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

    # ------------------------------------------------------------------
    # Drawing API (no-ops — implemented by CanvasCapture in demo mode)
    # ------------------------------------------------------------------

    def on_mouse_down(
        self, display_pos: tuple[int, int], display_size: tuple[int, int]
    ) -> None:
        """No-op for video/camera sources."""

    def on_mouse_move(
        self, display_pos: tuple[int, int], display_size: tuple[int, int]
    ) -> None:
        """No-op for video/camera sources."""

    def on_mouse_up(self) -> None:
        """No-op for video/camera sources."""

    def on_eraser_down(
        self, display_pos: tuple[int, int], display_size: tuple[int, int]
    ) -> None:
        """No-op for video/camera sources."""

    def on_eraser_move(
        self, display_pos: tuple[int, int], display_size: tuple[int, int]
    ) -> None:
        """No-op for video/camera sources."""

    def on_eraser_up(self) -> None:
        """No-op for video/camera sources."""

    def clear(self) -> None:
        """No-op for video/camera sources."""

    def __enter__(self) -> Capture:
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _probe_metadata(self) -> None:
        """Read fps and frame dimensions from the source header without decoding frames."""
        if self._is_image_src:
            return
        cap = cv2.VideoCapture(self._source)
        if not cap.isOpened():
            return
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        raw_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if w > 0 and h > 0:
            self.frame_size = (w, h)
        if raw_fps > 0 and isinstance(self._source, str):
            self.fps = raw_fps

    def _camera_loop(self) -> None:
        """Background thread — drains the camera into the drop-old queue."""
        assert self._cap is not None
        try:
            while self._active:
                self._paused_event.wait()
                if not self._active:
                    break
                ok, frame = self._cap.read()
                if not ok:
                    log.info("Camera read failed — stopping capture thread.")
                    break
                try:
                    self._queue.get_nowait()  # evict stale frame
                except queue.Empty:
                    pass
                try:
                    self._queue.put_nowait(frame)
                except queue.Full:
                    pass
        finally:
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_image(source: int | str) -> bool:
    return isinstance(source, str) and Path(source).suffix.lower() in _IMAGE_SUFFIXES
