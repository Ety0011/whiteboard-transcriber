"""Stage 1 — Video Feed.

Reads frames from a camera or file. File sources decode directly on each
try_read() call — caller's loop rate determines playback speed. Camera sources
use a background thread so try_read() never blocks on hardware I/O.

Usage::

    cap = Capture("recording.mp4").start()
    frame = cap.try_read()   # None on end-of-stream or no frame ready yet
    cap.stop()

    # Or as a context manager:
    with Capture("recording.mp4") as cap:
        frame = cap.try_read()
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .source import FrameSource

log = logging.getLogger(__name__)

_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"})


class Capture(FrameSource):
    """Video/image source with pause/resume support.

    For file sources, frames are decoded directly on each try_read() call —
    the caller's loop rate determines playback speed. For camera sources, a
    background thread drains the camera so the UI thread is never blocked.

    Metadata (fps, frame_size) is populated in start() once the source is open.

    Args:
        source: File path, camera index (int), or None (defaults to camera 0).
    """

    def __init__(self, source: str | int | None = None) -> None:
        self._source: int | str = 0 if source is None else source
        self._is_camera = isinstance(self._source, int)
        self._is_image_src = _is_image(self._source)

        self._cap: cv2.VideoCapture | None = None
        self._cached_image: np.ndarray | None = None
        self._active = False
        self._paused = False

        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=1)
        self._thread: threading.Thread | None = None

        self.fps: float | None = None
        self.frame_size: tuple[int, int] | None = None

        # Pre-read file metadata so frame_size is available before start() if needed.
        if not self._is_camera and not self._is_image_src:
            self._probe_metadata()

    # ------------------------------------------------------------------
    # FrameSource implementation
    # ------------------------------------------------------------------

    def start(self) -> Capture:
        """Open the source and, for cameras, spawn the capture thread.

        Returns:
            self for fluent chaining.

        Raises:
            RuntimeError: If the source cannot be opened.
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
                w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if w > 0 and h > 0:
                    self.frame_size = (w, h)
                self._thread = threading.Thread(
                    target=self._camera_loop,
                    daemon=True,
                    name=f"capture-{self._source}",
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

    def try_read(self) -> np.ndarray | None:
        """Non-blocking frame read.

        Returns:
            BGR uint8 frame, or None if paused, no frame buffered this tick,
            or end-of-stream. Never raises. Use is_active to distinguish
            end-of-stream (False) from a transient empty tick (True).
        """
        if self._paused:
            return None
        if self._is_image_src:
            return self._cached_image
        if self._is_camera:
            try:
                return self._queue.get_nowait()
            except queue.Empty:
                return None
        if self._cap is None or not self._active:
            return None
        ok, frame = self._cap.read()
        if not ok:
            self._active = False
            return None
        return frame

    def stop(self) -> None:
        """Release the source and unblock any pending reads."""
        self._active = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

    def pause(self) -> None:
        """Freeze playback — try_read() returns None while paused."""
        self._paused = True

    def resume(self) -> None:
        """Unfreeze playback."""
        self._paused = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        """True while the source is open and has not reached end-of-stream."""
        return self._active

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _probe_metadata(self) -> None:
        """Read fps and frame dimensions from the file header without decoding frames."""
        cap = cv2.VideoCapture(self._source)
        if not cap.isOpened():
            return
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        raw_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if w > 0 and h > 0:
            self.frame_size = (w, h)
        if raw_fps > 0:
            self.fps = raw_fps

    def _camera_loop(self) -> None:
        """Background thread — drains the camera into the drop-old queue."""
        assert self._cap is not None
        try:
            while self._active:
                if self._paused:
                    time.sleep(0.01)
                    continue
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
            self._active = False
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_image(source: int | str) -> bool:
    return isinstance(source, str) and Path(source).suffix.lower() in _IMAGE_SUFFIXES
