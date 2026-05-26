"""Stage 1 — Video Feed.

Reads frames from a camera or file in a background thread. Callers interact
only through the ``Capture`` object — the internal queue is an implementation
detail.

Usage::

    cap = Capture("recording.mp4").start()
    frame = cap.read()          # blocks until next frame (or None at EOF)
    cap.pause()
    cap.resume()
    cap.stop()

    # Or as a context manager:
    with Capture("recording.mp4") as cap:
        frame = cap.read()
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"})


class Capture:
    """Thread-backed video/image source with pause/resume support.

    Metadata (fps, frame_size) is read synchronously from the source header
    in ``__init__`` so callers can inspect it before the stream starts.

    The internal frame queue holds at most one entry — stale frames are
    dropped automatically so callers always receive the latest frame.
    """

    def __init__(self, source: str | int | None = None) -> None:
        self._source: int | str = 0 if source is None else source
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=1)
        self._paused = threading.Event()
        self._paused.set()  # not paused initially
        self._active = False
        self._thread: threading.Thread | None = None

        self.fps: float | None = None
        """Source fps from file header. ``None`` for live cameras (unreliable)."""

        self.frame_size: tuple[int, int] | None = None
        """``(width, height)`` from source metadata. ``None`` if unknown."""

        self._probe_metadata()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> Capture:
        """Spawn the capture thread. Returns *self* for fluent chaining."""
        self._active = True
        fn = self._image_loop if _is_image(self._source) else self._video_loop
        self._thread = threading.Thread(target=fn, daemon=True, name="capture")
        self._thread.start()
        log.info("Capture started: %r", self._source)
        return self

    def read(self) -> np.ndarray | None:
        """Block until the next frame is available.

        Returns:
            BGR uint8 frame, or ``None`` at end-of-stream.
        """
        return self._queue.get()

    def pause(self) -> None:
        """Freeze the capture thread before its next frame read."""
        self._paused.clear()

    def resume(self) -> None:
        """Unfreeze the capture thread."""
        self._paused.set()

    def stop(self) -> None:
        """Signal the capture thread to exit and unblock any pending read()."""
        self._active = False
        self._paused.set()  # unblock if paused
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

    def __enter__(self) -> Capture:
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _probe_metadata(self) -> None:
        """Read fps and frame dimensions from the source header.

        Opens and immediately releases a VideoCapture — no frames decoded.
        Camera fps is not populated because driver-reported values are unreliable.
        """
        if _is_image(self._source):
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

    def _image_loop(self) -> None:
        """Republish a still image at ~30 fps so the processing loop keeps ticking."""
        frame = cv2.imread(str(self._source))
        if frame is None:
            log.error("Cannot read image: %s", self._source)
            self._queue.put(None)
            return
        log.info("Image %dx%d: %s", frame.shape[1], frame.shape[0], self._source)
        while self._active:
            self._paused.wait()
            if not self._active:
                break
            _drop_put(self._queue, frame)
            time.sleep(1 / 30)

    def _video_loop(self) -> None:
        """Stream frames from a camera or video file."""
        cap = cv2.VideoCapture(self._source)
        if not cap.isOpened():
            log.error("Cannot open source: %r", self._source)
            self._queue.put(None)
            return
        log.info(
            "Opened %dx%d @ %s: %r",
            cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            f"{self.fps:.0f} fps" if self.fps else "live",
            self._source,
        )
        # Files pace to source fps; cameras block naturally in cap.read().
        frame_interval = (1.0 / self.fps) if self.fps is not None else 0.0
        try:
            while self._active:
                self._paused.wait()
                if not self._active:
                    break
                t = time.monotonic()
                ok, frame = cap.read()
                if not ok:
                    log.info("End of stream: %r", self._source)
                    break
                _drop_put(self._queue, frame)
                if frame_interval:
                    time.sleep(max(0.0, frame_interval - (time.monotonic() - t)))
        finally:
            cap.release()
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_image(source: int | str) -> bool:
    return isinstance(source, str) and Path(source).suffix.lower() in _IMAGE_SUFFIXES


def _drop_put(q: queue.Queue, frame: np.ndarray) -> None:
    """Evict the stale frame (if any) and publish the latest one."""
    try:
        q.get_nowait()
    except queue.Empty:
        pass
    q.put(frame)
