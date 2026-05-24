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

    The internal frame queue holds at most one entry — stale frames are
    dropped automatically so callers always receive the latest frame.
    """

    def __init__(self, source: str | int | None = None) -> None:
        self._source: int | str = 0 if source is None else source
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=1)
        self._running = threading.Event()
        self._running.set()
        self._active = False
        self._thread: threading.Thread | None = None

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
        self._running.clear()

    def resume(self) -> None:
        """Unfreeze the capture thread."""
        self._running.set()

    def stop(self) -> None:
        """Signal the capture thread to exit and unblock any pending read()."""
        self._active = False
        self._running.set()  # unblock if currently paused
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

    def __enter__(self) -> Capture:
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Thread loops (private)
    # ------------------------------------------------------------------

    def _image_loop(self) -> None:
        """Republish a still image at ~30 fps so the processing loop keeps ticking."""
        frame = cv2.imread(str(self._source))
        if frame is None:
            log.error("Cannot read image: %s", self._source)
            self._queue.put(None)
            return
        log.info("Image %dx%d: %s", frame.shape[1], frame.shape[0], self._source)
        while self._active:
            self._running.wait()
            if not self._active:
                break
            _publish(self._queue, frame)
            time.sleep(1 / 30)

    def _video_loop(self) -> None:
        """Stream frames from a camera or video file."""
        cap = cv2.VideoCapture(self._source)
        if not cap.isOpened():
            log.error("Cannot open source: %r", self._source)
            self._queue.put(None)
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        log.info(
            "Opened %dx%d @ %.0f fps: %r",
            cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            fps,
            self._source,
        )
        # Throttle only for files; live cameras pace themselves via cap.read().
        frame_time = (1.0 / fps) if isinstance(self._source, str) else 0.0

        try:
            while self._active:
                self._running.wait()
                if not self._active:
                    break
                t = time.monotonic()
                ok, frame = cap.read()
                if not ok:
                    log.info("End of stream: %r", self._source)
                    break
                _publish(self._queue, frame)
                if frame_time:
                    time.sleep(max(0.0, frame_time - (time.monotonic() - t)))
        finally:
            cap.release()
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_image(source: int | str) -> bool:
    return isinstance(source, str) and Path(source).suffix.lower() in _IMAGE_SUFFIXES


def _publish(q: queue.Queue, frame: np.ndarray) -> None:
    """Drop the stale frame (if any) and put the latest one."""
    try:
        q.get_nowait()
    except queue.Empty:
        pass
    q.put(frame)
