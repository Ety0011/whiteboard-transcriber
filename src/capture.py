"""Camera capture thread (Stage 0).

Reads frames from a camera or file and exposes the most recent frame via a
thread-safe ``Queue(maxsize=1)``. The queue always holds the latest frame;
stale frames are dropped automatically, providing natural back-pressure.

Still images are republished on a loop so the processing thread keeps
receiving frames without special-casing.

Usage::

    q = capture.process()                  # default webcam
    q = capture.process("recording.mp4")   # video file
    frame = q.get()                        # blocks until a frame is ready
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


def process(source: str | None = None) -> queue.Queue[np.ndarray | None]:
    """Start the capture daemon and return the shared frame queue.

    Args:
        source: Path to a video or image file, or ``None`` to use the default
            webcam.

    Returns:
        ``Queue(maxsize=1)`` yielding BGR uint8 frames. A ``None`` sentinel
        marks end-of-stream for video and camera sources.
    """
    _source: int | str = source if source is not None else 0
    q: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=1)
    fn = _image_source if _is_image(_source) else _video_source
    threading.Thread(target=fn, args=(_source, q), daemon=True, name="capture").start()
    log.info("Capture started: %r", _source)
    return q


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


def _image_source(path: str, q: queue.Queue) -> None:
    """Republish a still image at ~30 fps so the processing loop keeps ticking."""
    frame = cv2.imread(path)
    if frame is None:
        log.error("Cannot read image: %s", path)
        q.put(None)
        return
    log.info("Image %dx%d: %s", frame.shape[1], frame.shape[0], path)
    while True:
        _publish(q, frame)
        time.sleep(1 / 30)


def _video_source(source: int | str, q: queue.Queue) -> None:
    """Stream frames from a camera or video file into *q*."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        log.error("Cannot open source: %r", source)
        q.put(None)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    log.info(
        "Opened %dx%d @ %.0f fps: %r",
        cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
        fps,
        source,
    )

    # Throttle only for files; live cameras pace themselves via cap.read().
    frame_time = (1.0 / fps) if isinstance(source, str) else 0.0

    try:
        while True:
            t = time.monotonic()
            ok, frame = cap.read()
            if not ok:
                log.info("End of stream: %r", source)
                break
            _publish(q, frame)
            if frame_time:
                time.sleep(max(0.0, frame_time - (time.monotonic() - t)))
    finally:
        cap.release()
        q.put(None)
        log.info("Capture released: %r", source)
