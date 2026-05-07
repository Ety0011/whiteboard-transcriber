"""Camera capture thread (Stage 0).

Reads frames from cv2.VideoCapture at up to 30 fps and exposes the
most recent frame via a thread-safe Queue(maxsize=1). Old frames are
discarded automatically, implementing back-pressure without blocking
the camera reader.

Also handles still images: the single frame is placed in the queue once
and the thread then blocks indefinitely, keeping the queue populated until
the main thread exits (daemon thread).

The camera thread runs as a daemon so it exits when the main thread ends.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def process(source: int | str = 0) -> queue.Queue:
    """Start the capture daemon thread and return the shared frame queue.

    Args:
        source: Camera device index (int), path to a video file, or path to
            a still image (jpg/png/bmp/tiff/webp).

    Returns:
        A ``Queue(maxsize=1)`` that always holds the most recent BGR frame.
        Consumers should call ``queue.get()`` to block until a frame arrives.
        A ``None`` sentinel is placed in the queue on end-of-stream (video/
        camera only — image sources never emit a sentinel).
    """
    frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
    is_image = isinstance(source, str) and Path(source).suffix.lower() in _IMAGE_EXTENSIONS
    target = _image_loop if is_image else _camera_loop
    thread = threading.Thread(
        target=target,
        args=(source, frame_queue),
        daemon=True,
        name="camera-capture",
    )
    thread.start()
    logger.info("Capture thread started for source %r", source)
    return frame_queue


def _image_loop(path: str, frame_queue: queue.Queue) -> None:
    """Continuously republish a still image so the display loop keeps ticking."""
    frame = cv2.imread(path)
    if frame is None:
        logger.error("Cannot read image %r", path)
        frame_queue.put(None)
        return
    logger.info("Image loaded: %dx%d — %s", frame.shape[1], frame.shape[0], path)
    while True:
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            pass
        frame_queue.put(frame)
        time.sleep(0.1)


def _camera_loop(source: int | str, frame_queue: queue.Queue) -> None:
    """Capture loop — runs in daemon thread until the camera fails or closes.

    For video files the loop sleeps between reads so frames are consumed at the
    file's native frame rate rather than instantly. For live cameras cap.read()
    already blocks at the hardware rate so no additional sleep is needed.

    A ``None`` sentinel is placed in the queue when the loop exits so that
    consumers can detect end-of-stream without polling.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error("Cannot open camera source %r", source)
        frame_queue.put(None)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(
        "Camera opened: %.0fx%.0f @ %.0f fps",
        cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
        fps,
    )

    # Throttle only for video files; live cameras block naturally in cap.read().
    frame_interval = (1.0 / fps) if (isinstance(source, str) and fps > 0) else 0.0

    try:
        while True:
            t0 = time.monotonic()
            ret, frame = cap.read()
            if not ret:
                logger.info("End of stream for source %r", source)
                break
            # Drain the stale frame (if any) then publish the latest
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
            frame_queue.put(frame)
            if frame_interval > 0:
                elapsed = time.monotonic() - t0
                time.sleep(max(0.0, frame_interval - elapsed))
    finally:
        cap.release()
        frame_queue.put(None)  # sentinel: signals end-of-stream to consumers
        logger.info("Camera released")
