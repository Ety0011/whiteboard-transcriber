"""Camera capture thread (Stage 0).

Reads frames from cv2.VideoCapture at up to 30 fps and exposes the
most recent frame via a thread-safe Queue(maxsize=1). Old frames are
discarded automatically, implementing back-pressure without blocking
the camera reader.

The camera thread runs as a daemon so it exits when the main thread ends.
"""

from __future__ import annotations

import logging
import queue
import threading
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def process(source: int | str = 0) -> queue.Queue:
    """Start the camera daemon thread and return the shared frame queue.

    Args:
        source: Camera device index (int) or path to a video file (str).

    Returns:
        A ``Queue(maxsize=1)`` that always holds the most recent BGR frame.
        Consumers should call ``queue.get()`` to block until a frame arrives.
    """
    frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
    thread = threading.Thread(
        target=_camera_loop,
        args=(source, frame_queue),
        daemon=True,
        name="camera-capture",
    )
    thread.start()
    logger.info("Camera thread started for source %r", source)
    return frame_queue


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
