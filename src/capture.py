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
    """Capture loop — runs in daemon thread until the camera fails or closes."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error("Cannot open camera source %r", source)
        return

    logger.info(
        "Camera opened: %.0fx%.0f @ %.0f fps",
        cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
        cap.get(cv2.CAP_PROP_FPS),
    )

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Frame read failed — stopping camera thread")
                break
            # Drain the stale frame (if any) then publish the latest
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
            frame_queue.put(frame)
    finally:
        cap.release()
        logger.info("Camera released")
