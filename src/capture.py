"""Camera capture thread (Stage 0).

Reads frames from cv2.VideoCapture at up to 30 fps and exposes the
most recent frame via a thread-safe Queue(maxsize=1). Old frames are
discarded automatically via put_nowait(), implementing back-pressure
without blocking the camera reader.

The camera thread runs as a daemon so it exits when the main thread ends.
"""

from __future__ import annotations

import queue

import numpy as np


def process(source: int | str = 0) -> queue.Queue:
    """Start the camera daemon thread and return the shared frame queue.

    Args:
        source: Camera device index (int) or path to a video file (str).

    Returns:
        A ``Queue(maxsize=1)`` that always holds the most recent BGR frame.
        Consumers should call ``queue.get()`` to block until a frame arrives.
    """
    raise NotImplementedError
