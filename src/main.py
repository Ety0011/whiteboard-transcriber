"""Entry point for the whiteboard transcription pipeline.

Starts the camera daemon thread and the processing thread, wires them
together via a Queue(maxsize=1), and handles graceful shutdown on
KeyboardInterrupt or when the input source is exhausted.

Usage:
    python src/main.py                    # live webcam (index 0)
    python src/main.py --input video.mp4  # video file for testing
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def main() -> None:
    """Start the camera thread and processing pipeline."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
