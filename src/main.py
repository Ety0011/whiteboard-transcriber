"""Entry point for the whiteboard transcription pipeline.

Starts the camera daemon thread and the processing thread, wires them
together via a Queue(maxsize=1), and handles graceful shutdown on
KeyboardInterrupt or when the input source is exhausted.

Usage:
    python src/main.py                    # live webcam (default)
    python src/main.py video.mp4          # video file for testing
    python src/main.py image.jpg          # still image for testing
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import capture
from pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Start the camera thread and processing pipeline."""
    parser = argparse.ArgumentParser(description="Whiteboard transcription pipeline")
    parser.add_argument(
        "source",
        nargs="?",
        metavar="FILE",
        help="video or image file (omit to use the default webcam)",
    )
    args = parser.parse_args()

    source: int | str = args.source if args.source is not None else 0

    output_path = Path("output/whiteboard.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame_queue = capture.process(source)
    pipeline = Pipeline(output_path)

    logger.info("Processing started. Press Ctrl-C to stop.")
    try:
        while True:
            frame = frame_queue.get()
            if frame is None:
                logger.info("End of stream.")
                break
            pipeline.process(frame)
    except KeyboardInterrupt:
        pass

    logger.info("Shutting down.")


if __name__ == "__main__":
    main()
