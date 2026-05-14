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

import cv2

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
    parser.add_argument("--debug", action="store_true", help="enable debug logging")
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.DEBUG if args.debug else logging.INFO)

    output_path = Path("output/whiteboard.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame_queue = capture.process(args.source)
    pipeline = Pipeline(output_path)

    logger.info("Processing started. Press q or Ctrl-C to stop.")
    try:
        while True:
            frame = frame_queue.get()
            if frame is None:
                logger.info("End of stream.")
                break
            pipeline.process(frame)
            cv2.imshow("Whiteboard", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()

    logger.info("Shutting down.")


if __name__ == "__main__":
    main()
