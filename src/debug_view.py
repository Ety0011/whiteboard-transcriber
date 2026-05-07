"""Dev tool — Stage 0 + Stage 1 live preview.

Opens the camera (or a video file), runs perspective registration, and
shows the warped output in a cv2.imshow window.

Usage::

    python src/debug_view.py          # webcam index 0
    python src/debug_view.py 1        # alternate camera index
    python src/debug_view.py video.mp4

Keyboard controls:
    q  — quit
    d  — toggle debug overlay (corners + quad drawn on input frame)
"""

from __future__ import annotations

import logging
import sys

import cv2

from capture import process as start_camera
from registration import Registrar

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main(source: int | str = 0) -> None:
    """Run the Stage 0 + Stage 1 preview loop.

    Args:
        source: Camera device index or path to a video file.
    """
    frame_queue = start_camera(source)
    registrar = Registrar(debug=True)

    print("Stage 1 preview running. Press 'q' to quit, 'd' to toggle debug overlay.")

    while True:
        frame = frame_queue.get()  # blocks until a frame is available
        if frame is None:
            logger.info("End of stream — exiting")
            break

        warped = registrar.process(frame)

        cv2.imshow("Stage 1 — Warped Board", warped)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("d"):
            registrar.debug = not registrar.debug
            logger.info("Debug overlay: %s", "ON" if registrar.debug else "OFF")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    raw = sys.argv[1] if len(sys.argv) > 1 else "0"
    # Accept both integer camera indices and file paths
    src: int | str = int(raw) if raw.isdigit() else raw
    main(src)
