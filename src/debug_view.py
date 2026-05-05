"""Dev tool — Stage 0 + Stage 1 + Stage 2 live preview.

Opens the camera (or a video file), runs perspective registration and
person segmentation, and shows the results in a cv2.imshow window.

Usage::

    python src/debug_view.py          # webcam index 0
    python src/debug_view.py 1        # alternate camera index
    python src/debug_view.py video.mp4

Keyboard controls:
    q  — quit
    d  — toggle Stage 1 debug overlay (detected board corners + quad)
    s  — toggle Stage 2 person-mask overlay (semi-transparent red)
"""

from __future__ import annotations

import logging
import sys

import cv2
import numpy as np

from src.capture import process as start_camera
from src.registration import Registrar
from src.segmentation import Segmenter

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _apply_mask_overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Blend a semi-transparent red highlight over *frame* where *mask* is 1.

    Args:
        frame: BGR uint8 image to annotate.
        mask: Binary uint8 mask (H, W), values 0 or 1.

    Returns:
        New BGR image with red overlay applied.
    """
    overlay = frame.copy()
    overlay[mask == 1] = (0, 0, 220)  # red in BGR
    return cv2.addWeighted(frame, 0.65, overlay, 0.35, 0)


def main(source: int | str = 0) -> None:
    """Run the Stage 0 + 1 + 2 preview loop.

    Args:
        source: Camera device index or path to a video file.
    """
    frame_queue = start_camera(source)
    registrar = Registrar(debug=False)

    print("Initialising MediaPipe segmenter (first load may take a moment)…")
    segmenter = Segmenter()

    show_corners = False
    show_mask = True

    print(
        "Stage 1+2 preview running.\n"
        "  d — toggle board-corner overlay\n"
        "  s — toggle person-mask overlay\n"
        "  q — quit"
    )

    while True:
        frame = frame_queue.get()  # blocks until a frame is available

        registrar.debug = show_corners
        warped = registrar.process(frame)

        mask = segmenter.process(warped)

        display = _apply_mask_overlay(warped, mask) if show_mask else warped

        cv2.imshow("Stage 1+2 — Warped Board / Person Mask", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("d"):
            show_corners = not show_corners
            logger.info("Corner overlay: %s", "ON" if show_corners else "OFF")
        if key == ord("s"):
            show_mask = not show_mask
            logger.info("Mask overlay: %s", "ON" if show_mask else "OFF")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    raw = sys.argv[1] if len(sys.argv) > 1 else "0"
    src: int | str = int(raw) if raw.isdigit() else raw
    main(src)
