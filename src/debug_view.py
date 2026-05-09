"""Dev tool — Stage 0 + Stage 1 + Stage 2 + Stage 3 live preview.

Opens the camera (or a video file), runs perspective registration,
person segmentation, and surface reconstruction, showing the results
in cv2.imshow windows.

Usage::

    python src/debug_view.py          # webcam index 0
    python src/debug_view.py 1        # alternate camera index
    python src/debug_view.py video.mp4

Keyboard controls:
    q  — quit
    d  — toggle Stage 1 corner overlay: shows raw camera frame with detected
         quad drawn on it so you can verify corner positions before warping
    s  — toggle Stage 2 person-mask overlay (semi-transparent red, warped view)
    b  — toggle Stage 3 background composite (separate window, side-by-side)
"""

from __future__ import annotations

import logging
import sys

import cv2
import numpy as np

from background import BackgroundReconstructor
from capture import process as start_camera
from registration import Registrar
from segmentation import Segmenter

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_LABELS = ["TL", "TR", "BR", "BL"]


def _video_fps(source: int | str) -> float:
    """Return the native FPS of *source*, or 30.0 if it cannot be determined."""
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else 30.0


def _apply_mask_overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Blend a semi-transparent red highlight over *frame* where *mask* is 1."""
    overlay = frame.copy()
    overlay[mask == 1] = (0, 0, 220)
    return cv2.addWeighted(frame, 0.65, overlay, 0.35, 0)


def _draw_corners(frame: np.ndarray, registrar: Registrar) -> np.ndarray:
    """Draw the detected board quad on *frame* in-place and return it."""
    with registrar._lock:
        corners = registrar._cached_corners

    if registrar._detecting:
        cv2.putText(
            frame,
            "Detecting board...",
            (40, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 180, 220),
            2,
        )
    elif corners is None:
        cv2.putText(
            frame,
            "No board detected",
            (40, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 180, 220),
            2,
        )

    if corners is None:
        return frame

    pts = corners.astype(np.int32)
    cv2.polylines(
        frame, [pts.reshape(-1, 1, 2)], isClosed=True, color=(0, 0, 220), thickness=3
    )
    for i, (x, y) in enumerate(pts):
        cv2.circle(frame, (int(x), int(y)), 12, (0, 200, 0), -1)
        cv2.putText(
            frame,
            _LABELS[i],
            (int(x) + 14, int(y) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
    return frame


def main(source: int | str = 0) -> None:
    """Run the Stage 0 + 1 + 2 + 3 preview loop."""
    fps = _video_fps(source)
    wait_ms = max(1, int(1000 / fps))

    frame_queue = start_camera(source)
    registrar = Registrar()

    print("Initialising MediaPipe segmenter (first load may take a moment)...")
    segmenter = Segmenter()
    reconstructor = BackgroundReconstructor()

    show_corners = True
    show_mask = True
    show_bg = True

    print(
        "Stage 1+2+3 preview running.\n"
        "  d — toggle detected board corners\n"
        "  s — toggle person-mask overlay\n"
        "  b — toggle Stage 3 background composite (separate window)\n"
        "  q — quit"
    )

    while True:
        frame = frame_queue.get()
        if frame is None:
            logger.info("End of stream — exiting")
            break

        warped = registrar.process(frame)

        display = frame.copy()

        if show_mask:
            mask = segmenter.process(frame)
            display = _apply_mask_overlay(display, mask)

        if show_corners:
            display = _draw_corners(display, registrar)

        cv2.imshow("Stage 1+2", display)

        if show_bg:
            bg_mask = segmenter.process(warped)
            composite = reconstructor.process(warped, bg_mask)
            thumb_w, thumb_h = 640, 360
            left = cv2.resize(warped, (thumb_w, thumb_h))
            right = cv2.resize(composite, (thumb_w, thumb_h))
            cv2.imshow("Stage 3", np.hstack([left, right]))

        key = cv2.waitKey(wait_ms) & 0xFF
        if key == ord("q"):
            break
        if key == ord("d"):
            show_corners = not show_corners
            logger.info("Corner overlay: %s", "ON" if show_corners else "OFF")
        if key == ord("s"):
            show_mask = not show_mask
            logger.info("Mask overlay: %s", "ON" if show_mask else "OFF")
        if key == ord("b"):
            show_bg = not show_bg
            if not show_bg:
                cv2.destroyWindow("Stage 3")
            logger.info("Background composite: %s", "ON" if show_bg else "OFF")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    raw = sys.argv[1] if len(sys.argv) > 1 else "0"
    src = int(raw) if raw.isdigit() else raw
    main(src)
