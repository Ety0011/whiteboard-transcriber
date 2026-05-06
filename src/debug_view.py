"""Dev tool — Stage 0 + Stage 1 + Stage 2 live preview.

Opens the camera (or a video file), runs perspective registration and
person segmentation, and shows the results in a cv2.imshow window.

Usage::

    python src/debug_view.py          # webcam index 0
    python src/debug_view.py 1        # alternate camera index
    python src/debug_view.py video.mp4

Keyboard controls:
    q  — quit
    d  — toggle Stage 1 corner overlay: shows raw camera frame with detected
         quad drawn on it so you can verify corner positions before warping
    s  — toggle Stage 2 person-mask overlay (semi-transparent red, warped view)
"""

from __future__ import annotations

import logging
import sys

import cv2
import numpy as np

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


def _draw_corners_on_raw(frame: np.ndarray, registrar: Registrar) -> np.ndarray:
    """Return a copy of the raw camera *frame* with the detected quad drawn on it."""
    out = frame.copy()
    corners = registrar._cached_corners  # (4,2) float32, TL/TR/BR/BL

    if corners is None:
        cv2.putText(
            out, "No board detected", (40, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 220), 2,
        )
        return out

    pts = corners.astype(np.int32)
    cv2.polylines(
        out, [pts.reshape(-1, 1, 2)], isClosed=True, color=(0, 0, 220), thickness=3
    )
    for i, (x, y) in enumerate(pts):
        cv2.circle(out, (int(x), int(y)), 12, (0, 200, 0), -1)
        cv2.putText(
            out, _LABELS[i],
            (int(x) + 14, int(y) - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )
    return out


def main(source: int | str = 0) -> None:
    """Run the Stage 0 + 1 + 2 preview loop."""
    fps = _video_fps(source)
    wait_ms = max(1, int(1000 / fps))

    frame_queue = start_camera(source)
    registrar = Registrar()

    print("Initialising MediaPipe segmenter (first load may take a moment)…")
    segmenter = Segmenter()

    show_corners = False
    show_mask = True

    print(
        "Stage 1+2 preview running.\n"
        "  d — toggle corner overlay (raw camera view with detected quad)\n"
        "  s — toggle person-mask overlay\n"
        "  q — quit"
    )

    while True:
        frame = frame_queue.get()
        if frame is None:
            logger.info("End of stream — exiting")
            break

        warped = registrar.process(frame)

        if show_corners:
            display = _draw_corners_on_raw(frame, registrar)
        else:
            mask = segmenter.process(warped)
            display = _apply_mask_overlay(warped, mask) if show_mask else warped

        cv2.imshow("Stage 1+2", display)

        key = cv2.waitKey(wait_ms) & 0xFF
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
