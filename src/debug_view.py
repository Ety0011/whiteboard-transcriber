"""Dev tool — Stage 0 + Stage 1 + Stage 2 + Stage 3 + Stage 4 + Stage 5 live preview.

Opens the camera (or a video file), runs perspective registration,
person segmentation, surface reconstruction, layout detection, and text line
detection, showing the results in cv2.imshow windows.

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
    l  — toggle Stage 4 layout bounding boxes on the Stage 3 composite
    t  — toggle Stage 5 text line bounding boxes within each layout region
    r  — toggle Stage 5 region tracker lifecycle visualization
"""

from __future__ import annotations

import logging
import sys

import cv2
import numpy as np

from background import BackgroundReconstructor
from capture import process as start_camera
from layout import LayoutDetector, Region
from registration import Registrar
from segmentation import Segmenter
from text_detection import RegionWithLines, TextDetector
from tracker import Detection, RegionState, RegionTracker

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_LABELS = ["TL", "TR", "BR", "BL"]

# BGR colours per layout label for the Stage 4 overlay.
_LAYOUT_COLOURS: dict[str, tuple[int, int, int]] = {
    "text": (0, 200, 0),
    "paragraph_title": (0, 220, 220),
    "table": (0, 220, 220),
    "image": (220, 0, 220),
    "formula": (0, 100, 255),
    "content": (100, 200, 0),
    "algorithm": (200, 200, 200),
    "chart": (220, 100, 0),
}
_LAYOUT_DEFAULT_COLOUR: tuple[int, int, int] = (180, 180, 180)

# --- Updated Stage 5 State Colours ---
_STATE_COLOURS = {
    RegionState.CANDIDATE: (255, 255, 0),  # Cyan
    RegionState.STABILIZING: (0, 165, 255),  # Orange
    RegionState.STABLE: (0, 255, 0),  # Green
    RegionState.MISSING: (200, 0, 200),  # Magenta/Purple
    RegionState.REMOVED: (0, 0, 255),  # Red
}


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


def _draw_text_lines(frame: np.ndarray, regions: list[RegionWithLines]) -> np.ndarray:
    """Draw text line bounding boxes on *frame* in board-composite coordinates."""
    out = frame.copy()
    for r in regions:
        rx1, ry1, _, _ = r.bbox
        for line in r.lines:
            lx1, ly1, lx2, ly2 = line.bbox
            cv2.rectangle(
                out, (rx1 + lx1, ry1 + ly1), (rx1 + lx2, ry1 + ly2), (255, 150, 0), 1
            )
    return out


# --- Updated Stage 5 Drawing Helper ---
def _draw_tracker(frame: np.ndarray, regions: list) -> np.ndarray:
    """Draw persistent tracked regions with IDs, States, and OCR status."""
    out = frame.copy()
    for reg in regions:
        x1, y1, x2, y2 = reg.bbox
        color = _STATE_COLOURS.get(reg.state, (255, 255, 255))

        # Draw dotted line for MISSING regions, solid for others
        thickness = 1 if reg.state == RegionState.MISSING else 2
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        # Indicate OCR status in the label
        ocr_status = "✓" if reg.ocr_text else "..."
        label = f"ID:{reg.id} {reg.state.value} {reg.confidence:.2f} [{ocr_status}]"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(out, (x1, y1), (x1 + tw + 4, y1 + th + 6), color, -1)
        cv2.putText(
            out,
            label,
            (x1 + 2, y1 + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return out


def _draw_layout(frame: np.ndarray, regions: list[Region]) -> np.ndarray:
    """Draw layout bounding boxes and labels on *frame* and return it."""
    out = frame.copy()
    for r in regions:
        x1, y1, x2, y2 = r.bbox
        colour = _LAYOUT_COLOURS.get(r.label, _LAYOUT_DEFAULT_COLOUR)
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        tag = f"{r.label} {r.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(
            out,
            tag,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return out


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
    """Run the Stage 0-5 preview loop."""
    fps = _video_fps(source)
    wait_ms = max(1, int(1000 / fps))

    frame_queue = start_camera(source)
    registrar = Registrar()

    print("Initialising MediaPipe segmenter...")
    segmenter = Segmenter()
    reconstructor = BackgroundReconstructor()
    print("Initialising PP-DocLayout model...")
    layout_detector = LayoutDetector()
    print("Initialising PP-OCRv5_server_det model...")
    text_detector = TextDetector()

    # --- Stage 5 Tracker Instance ---
    region_tracker = RegionTracker()
    region_tracker.load_dino()

    show_corners = True
    show_mask = True
    show_bg = True
    show_layout = True
    show_text_lines = True
    show_tracker = True

    print(
        "Stage 1+2+3+4+5 preview running.\n"
        "  d — toggle detected board corners\n"
        "  s — toggle person-mask overlay\n"
        "  b — toggle Stage 3 background composite\n"
        "  l — toggle Stage 4 layout bounding boxes\n"
        "  t — toggle Stage 5 text line bounding boxes\n"
        "  r — toggle Stage 5 region tracker\n"
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
            regions = layout_detector.process(composite)

            # Stage 4: Run text detection (always run to feed tracker)
            regions_with_lines = text_detector.process(regions)

            # --- Stage 5: Prepare detections for tracker ---
            all_detections = []
            for r in regions_with_lines:
                rx1, ry1, _, _ = r.bbox
                for line in r.lines:
                    lx1, ly1, lx2, ly2 = line.bbox
                    # Translate local region coords to global composite coords
                    global_bbox = (rx1 + lx1, ry1 + ly1, rx1 + lx2, ry1 + ly2)
                    all_detections.append(
                        Detection(bbox=global_bbox, confidence=line.confidence)
                    )

            # Update Tracker using the Background Composite as the truth source
            tracker_result = region_tracker.process(all_detections, composite)

            # Visualisation overlays
            vis_composite = composite.copy()
            if show_layout:
                vis_composite = _draw_layout(vis_composite, regions)
            if show_text_lines:
                vis_composite = _draw_text_lines(vis_composite, regions_with_lines)
            if show_tracker:
                vis_composite = _draw_tracker(vis_composite, tracker_result.regions)

            cv2.imshow("Stage 3+4+5", vis_composite)

        key = cv2.waitKey(wait_ms) & 0xFF
        if key == ord("q"):
            break
        if key == ord("d"):
            show_corners = not show_corners
        if key == ord("s"):
            show_mask = not show_mask
        if key == ord("b"):
            show_bg = not show_bg
            if not show_bg:
                cv2.destroyWindow("Stage 3+4+5")
        if key == ord("l"):
            show_layout = not show_layout
        if key == ord("t"):
            show_text_lines = not show_text_lines
        if key == ord("r"):
            show_tracker = not show_tracker

    cv2.destroyAllWindows()


if __name__ == "__main__":
    raw = sys.argv[1] if len(sys.argv) > 1 else "0"
    src = int(raw) if raw.isdigit() else raw
    main(src)
