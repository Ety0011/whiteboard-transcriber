"""Dev tool — Stage 1–7 live preview.

Opens the camera (or a video file), runs board detection, person masking,
perspective rectification, board reconstruction, and text detection, showing
the results in cv2.imshow windows.

Usage::

    python src/debug_view.py          # webcam index 0
    python src/debug_view.py 1        # alternate camera index
    python src/debug_view.py video.mp4

Keyboard controls:
    q  — quit
    d  — toggle Stage 1 corner overlay: shows raw camera frame with detected
         quad drawn on it so you can verify corner positions before warping
    s  — toggle Stage 2 person-mask overlay (semi-transparent red, raw frame)
    b  — toggle Stage 4 board reconstruction (separate window, side-by-side)
    l  — toggle layout bounding boxes on the board reconstruction
    t  — toggle Stage 5 text line bounding boxes within each layout region
    r  — toggle Stage 6 region tracker lifecycle visualization
"""

from __future__ import annotations

import logging
import sys

import cv2
import numpy as np

from board_detector import BoardDetector
from board_reconstructor import BoardReconstructor
from capture import process as start_camera
from document import WhiteboardDoc
from layout import LayoutRegion
from person_masker import PersonMasker
from rectifier import Rectifier
from text_detector import RegionWithLines, TextDetector
from text_recognizer import TextRecognizer
from tracker import Detection, RegionState, RegionTracker

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_LABELS = ["TL", "TR", "BR", "BL"]

# BGR colours per layout label for the layout overlay.
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


def _draw_tracker(frame: np.ndarray, regions: list) -> np.ndarray:
    """Draw persistent tracked regions with IDs, States, and OCR status."""
    out = frame.copy()
    for reg in regions:
        x1, y1, x2, y2 = reg.bbox
        color = _STATE_COLOURS.get(reg.state, (255, 255, 255))

        thickness = 1 if reg.state == RegionState.MISSING else 2
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        ocr_status = "OK" if reg.ocr_text else "..."
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

        if reg.ocr_text:
            for i, line in enumerate(reg.ocr_text.splitlines()):
                ty = y2 + 14 + i * 14
                cv2.putText(
                    out,
                    line[:80],
                    (x1, ty),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
    return out


def _draw_layout(frame: np.ndarray, regions: list[LayoutRegion]) -> np.ndarray:
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


def _render_doc(doc: WhiteboardDoc, width: int = 800) -> np.ndarray:
    """Render doc.to_markdown() as a BGR image for cv2.imshow."""
    text = doc.to_markdown()
    raw_lines = text.splitlines() if text else ["(no content yet)"]

    line_h = 20
    padding = 12
    img_h = max(200, len(raw_lines) * line_h + 2 * padding)
    img = np.full((img_h, width, 3), 30, dtype=np.uint8)

    for i, line in enumerate(raw_lines):
        y = padding + (i + 1) * line_h
        cv2.putText(
            img,
            line[:110],
            (padding, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 255, 180),
            1,
            cv2.LINE_AA,
        )
    return img


def _draw_corners(frame: np.ndarray, detector: BoardDetector) -> np.ndarray:
    """Draw the detected board quad on *frame* in-place and return it."""
    with detector._lock:
        corners = detector._cached_corners

    if detector._detecting:
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
    """Run the Stage 1–7 preview loop."""
    fps = _video_fps(source)
    wait_ms = max(1, int(1000 / fps))

    frame_queue = start_camera(source)
    board_detector = BoardDetector()
    person_masker = PersonMasker()
    rectifier = Rectifier()
    reconstructor = BoardReconstructor()
    text_detector = TextDetector()
    region_tracker = RegionTracker()
    recognizer = TextRecognizer()
    doc = WhiteboardDoc()

    show_corners = True
    show_mask = True
    show_bg = True
    show_layout = True
    show_text_lines = True
    show_tracker = True
    show_doc = True

    print(
        "Stage 1–7 preview running.\n"
        "  d — toggle detected board corners\n"
        "  s — toggle person-mask overlay\n"
        "  b — toggle Stage 4 board reconstruction\n"
        "  l — toggle layout bounding boxes\n"
        "  t — toggle Stage 5 text line bounding boxes\n"
        "  r — toggle Stage 6 region tracker\n"
        "  m — toggle Markdown document view\n"
        "  q — quit"
    )

    while True:
        frame = frame_queue.get()
        if frame is None:
            logger.info("End of stream — exiting")
            break

        # Stage 1: detect board corners (async, non-blocking)
        corners = board_detector.process(frame)

        # Stage 2: person mask on raw frame
        mask_raw = person_masker.process(frame)

        # Stage 3: rectify frame and mask together
        warped, warped_mask = rectifier.process(frame, mask_raw, corners)

        display = frame.copy()

        if show_mask:
            display = _apply_mask_overlay(display, mask_raw)

        if show_corners:
            display = _draw_corners(display, board_detector)

        cv2.imshow("Stage 1+2 (raw)", display)

        if show_bg:
            # Stage 4: reconstruct clean board surface
            composite = reconstructor.process(warped, warped_mask)

            h, w = composite.shape[:2]
            full_region = LayoutRegion(
                bbox=np.array([0, 0, w, h], dtype=np.int32),
                label="text",
                confidence=1.0,
                crop=composite,
            )
            regions_with_lines = text_detector.process([full_region])

            all_detections = []
            for r in regions_with_lines:
                for line in r.lines:
                    all_detections.append(
                        Detection(
                            bbox=line.bbox,
                            confidence=line.confidence,
                            line_bboxes=[line.bbox],
                        )
                    )

            tracker_result = region_tracker.process(all_detections, composite)
            recognizer.process(tracker_result, region_tracker, doc)

            vis_composite = composite.copy()
            if show_text_lines:
                vis_composite = _draw_text_lines(vis_composite, regions_with_lines)
            if show_tracker:
                vis_composite = _draw_tracker(vis_composite, tracker_result.regions)

            cv2.imshow("Stage 4+5+6 (rectified)", vis_composite)

        if show_doc:
            cv2.imshow("Stage 7 (document)", _render_doc(doc))

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
                cv2.destroyWindow("Stage 4+5+6 (rectified)")
        if key == ord("l"):
            show_layout = not show_layout
        if key == ord("t"):
            show_text_lines = not show_text_lines
        if key == ord("r"):
            show_tracker = not show_tracker
        if key == ord("m"):
            show_doc = not show_doc
            if not show_doc:
                cv2.destroyWindow("Stage 7 (document)")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    raw = sys.argv[1] if len(sys.argv) > 1 else "0"
    src = int(raw) if raw.isdigit() else raw
    main(src)
