"""Whiteboard transcription pipeline — entry point.

Usage::

    python src/main.py                    # live webcam (default)
    python src/main.py video.mp4          # video file
    python src/main.py --debug            # webcam + full debug overlays
    python src/main.py --debug video.mp4  # file + full debug overlays

Keyboard controls (debug mode only):
    q  — quit
    w  — toggle Stage 1/2 corner overlay
    p  — toggle Stage 1/2 body-mask overlay
    t  — toggle Stage 5 anchor boxes
    r  — toggle Stage 6 region tracker
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

import capture
from anchor_service.detector import AnchorDetector, AnchorType
from board_service.reconstructor import BoardReconstructor
from board_service.rectifier import Rectifier
from board_service.tracker import BoardTracker
from document import WhiteboardDoc
from text_recognizer import TextRecognizer
from tracker import Detection, RegionState, RegionTracker

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Debug drawing helpers
# ---------------------------------------------------------------------------

_LABELS = ["TL", "TR", "BR", "BL"]

_STATE_COLOURS = {
    RegionState.CANDIDATE: (255, 255, 0),
    RegionState.STABILIZING: (0, 165, 255),
    RegionState.STABLE: (0, 255, 0),
    RegionState.MISSING: (200, 0, 200),
    RegionState.REMOVED: (0, 0, 255),
}


def _apply_mask_overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Blend a semi-transparent red highlight over *frame* where *mask* is 1."""
    overlay = frame.copy()
    overlay[mask == 1] = (0, 0, 220)
    return cv2.addWeighted(frame, 0.65, overlay, 0.35, 0)


def _draw_corners(frame: np.ndarray, corners: np.ndarray | None) -> np.ndarray:
    """Draw the detected board quad and corner labels on *frame*."""
    if corners is None:
        cv2.putText(
            frame,
            "Detecting board...",
            (40, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 180, 220),
            2,
        )
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


def _draw_anchors(frame: np.ndarray, anchors: list) -> np.ndarray:
    """Draw anchor bounding boxes colour-coded by type."""
    out = frame.copy()
    for a in anchors:
        x1, y1, x2, y2 = a.bbox
        colour = (255, 150, 0) if a.anchor_type == AnchorType.TEXT_LINE else (0, 200, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 1)
        cv2.putText(out, a.anchor_type.value, (x1, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, colour, 1, cv2.LINE_AA)
    return out


def _draw_tracker(frame: np.ndarray, regions: list) -> np.ndarray:
    """Draw tracked regions with ID, state, confidence, and OCR text."""
    out = frame.copy()
    for reg in regions:
        x1, y1, x2, y2 = reg.bbox
        colour = _STATE_COLOURS.get(reg.state, (255, 255, 255))
        thickness = 1 if reg.state == RegionState.MISSING else 2
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, thickness)

        ocr_tag = "OK" if reg.ocr_text else "..."
        label = f"ID:{reg.id} {reg.state.value} {reg.confidence:.2f} [{ocr_tag}]"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(out, (x1, y1), (x1 + tw + 4, y1 + th + 6), colour, -1)
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
                cv2.putText(
                    out,
                    line[:80],
                    (x1, y2 + 14 + i * 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Start the capture thread and run the pipeline loop."""
    parser = argparse.ArgumentParser(description="Whiteboard transcription pipeline")
    parser.add_argument(
        "source",
        nargs="?",
        metavar="FILE",
        help="video or image file (omit to use the default webcam)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="show stage overlays and enable debug logging",
    )
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.DEBUG if args.debug else logging.INFO)

    output_path = Path("output/whiteboard.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame_queue = capture.process(args.source)

    log.info("Loading models …")
    board_tracker = BoardTracker()
    rectifier = Rectifier()
    reconstructor = BoardReconstructor()
    anchor_detector = AnchorDetector()
    tracker = RegionTracker()
    recognizer = TextRecognizer()
    doc = WhiteboardDoc()
    log.info("Ready. Press q or Ctrl-C to stop.")

    show_corners = show_mask = show_text_lines = show_tracker = True
    composite: np.ndarray | None = None  # last clean board composite from Stage 4

    try:
        while True:
            frame = frame_queue.get()
            if frame is None:
                log.info("End of stream.")
                break

            # Stage 1+2 — board corners + body/shadow mask (async, non-blocking)
            track = board_tracker.process(frame)
            corners = track.corners

            # Stages 3+4 — only run when SAM produced a fresh (matched) frame+mask.
            # track.frame is non-None exactly once per SAM cycle; using it ensures the
            # body mask and the frame it describes are always in sync, preventing person
            # pixels from leaking into the board composite between SAM updates.
            if track.frame is not None:
                mask = track.body_mask
                warped, warped_mask = rectifier.process(track.frame, mask, corners)
                composite = reconstructor.process(warped, warped_mask)

            if composite is None:
                # Waiting for first SAM result — show raw feed and wait
                if args.debug:
                    display = frame.copy()
                    cv2.putText(display, "Waiting for SAM...", (40, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 180, 220), 2)
                    cv2.imshow("raw", display)
                    cv2.waitKey(1)
                continue

            # Stage 5 — anchor discovery (async, non-blocking)
            detector_result = anchor_detector.process(composite)
            detections = [
                Detection(
                    bbox=a.bbox,
                    confidence=a.confidence,
                    line_bboxes=[a.bbox],
                )
                for a in detector_result.anchors
            ]
            # Stage 6 — region tracker
            tracker_result = tracker.process(detections, composite)
            # Stage 7 — OCR + markdown write
            if tracker_result.newly_stable:
                recognizer.process(tracker_result, tracker, doc)
                tmp = output_path.with_suffix(".tmp")
                tmp.write_text(doc.to_markdown(), encoding="utf-8")
                tmp.rename(output_path)
                log.debug("Markdown written to %s", output_path)

            if args.debug:
                display = frame.copy()
                if show_mask and track.body_mask is not None:
                    display = _apply_mask_overlay(display, track.body_mask)
                if show_corners:
                    display = _draw_corners(display, corners)
                cv2.imshow("raw", display)

                vis = composite.copy()
                if show_text_lines:
                    vis = _draw_anchors(vis, detector_result.anchors)
                if show_tracker:
                    vis = _draw_tracker(vis, tracker_result.regions)
                cv2.imshow("board", vis)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("w"):
                    show_corners = not show_corners
                elif key == ord("p"):
                    show_mask = not show_mask
                elif key == ord("t"):
                    show_text_lines = not show_text_lines
                elif key == ord("r"):
                    show_tracker = not show_tracker
            else:
                cv2.imshow("Whiteboard", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        board_tracker.shutdown()
        anchor_detector.shutdown()
        cv2.destroyAllWindows()

    log.info("Shutting down.")


if __name__ == "__main__":
    main()
