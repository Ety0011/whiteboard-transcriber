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
    r  — toggle Stage 6 entity overlay
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

import capture
from anchor_service.detector import AnchorDetector
from anchor_service.entity_registry import EntityRegistry, EntityState
from anchor_service.grouper import EntityGrouper
from board_service.board_masker import BoardMasker
from board_service.person_masker import PersonMasker
from board_service.reconstructor import BoardReconstructor
from board_service.rectifier import Rectifier
from ledger_service import assembly
from ledger_service.registry import LedgerRegistry
from transcriber_service.transcriber import MockTranscriber

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Debug drawing helpers
# ---------------------------------------------------------------------------

_CORNER_LABELS = ["TL", "TR", "BR", "BL"]

_STATE_COLORS = {
    EntityState.STABILIZING: (0, 165, 255),
    EntityState.INFERRING:   (0, 200, 255),
    EntityState.ACTIVE:      (0, 230, 0),
    EntityState.ERASED:      (0, 0, 220),
}


def _apply_mask_overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Blend a semi-transparent red highlight over *frame* where *mask* is 1."""
    overlay = frame.copy()
    overlay[mask == 1] = (0, 0, 220)
    return cv2.addWeighted(frame, 0.65, overlay, 0.35, 0)


def _draw_corners(frame: np.ndarray, corners: np.ndarray | None) -> np.ndarray:
    if corners is None:
        cv2.putText(
            frame,
            "Detecting board...",
            (40, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 180, 220),
            2,
            cv2.LINE_AA,
        )
        return frame

    pts = corners.astype(np.int32)
    cv2.polylines(
        frame, [pts.reshape(-1, 1, 2)], isClosed=True, color=(0, 0, 220), thickness=3,
        lineType=cv2.LINE_AA,
    )
    for i, (x, y) in enumerate(pts):
        cv2.circle(frame, (int(x), int(y)), 12, (0, 200, 0), -1, cv2.LINE_AA)
        cv2.putText(
            frame,
            _CORNER_LABELS[i],
            (int(x) + 14, int(y) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return frame


def _draw_anchors(frame: np.ndarray, anchors: list) -> np.ndarray:
    overlay = frame.copy()
    for a in anchors:
        x1, y1, x2, y2 = a.bbox
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 150, 0), -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 150, 0), 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    return frame


def _draw_entities(frame: np.ndarray, regions: list) -> np.ndarray:
    overlay = frame.copy()
    for reg in regions:
        x1, y1, x2, y2 = reg.bbox
        color = _STATE_COLORS.get(reg.state, (255, 255, 255))

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        text_content = (reg.ocr_text or "")[:30]
        display_label = f"[{reg.state.value}] {text_content}"
        (tw, th), _ = cv2.getTextSize(display_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)

        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
        cv2.putText(
            frame,
            display_label,
            (x1 + 3, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
    return frame


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

    frame_queue = capture.start(args.source)

    log.info("Loading models …")
    board_masker = BoardMasker()
    person_masker = PersonMasker()
    rectifier = Rectifier()
    reconstructor = BoardReconstructor()
    anchor_detector = AnchorDetector()
    grouper = EntityGrouper()
    entity_registry = EntityRegistry()
    transcriber = MockTranscriber()
    ledger = LedgerRegistry()
    pending_ocr: dict[int, object] = {}  # entity_id → SemanticEntity, awaiting VLM
    log.info("Ready. Press q or Ctrl-C to stop.")

    show_corners = show_mask = show_text_lines = show_tracker = True
    frame_count = 0

    try:
        while True:
            frame = frame_queue.get()
            if frame is None:
                log.info("End of stream.")
                break

            frame_count += 1

            # Stage 1 — board mask (SAM, async, ~10s cadence)
            board_mask = board_masker.segment(frame)
            # Stage 2 — person mask (MediaPipe, sync, per-frame)
            person_mask = person_masker.segment(frame)
            # Stage 3+4 — rectify every frame using cached homography
            rect_frame, rect_mask = rectifier.rectify(frame, board_mask, person_mask)
            composite = reconstructor.update(rect_frame, rect_mask)

            # Stage 5 — anchor discovery (async, non-blocking)
            detector_result = anchor_detector.detect(composite)

            # Stage 6 — group anchors into Semantic Entities
            groups = grouper.group(detector_result.anchors)
            # Stage 6 → entity lifecycle (cross-frame persistence)
            entity_update = entity_registry.tick(groups, composite)
            # Stage 7 — submit newly dispatched entities to VLM (non-blocking)
            for entity in entity_update.newly_inferring:
                x1, y1, x2, y2 = entity.bbox
                crop = composite[y1:y2, x1:x2]
                if crop.size > 0:
                    pending_ocr[entity.id] = entity
                    transcriber.submit(entity.id, crop)

            # Poll VLM results — update ledger and synthesise output files
            for result in transcriber.get_results():
                entity = pending_ocr.pop(result.entity_id, None)
                if entity is not None:
                    entity_registry.mark_active(entity, result.text, confidence=1.0)
                    ledger.update(entity.id, entity.bbox, result.text)
                    assembly.synthesize(ledger, output_path.parent)
                    log.debug("Ledger written for entity %d", entity.id)

            # Stage 8 — erasure events
            for entity in entity_update.newly_erased:
                ledger.mark_erased(entity.id)
                assembly.synthesize(ledger, output_path.parent)

            if args.debug:
                display = frame.copy()
                if show_mask:
                    display = _apply_mask_overlay(display, person_mask)
                if show_corners and rectifier.cached_corners is not None:
                    display = _draw_corners(display, rectifier.cached_corners)
                cv2.putText(
                    display,
                    f"Frame: {frame_count} | STAGE 1+2: INPUT TRACKING",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.imshow("raw", display)

                board_display = composite.copy()
                if show_text_lines:
                    board_display = _draw_anchors(
                        board_display, detector_result.anchors
                    )
                if show_tracker:
                    board_display = _draw_entities(
                        board_display, entity_update.entities
                    )
                cv2.imshow("board", board_display)

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
        board_masker.shutdown()
        anchor_detector.shutdown()
        transcriber.shutdown()
        cv2.destroyAllWindows()

    log.info("Shutting down.")


if __name__ == "__main__":
    main()
