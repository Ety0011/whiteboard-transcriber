"""Whiteboard transcription pipeline — entry point.

Usage::

    python src/main.py                           # live webcam, hierarchical_union_find
    python src/main.py video.mp4                 # video file
    python src/main.py --model yolo video.mp4    # YOLO backend

Keyboard controls:
    q      — quit
    a      — toggle auto/manual Stage 5 detection
    Space  — manual Stage 5 submit (manual mode only)
    w      — toggle Stage 1/2 corner overlay
    p      — toggle Stage 1/2 body-mask overlay
    t      — toggle Stage 5 block overlay
    r      — toggle Stage 6 entity overlay
"""

from __future__ import annotations

import argparse
import logging
from functools import partial
from pathlib import Path

import cv2
import numpy as np

import capture
from board_service.board_masker import BoardMasker
from board_service.person_masker import PersonMasker
from board_service.reconstructor import BoardReconstructor
from board_service.rectifier import Rectifier
from discovery import Discovery
from layout_service import (
    Block,
    DBSCANGrouper,
    DocLayoutDetector,
    HDBSCANGrouper,
    PaddleVLDetector,
    StrokeDetector,
    TextBlockDetector,
    UnionFindGrouper,
    XYCutGrouper,
    YOLODetector,
)
from ledger_service import assembly
from ledger_service.registry import LedgerRegistry
from registry import EntityState, Registry
from transcriber_service.transcriber import MockTranscriber

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label → display colour (BGR)
# ---------------------------------------------------------------------------

_LABEL_COLORS: dict[str, tuple[int, int, int]] = {
    "TEXT": (255, 165, 0),
    "MATH": (0, 200, 255),
    "TABLE": (255, 255, 0),
    "DIAGRAM": (255, 100, 0),
}

_STATE_COLORS = {
    EntityState.STABILIZING: (0, 165, 255),
    EntityState.INFERRING: (0, 200, 255),
    EntityState.ACTIVE: (0, 230, 0),
    EntityState.ERASED: (0, 0, 220),
}

_CORNER_LABELS = ["TL", "TR", "BR", "BL"]

# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def _apply_mask_overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
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
        frame,
        [pts.reshape(-1, 1, 2)],
        isClosed=True,
        color=(0, 0, 220),
        thickness=3,
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


def _draw_blocks(frame: np.ndarray, blocks: list[Block]) -> np.ndarray:
    overlay = frame.copy()
    for block in blocks:
        color = _LABEL_COLORS.get(block.label, (255, 255, 255))
        pts = block.poly.reshape(-1, 1, 2)
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(
            frame, [pts], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA
        )
        label_txt = f"{block.label} ({block.confidence:.0%})"
        x1, y1 = int(block.poly[:, 0].min()), int(block.poly[:, 1].min())
        (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            frame,
            label_txt,
            (x1 + 2, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
    return frame


def _draw_entities(frame: np.ndarray, entities: list) -> np.ndarray:
    overlay = frame.copy()
    for ent in entities:
        x1, y1, x2, y2 = ent.bbox
        color = _STATE_COLORS.get(ent.state, (255, 255, 255))

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        text_content = (ent.ocr_text or "")[:30]
        display_label = f"[{ent.state.value}] {text_content}"
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
    parser = argparse.ArgumentParser(description="Whiteboard transcription pipeline")
    parser.add_argument(
        "source",
        nargs="?",
        metavar="FILE",
        help="video or image file (omit to use the default webcam)",
    )
    parser.add_argument(
        "--model",
        choices=[
            "stroke_cluster",
            "yolo",
            "doclayoutv3",
            "paddleocrvl",
            "hierarchical_union_find",
            "dbscan",
            "hdbscan",
            "xycut",
        ],
        default="hierarchical_union_find",
        help="Stage 5 layout detection backend (default: hierarchical_union_find)",
    )
    args = parser.parse_args()

    output_path = Path("output/whiteboard.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame_queue = capture.start(args.source)

    log.info("Loading models …")
    board_masker = BoardMasker()
    person_masker = PersonMasker()
    rectifier = Rectifier()
    reconstructor = BoardReconstructor()

    factories = {
        "stroke_cluster": StrokeDetector,
        "yolo": YOLODetector,
        "doclayoutv3": DocLayoutDetector,
        "paddleocrvl": PaddleVLDetector,
        "hierarchical_union_find": partial(
            TextBlockDetector, strategy=UnionFindGrouper()
        ),
        "dbscan": partial(TextBlockDetector, strategy=DBSCANGrouper()),
        "hdbscan": partial(TextBlockDetector, strategy=HDBSCANGrouper()),
        "xycut": partial(TextBlockDetector, strategy=XYCutGrouper()),
    }

    discovery = Discovery(factory=factories[args.model])
    registry = Registry()
    transcriber = MockTranscriber()
    ledger = LedgerRegistry()
    pending_ocr: dict[int, object] = {}
    log.info("Ready. Model: %s | Press q or Ctrl-C to stop.", args.model)

    frame_count = 0
    auto_mode = True
    status_msg = f"Model: {args.model.upper()} | AUTO"
    last_blocks: list[Block] = []

    show_corners = True
    show_mask = True
    show_blocks = True
    show_tracker = True

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

            # Stage 5 — layout detection (async, non-blocking)
            if auto_mode:
                blocks, latency = discovery.detect(composite)
                if latency:
                    status_msg = (
                        f"Model: {args.model.upper()} | AUTO | {latency * 1000:.1f}ms"
                    )
            else:
                blocks, latency = discovery.poll()

            if blocks:
                last_blocks = blocks

            # Stage 6 — entity lifecycle (cross-frame persistence)
            entity_update = registry.tick(blocks, composite)

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
                    registry.mark_active(entity, result.text, confidence=1.0)
                    ledger.update(entity.id, entity.bbox, result.text)
                    assembly.synthesize(ledger, output_path.parent)
                    log.debug("Ledger written for entity %d", entity.id)

            # Stage 8 — erasure events
            for entity in entity_update.newly_erased:
                ledger.mark_erased(entity.id)
                assembly.synthesize(ledger, output_path.parent)

            # -----------------------------------------------------------------------
            # Render — board composite
            # -----------------------------------------------------------------------

            board_display = composite.copy()
            if show_blocks:
                board_display = _draw_blocks(board_display, last_blocks)
            if show_tracker:
                board_display = _draw_entities(
                    board_display,
                    [
                        e
                        for e in entity_update.entities
                        if e.state != EntityState.ERASED
                    ],
                )

            cv2.putText(
                board_display,
                f"Frame: {frame_count} | {'AUTO' if auto_mode else 'MANUAL'}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                board_display,
                status_msg,
                (20, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.circle(
                board_display,
                (board_display.shape[1] - 30, 30),
                10,
                (0, 165, 255) if discovery.is_busy else (0, 255, 0),
                -1,
            )
            cv2.imshow("Whiteboard", board_display)

            # -----------------------------------------------------------------------
            # Render — raw input with stage 1/2 overlays
            # -----------------------------------------------------------------------

            raw_display = frame.copy()
            if show_mask:
                raw_display = _apply_mask_overlay(raw_display, person_mask)
            if show_corners and rectifier.cached_corners is not None:
                raw_display = _draw_corners(raw_display, rectifier.cached_corners)
            cv2.putText(
                raw_display,
                "STAGE 1+2: INPUT TRACKING",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow("Raw Input", raw_display)

            # -----------------------------------------------------------------------
            # Keyboard
            # -----------------------------------------------------------------------

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                log.info("[q] Quit")
                break
            elif key == ord("a"):
                auto_mode = not auto_mode
                mode = "AUTO" if auto_mode else "MANUAL"
                status_msg = f"Model: {args.model.upper()} | {mode}"
                log.info("[a] Mode → %s", mode)
            elif key == ord(" ") and not auto_mode:
                blocks, latency = discovery.detect(composite)
                status_msg = (
                    f"Model: {args.model.upper()} | MANUAL | {latency * 1000:.0f}ms"
                )
                log.info("[space] Manual submit | latency=%.0fms", latency * 1000)
            elif key == ord("w"):
                show_corners = not show_corners
                log.info("[w] Corners → %s", "ON" if show_corners else "OFF")
            elif key == ord("p"):
                show_mask = not show_mask
                log.info("[p] Mask → %s", "ON" if show_mask else "OFF")
            elif key == ord("t"):
                show_blocks = not show_blocks
                log.info("[t] Blocks → %s", "ON" if show_blocks else "OFF")
            elif key == ord("r"):
                show_tracker = not show_tracker
                log.info("[r] Entities → %s", "ON" if show_tracker else "OFF")

    except KeyboardInterrupt:
        pass
    finally:
        board_masker.shutdown()
        discovery.shutdown()
        transcriber.shutdown()
        cv2.destroyAllWindows()

    log.info("Shutting down.")


if __name__ == "__main__":
    main()
