"""Whiteboard transcription pipeline — entry point.

Orchestrates preprocessing, homography rectification, specular-free surface reconstruction,
and asynchronous background MLX layout discovery via PaddleOCR-VL-1.5, backed by a
temporal lifecycle tracking registry.

Usage::

    python src/main.py                    # live webcam (default)
    python src/main.py video.mp4          # video file
    python src/main.py --debug            # webcam + full debug overlays

Keyboard controls (debug mode only):
    q  — quit
    w  — toggle Stage 1/2 corner quad overlays
    p  — toggle Stage 1/2 human body-mask highlights
"""

from __future__ import annotations

import argparse
import logging

import cv2
import numpy as np

import capture
from anchor_service.detector import AnchorDetector
from anchor_service.entity_registry import EntityRegistry, RegionState
from board_service.board_masker import BoardMasker
from board_service.person_masker import PersonMasker
from board_service.reconstructor import BoardReconstructor
from board_service.rectifier import Rectifier

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

TARGET_W = 1280
TARGET_H = 720

_CORNER_LABELS = ["TL", "TR", "BR", "BL"]

# State Machine Color Coding matching the interactive test engine specs
_STATE_COLORS = {
    RegionState.STABILIZING: (0, 165, 255),  # Vivid Orange during initial layout write
    RegionState.STABLE: (0, 230, 0),  # Emerald Green when settled and validated
    RegionState.ERASED: (0, 0, 220),  # Deep Red if archived into background buffers
}

# ---------------------------------------------------------------------------
# Core Visualization Engines
# ---------------------------------------------------------------------------


def _apply_mask_overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Blend a semi-transparent crimson highlight over frame where mask is 1."""
    overlay = frame.copy()
    overlay[mask == 1] = (0, 0, 220)
    return cv2.addWeighted(frame, 0.65, overlay, 0.35, 0)


def _draw_corners(frame: np.ndarray, corners: np.ndarray | None) -> np.ndarray:
    """Draw the detected board quad and corner labels on frame."""
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


def _render_registry_overlay(
    target_canvas: np.ndarray,
    tracked_regions: list,
    model_h: int,
    model_w: int,
    display_h: int,
    display_w: int,
) -> np.ndarray:
    """Rescale and draw transparent state-machine tracking polygons directly onto the viewport."""
    if not tracked_regions:
        return target_canvas

    overlay = target_canvas.copy()
    scale_x = display_w / model_w
    scale_y = display_h / model_h

    for track in tracked_regions:
        if track.raw_polygon is None:
            continue

        # Rescale model space coordinates down to UI target window boundaries
        poly_display = (track.raw_polygon * np.array([scale_x, scale_y])).astype(
            np.int32
        )
        color = _STATE_COLORS.get(track.state, (0, 230, 0))

        cv2.fillPoly(overlay, [poly_display], color)
        cv2.polylines(
            target_canvas,
            [poly_display],
            isClosed=True,
            color=color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        x, y = int(poly_display[:, 0].min()), int(poly_display[:, 1].min())
        display_label = f"ID:{track.id} [{track.state.value}] {track.text[:25]}"
        (tw, th), _ = cv2.getTextSize(display_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)

        cv2.rectangle(target_canvas, (x, y - th - 6), (x + tw + 6, y), color, -1)
        cv2.putText(
            target_canvas,
            display_label,
            (x + 3, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return cv2.addWeighted(overlay, 0.25, target_canvas, 0.75, 0)


# ---------------------------------------------------------------------------
# Main Orchestration Hub
# ---------------------------------------------------------------------------


def main() -> None:
    """Orchestrate baseline streaming queues and handle asynchronous visual grounding pipeline stages."""
    parser = argparse.ArgumentParser(description="Whiteboard processing pipeline core")
    parser.add_argument(
        "source",
        nargs="?",
        metavar="FILE",
        help="video or image file (omit to use the default webcam)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="show transformation stage overlays and enable debug logging",
    )
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.DEBUG if args.debug else logging.INFO)

    frame_queue = capture.start(args.source)

    log.info("Loading baseline stabilization models…")
    board_masker = BoardMasker()
    person_masker = PersonMasker()
    rectifier = Rectifier()
    reconstructor = BoardReconstructor()

    log.info("Spawning non-blocking MLX Layout Discovery background process…")
    anchor_detector = AnchorDetector()

    log.info("Initializing temporal state machine lifecycle registry…")
    entity_registry = EntityRegistry(stability_window=2.0, erasure_grace_period=4.0)

    log.info("Lecture Historian fully running. Press q or Ctrl-C to stop.")

    frame_count = 0
    show_corners = show_mask = True
    cached_tracks = []

    try:
        while True:
            frame = frame_queue.get()
            if frame is None:
                log.info("End of stream.")
                break

            frame_count += 1

            # Scale camera/video frame to UI display target metrics
            h, w = frame.shape[:2]
            if w != TARGET_W or h != TARGET_H:
                frame = cv2.resize(
                    frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA
                )
                h, w = TARGET_H, TARGET_W

            # Stage 1 — board mask (SAM, async background execution)
            board_mask = board_masker.segment(frame)
            # Stage 2 — person mask (MediaPipe, sync per-frame optimization)
            person_mask = person_masker.segment(frame)
            # Stage 3+4 — rectify frame cleanly using homography matrices
            rect_frame, rect_mask = rectifier.rectify(frame, board_mask, person_mask)
            composite = reconstructor.update(rect_frame, rect_mask)

            # Stage 5 — Asynchronous visual grounding check
            # Non-blocking pull of the latest generated regions array
            detector_result = anchor_detector.detect(composite)

            # Drive the state machinery lifecycle tracks using the pooled detector updates
            registry_update = entity_registry.tick(detector_result.regions, composite)
            cached_tracks = registry_update.regions

            # Print transactional state modifications out to terminal logs
            for stable_block in registry_update.newly_stable:
                log.info(
                    "Region Committed to Ledger [ID:%d] | Type: %s | Text: %r",
                    stable_block.id,
                    stable_block.label,
                    stable_block.text,
                )

            for erased_block in registry_update.newly_erased:
                log.info(
                    "Region Erased from Surface [ID:%d] | Archived successfully.",
                    erased_block.id,
                )

            # Extract absolute high-resolution metrics from the model workspace buffer
            comp_h, comp_w = composite.shape[:2]

            # --- Real-Time Visualisation Layer ---
            if args.debug:
                raw_canvas = frame.copy()
                if show_mask:
                    raw_canvas = _apply_mask_overlay(raw_canvas, person_mask)
                if show_corners and rectifier.cached_corners is not None:
                    raw_canvas = _draw_corners(raw_canvas, rectifier.cached_corners)

                cv2.putText(
                    raw_canvas,
                    f"Frame: {frame_count} | STAGE 1+2: INPUT TRACKING",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.imshow("raw", raw_canvas)

                # Resize high-res surface map and layer the active persistent tracking states
                board_canvas = cv2.resize(
                    composite, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA
                )
                board_canvas = _render_registry_overlay(
                    board_canvas, cached_tracks, comp_h, comp_w, TARGET_H, TARGET_W
                )

                cv2.putText(
                    board_canvas,
                    f"STAGE 3+4+5: RECONSTRUCTED LEDGER SURFACES | Active Blocks: {len(cached_tracks)}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.imshow("board", board_canvas)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("w"):
                    show_corners = not show_corners
                elif key == ord("p"):
                    show_mask = not show_mask
            else:
                cv2.putText(
                    frame,
                    "LECTURE HISTORIAN: RECORDING ACTIVE",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 220, 0),
                    1,
                    cv2.LINE_AA,
                )
                cv2.imshow("Whiteboard", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        board_masker.shutdown()
        anchor_detector.shutdown()
        cv2.destroyAllWindows()

    log.info("Lecture Historian core halted cleanly.")


if __name__ == "__main__":
    main()
