"""Interactive Whiteboard Reconstruction + Swappable Layout Discovery Test.

Integrates Stages 1-4 board reconstruction and feeds the clean, rectified
whiteboard composite directly into a swappable Stage 5 async Layout Worker.

Usage:
    python -m src.layout video.mp4 --model stroke_cluster
    python -m src.layout video.mp4 --model yolo
    python -m src.layout video.mp4 --model doclayoutv3
    python -m src.layout video.mp4 --model paddleocrvl
    python -m src.layout video.mp4 --model hierarchical_union_find
    python -m src.layout video.mp4 --model dbscan
    python -m src.layout video.mp4 --model hdbscan
    python -m src.layout video.mp4 --model xycut
"""

import argparse
import time
from functools import partial

import cv2
import numpy as np

import capture
from anchor_service.block_registry import BlockRegistry, EntityState
from board_service.board_masker import BoardMasker
from board_service.person_masker import PersonMasker
from board_service.reconstructor import BoardReconstructor
from board_service.rectifier import Rectifier
from layout_service import (
    Anchor,
    AnchorBasedLayoutDetector,
    AnchorType,
    AnisotropicSpatialClusterer,
    BlockDiscovery,
    ConnectedComponentBFSDetector,
    DBSCANClusterer,
    EntityGroup,
    PaddleOCRVLDetector,
    PPDocLayoutV3Detector,
    RecursiveXYCutClusterer,
    UnionFindClusterer,
    YOLOLayoutDetector,
)

TARGET_W = 1280
TARGET_H = 720
_MOCK_OCR_DELAY = 1.2  # seconds


def _bbox_from_poly(poly: np.ndarray) -> np.ndarray:
    return np.array(
        [poly[:, 0].min(), poly[:, 1].min(), poly[:, 0].max(), poly[:, 1].max()],
        dtype=np.int32,
    )


def _regions_to_entity_groups(regions: list[dict]) -> list[EntityGroup]:
    """Convert BaseLayoutDetector list[dict] output to EntityGroup list."""
    groups = []
    for r in regions:
        bbox = _bbox_from_poly(r["poly"])
        anchor = Anchor(bbox=bbox, confidence=1.0, anchor_type=AnchorType.TEXT_LINE)
        groups.append(EntityGroup(anchors=[anchor], bbox=bbox, confidence=1.0))
    return groups


def main() -> None:
    """Start the capture thread and run the pipeline loop."""

    parser = argparse.ArgumentParser(
        description="Whiteboard transcription pipeline — Stage 5 Test"
    )
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
        help="Stage 5 Layout Discovery backend model to run",
    )
    args = parser.parse_args()

    frame_queue = capture.start(args.source)

    print("Loading Native Reconstruction Pipeline Models...")
    board_masker = BoardMasker()
    person_masker = PersonMasker()
    rectifier = Rectifier()
    reconstructor = BoardReconstructor()

    factories = {
        "stroke_cluster": ConnectedComponentBFSDetector,
        "yolo": YOLOLayoutDetector,
        "doclayoutv3": PPDocLayoutV3Detector,
        "paddleocrvl": PaddleOCRVLDetector,
        "hierarchical_union_find": partial(
            AnchorBasedLayoutDetector, strategy=UnionFindClusterer()
        ),
        "dbscan": partial(AnchorBasedLayoutDetector, strategy=DBSCANClusterer()),
        "hdbscan": partial(
            AnchorBasedLayoutDetector, strategy=AnisotropicSpatialClusterer()
        ),
        "xycut": partial(AnchorBasedLayoutDetector, strategy=RecursiveXYCutClusterer()),
    }

    stage5 = BlockDiscovery(
        factory=factories[args.model],
        target_w=TARGET_W,
        target_h=TARGET_H,
    )

    registry = BlockRegistry()

    print("\n" + "=" * 60)
    print(f" PIPELINE ACTIVE. Active Stage 5 Model: {args.model.upper()}")
    print(" CONTROLS:  [Space] manual submit  [a] auto-mode  [q] quit")
    print("=" * 60 + "\n")

    frame_count = 0
    auto_mode = False
    status_msg = "Idle - Press [SPACE] to evaluate composite"

    _mock_infer: dict[int, float] = {}
    prev_entities: dict[int, object] = {}

    state_themes = {
        "STABILIZING": ((0, 165, 255), "STABILIZING"),
        "INFERRING": ((255, 255, 0), "INFERRING..."),
        "ACTIVE": ((0, 230, 0), "ACTIVE"),
        "ERASED": ((0, 0, 220), "ERASED"),
    }

    try:
        while True:
            frame = frame_queue.get()
            if frame is None:
                print("[Pipeline Loop] End of stream.")
                break

            frame_count += 1

            # Stages 1-4: Whiteboard Reconstruction
            board_mask = board_masker.segment(frame)
            person_mask = person_masker.segment(frame)
            rect_frame, rect_mask = rectifier.rectify(frame, board_mask, person_mask)
            composite = reconstructor.update(rect_frame, rect_mask)

            # Stage 5: Layout Detection (non-blocking, subprocess)
            if auto_mode:
                regions, latency = stage5.detect(composite)
                if latency:
                    status_msg = f"Last Layout Latency: {latency * 1000:.1f}ms"
            else:
                regions, latency = stage5.poll()

            # Stage 6: Resolve mock OCR for ready INFERRING entities
            now = time.monotonic()
            for eid, ready_at in list(_mock_infer.items()):
                if now >= ready_at:
                    ent = prev_entities.get(eid)
                    if ent is not None and ent.state == EntityState.INFERRING:
                        registry.mark_active(ent, f"Mock OCR: Entity {eid}", 1.0)
                    _mock_infer.pop(eid, None)

            # Stage 6: Tick BlockRegistry
            groups = _regions_to_entity_groups(regions)
            update = registry.tick(groups, composite)

            for ent in update.newly_inferring:
                if ent.id not in _mock_infer:
                    _mock_infer[ent.id] = now + _MOCK_OCR_DELAY

            prev_entities = {e.id: e for e in update.entities}

            # Render
            board_display = composite.copy()
            active_entities = [
                e for e in update.entities if e.state != EntityState.ERASED
            ]

            if active_entities:
                overlay = board_display.copy()
                for entity in active_entities:
                    x1, y1, x2, y2 = entity.bbox.tolist()
                    poly = np.array(
                        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32
                    )
                    color, state_lbl = state_themes.get(
                        entity.state.value, ((255, 255, 255), "UNKNOWN")
                    )
                    cv2.fillPoly(overlay, [poly], color)
                    cv2.polylines(
                        board_display, [poly], isClosed=True, color=color, thickness=2
                    )
                    text_slice = (entity.ocr_text or "")[:40]
                    label = f"[{entity.id} | {state_lbl}] {text_slice}"
                    (tw, th), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
                    )
                    cv2.rectangle(
                        board_display, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1
                    )
                    cv2.putText(
                        board_display,
                        label,
                        (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 0, 0),
                        1,
                        cv2.LINE_AA,
                    )
                cv2.addWeighted(overlay, 0.25, board_display, 0.75, 0, board_display)

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
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.circle(
                board_display,
                (board_display.shape[1] - 30, 30),
                10,
                (0, 165, 255) if stage5.is_busy else (0, 255, 0),
                -1,
            )

            cv2.imshow("Lecture Historian — Whiteboard", board_display)
            cv2.imshow("Raw Input", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                regions, latency = stage5.detect(composite)
                status_msg = f"Submitted - Latency: {latency * 1000:.1f}ms"
            elif key == ord("a"):
                auto_mode = not auto_mode
                status_msg = f"Auto: {'ON' if auto_mode else 'OFF'}"

    except KeyboardInterrupt:
        pass
    finally:
        board_masker.shutdown()
        stage5.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
