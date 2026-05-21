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

import cv2

import capture
from board_service.board_masker import BoardMasker
from board_service.person_masker import PersonMasker
from board_service.reconstructor import BoardReconstructor
from board_service.rectifier import Rectifier
from layout_service import (
    AnchorBasedLayoutDetector,
    AnisotropicSpatialClusterer,
    ConnectedComponentBFSDetector,
    DBSCANClusterer,
    PaddleOCRVLDetector,
    PPDocLayoutV3Detector,
    RecursiveXYCutClusterer,
    Stage5LayoutDiscovery,
    Stage6TemporalRegistry,
    UnionFindClusterer,
    YOLOLayoutDetector,
)

TARGET_W = 1280
TARGET_H = 720


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

    # Instantiate the selected Stage 5 Layout Detector backend
    if args.model == "stroke_cluster":
        detector = ConnectedComponentBFSDetector()
    elif args.model == "yolo":
        detector = YOLOLayoutDetector()
    elif args.model == "doclayoutv3":
        detector = PPDocLayoutV3Detector()
    elif args.model == "paddleocrvl":
        detector = PaddleOCRVLDetector()
    elif args.model == "hierarchical_union_find":
        detector = AnchorBasedLayoutDetector(strategy=UnionFindClusterer())
    elif args.model == "dbscan":
        detector = AnchorBasedLayoutDetector(strategy=DBSCANClusterer())
    elif args.model == "hdbscan":
        detector = AnchorBasedLayoutDetector(strategy=AnisotropicSpatialClusterer())
    elif args.model == "xycut":
        detector = AnchorBasedLayoutDetector(strategy=RecursiveXYCutClusterer())
    else:
        raise ValueError(f"Unknown layout model: {args.model}")

    # Initialize Stage 6 Registry & Injected Stage 5 Worker
    registry = Stage6TemporalRegistry()
    stage5_worker = Stage5LayoutDiscovery(
        detector=detector,
        target_w=TARGET_W,
        target_h=TARGET_H,
        on_regions_discovered=registry.receive_new_regions,
    )
    stage5_worker.start()

    print("\n" + "=" * 60)
    print(f" PIPELINE ACTIVE. Active Stage 5 Model: {args.model.upper()}")
    print(" CONTROLS:")
    print(
        "   [Spacebar] -> Submit latest clean composite whiteboard frame to layout detector"
    )
    print("   [a]        -> Toggle Auto-Continuous layout mode")
    print("   [q]        -> Quit")
    print("=" * 60 + "\n")

    frame_count = 0
    active_regions = []
    auto_mode = False
    status_msg = "Idle - Press [SPACE] to evaluate composite"

    try:
        while True:
            frame = frame_queue.get()
            if frame is None:
                print("[Pipeline Loop] End of stream.")
                break

            frame_count += 1

            # -----------------------------------------------------------------
            # RUN STAGES 1-4: Whiteboard Reconstruction Pipeline
            # -----------------------------------------------------------------
            board_mask = board_masker.segment(frame)
            person_mask = person_masker.segment(frame)
            rect_frame, rect_mask = rectifier.rectify(frame, board_mask, person_mask)
            composite = reconstructor.update(rect_frame, rect_mask)

            # Retrieve background layout results
            new_regions, latency, updated = registry.get_tracked_elements()
            if updated:
                active_regions = new_regions
                status_msg = f"Last Layout Latency: {latency * 1000:.1f}ms"

            # -----------------------------------------------------------------
            # STAGES 5 & 6: Render real-time spatial-temporal tracking states
            # -----------------------------------------------------------------
            STATE_THEMES = {
                "STABILIZING": ((0, 165, 255), "STABILIZING"),  # Orange (Validating)
                "INFERRING": ((255, 255, 0), "INFERRING..."),  # Cyan (VLM queue)
                "ACTIVE": ((0, 230, 0), "ACTIVE"),  # Green (Active Ledger)
                "ERASED": ((0, 0, 220), "ERASED"),  # Red (Pruning)
            }

            board_display = composite.copy()
            if active_regions:
                overlay = board_display.copy()
                for reg in active_regions:
                    poly = reg["poly"]
                    state = reg["state"]
                    text_slice = reg["text"]

                    color, state_lbl = STATE_THEMES.get(
                        state, ((255, 255, 255), "UNKNOWN")
                    )

                    cv2.fillPoly(overlay, [poly], color)
                    cv2.polylines(
                        board_display, [poly], isClosed=True, color=color, thickness=2
                    )

                    x, y = int(poly[:, 0].min()), int(poly[:, 1].min())
                    display_label = f"[{reg['id']} | {state_lbl}] {text_slice}"
                    (tw, th), _ = cv2.getTextSize(
                        display_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
                    )

                    cv2.rectangle(
                        board_display, (x, y - th - 6), (x + tw + 6, y), color, -1
                    )
                    cv2.putText(
                        board_display,
                        display_label,
                        (x + 3, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 0, 0),
                        1,
                        cv2.LINE_AA,
                    )
                cv2.addWeighted(overlay, 0.25, board_display, 0.75, 0, board_display)

            # Draw HUD Overlays
            cv2.putText(
                board_display,
                f"Frame: {frame_count} | Mode: {'AUTO-TRACK' if auto_mode else 'MANUAL'}",
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

            if stage5_worker.is_busy:
                cv2.circle(
                    board_display,
                    (board_display.shape[1] - 30, 30),
                    10,
                    (0, 165, 255),
                    -1,
                )
            else:
                cv2.circle(
                    board_display,
                    (board_display.shape[1] - 30, 30),
                    10,
                    (0, 255, 0),
                    -1,
                )

            cv2.imshow(
                "Lecture Historian - Clean Whiteboard (Stage 4 + Stage 5)",
                board_display,
            )
            cv2.imshow("Raw Video Stream (Input)", frame)

            if auto_mode:
                stage5_worker.submit_frame(composite)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                submitted = stage5_worker.submit_frame(composite)
                if submitted:
                    status_msg = "Submitted reconstruction to background Stage 5..."
                else:
                    status_msg = "Layout Worker busy. Frame skipped."
            elif key == ord("a"):
                auto_mode = not auto_mode
                status_msg = f"Auto-discovery: {'ENABLED' if auto_mode else 'DISABLED'}"

    except KeyboardInterrupt:
        pass
    finally:
        board_masker.shutdown()
        stage5_worker.stop()
        stage5_worker.join()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
