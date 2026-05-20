"""Interactive MLX Detector Test Script.

Controls:
    Spacebar — Pause/Resume video stream and trigger a fresh PaddleOCR-VL-1.5 inference pass
    'q'      — Quit the test script
"""

import re
import sys

import cv2
import numpy as np
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from PIL import Image

MODEL_ID = "mlx-community/PaddleOCR-VL-1.5-8bit"
TARGET_W = 1280
TARGET_H = 720


def parse_tokens(raw_text: str, h: int, w: int) -> list[dict]:
    """Parse coordinate tokens from PaddleOCR-VL-1.5 matching [X, Y] format."""
    regions = []
    tokens_pattern = re.compile(r"((?:<\|LOC_\d+\|>)+)")
    parts = tokens_pattern.split(raw_text)

    for i in range(0, len(parts) - 1, 2):
        content = parts[i].strip()
        loc_block = parts[i + 1]

        if not content or not loc_block:
            continue

        coords = [int(val) for val in re.findall(r"\d+", loc_block)]
        if len(coords) < 6 or len(coords) % 2 != 0:
            continue

        poly_pts = []
        for j in range(0, len(coords), 2):
            # CORRECT ORDER: token_x comes first, token_y comes second
            token_x, token_y = coords[j], coords[j + 1]

            # Direct linear percentage mapping to target frame width and height
            abs_x = int((token_x / 1000.0) * w)
            abs_y = int((token_y / 1000.0) * h)

            # Guard clamping constraints
            abs_x = max(0, min(w, abs_x))
            abs_y = max(0, min(h, abs_y))
            poly_pts.append([abs_x, abs_y])

        pts_arr = np.array(poly_pts, dtype=np.int32)

        if "$$" in content or "\\(" in content or "e^{" in content or "=" in content:
            label = "MATH"
            color = (0, 200, 255)
        elif len(content) < 3 and not content.isalnum():
            label = "DIAGRAM"
            color = (255, 100, 0)
        else:
            label = "TEXT"
            color = (0, 230, 0)

        regions.append(
            {"text": content, "poly": pts_arr, "label": label, "color": color}
        )

    return regions


def main():
    if len(sys.argv) < 2:
        print("Error: Missing video file source path.")
        print("Usage: python test_detector.py path/to/video.mp4")
        sys.exit(1)

    video_path = sys.argv[1]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_path!r}")
        sys.exit(1)

    print(f"[MLX Test] Loading native model framework: {MODEL_ID}...")
    model, processor = load(MODEL_ID)
    config = model.config

    print("\n" + "=" * 60)
    print(" CONTROLS:")
    print("   [Spacebar] -> Trigger layout discovery on the current frame")
    print("   [q]        -> Quit script")
    print("=" * 60 + "\n")

    frame_count = 0
    current_regions = []
    paused_on_inference = False
    last_frame = None

    try:
        while True:
            # If we just ran an inference pass, hold the frame until spacebar releases it
            if paused_on_inference:
                key = (
                    cv2.waitKey(0) & 0xFF
                )  # Blocks execution completely, waiting for key
                if key == ord(" "):
                    paused_on_inference = False
                    continue
                elif key == ord("q"):
                    break
                else:
                    continue

            ok, frame = cap.read()
            if not ok:
                print("[MLX Test] Reached end of video file stream.")
                break

            frame_count += 1

            h, w = frame.shape[:2]
            if w != TARGET_W or h != TARGET_H:
                frame = cv2.resize(
                    frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA
                )
                h, w = TARGET_H, TARGET_W

            # Create base visualization copy
            display_frame = frame.copy()

            # Draw any previously cached layout regions onto the playing stream
            if current_regions:
                overlay = display_frame.copy()
                for reg in current_regions:
                    poly = reg["poly"]
                    color = reg["color"]
                    text_slice = reg["text"]

                    cv2.fillPoly(overlay, [poly], color)
                    cv2.polylines(
                        display_frame, [poly], isClosed=True, color=color, thickness=2
                    )

                    x, y = int(poly[:, 0].min()), int(poly[:, 1].min())
                    display_label = f"[{reg['label']}] {text_slice[:30]}"
                    (tw, th), _ = cv2.getTextSize(
                        display_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
                    )

                    cv2.rectangle(
                        display_frame, (x, y - th - 6), (x + tw + 6, y), color, -1
                    )
                    cv2.putText(
                        display_frame,
                        display_label,
                        (x + 3, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 0, 0),
                        1,
                        cv2.LINE_AA,
                    )
                cv2.addWeighted(overlay, 0.25, display_frame, 0.75, 0, display_frame)

            # Draw standard HUD instructions
            cv2.putText(
                display_frame,
                f"Frame: {frame_count} | Press [SPACE] to freeze and run layout discovery",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow("PaddleOCR-VL-1.5 Grounding Test", display_frame)

            # Non-blocking wait while playing video normally
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                # SPACEBAR TRIPPED: Run layout grounding immediately on this specific frame
                print(
                    f"\n[Inference Pass] Freezing and evaluating Frame #{frame_count}..."
                )

                # Update HUD to show calculation state
                processing_ui = display_frame.copy()
                cv2.rectangle(processing_ui, (20, 50), (320, 90), (0, 0, 0), -1)
                cv2.putText(
                    processing_ui,
                    "RUNNING MLX INFERENCE...",
                    (30, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 165, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("PaddleOCR-VL-1.5 Grounding Test", processing_ui)
                cv2.waitKey(1)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)

                formatted_prompt = apply_chat_template(
                    processor, config, "Spotting:", num_images=1
                )

                gen_result = generate(
                    model,
                    processor,
                    formatted_prompt,
                    pil_img,
                    max_tokens=512,
                    verbose=False,
                )

                raw_text = (
                    gen_result.text if hasattr(gen_result, "text") else str(gen_result)
                )

                # Overwrite cached regions with the clear layout parsing results
                current_regions = parse_tokens(raw_text, h, w)
                print(f"-> Extracted {len(current_regions)} active regions.")
                print(f"-> Raw Text: {raw_text[:200]}...")
                print("=========================================================")
                print("FRAME FROZEN. Press [Spacebar] again to resume playback.")
                print("=========================================================")

                # Re-render the frozen frame with the updated boxes immediately
                frozen_overlay = frame.copy()
                for reg in current_regions:
                    poly = reg["poly"]
                    color = reg["color"]
                    text_slice = reg["text"]
                    cv2.fillPoly(frozen_overlay, [poly], color)
                    cv2.polylines(
                        frame, [poly], isClosed=True, color=color, thickness=2
                    )
                    x, y = int(poly[:, 0].min()), int(poly[:, 1].min())
                    display_label = f"[{reg['label']}] {text_slice[:30]}"
                    (tw, th), _ = cv2.getTextSize(
                        display_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
                    )
                    cv2.rectangle(frame, (x, y - th - 6), (x + tw + 6, y), color, -1)
                    cv2.putText(
                        frame,
                        display_label,
                        (x + 3, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 0, 0),
                        1,
                        cv2.LINE_AA,
                    )
                cv2.addWeighted(frozen_overlay, 0.25, frame, 0.75, 0, frame)

                cv2.putText(
                    frame,
                    f"FROZEN Frame: {frame_count} | Press [SPACE] to resume",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("PaddleOCR-VL-1.5 Grounding Test", frame)

                paused_on_inference = True

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[MLX Test] Window destroyed.")


if __name__ == "__main__":
    main()
