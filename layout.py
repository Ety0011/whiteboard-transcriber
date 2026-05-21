"""Interactive Whiteboard Reconstruction + Decoupled Layout Discovery Test.

Integrates Stages 1-4 board reconstruction and feeds the clean, rectified
whiteboard composite directly into a swappable Stage 5 async Layout Worker.

Controls:
    Spacebar — Submit current clean 'composite' board to the Async Stage 5 Layout Worker
    'a'      — Toggle Continuous (Auto) Stage 5 discovery mode on the composite
    'q'      — Quit the test script
"""

import threading
import time
from abc import ABC, abstractmethod

import cv2
import numpy as np
import torch

# Native pipeline imports
from src import capture
from src.board_service.board_masker import BoardMasker
from src.board_service.person_masker import PersonMasker
from src.board_service.reconstructor import BoardReconstructor
from src.board_service.rectifier import Rectifier

TARGET_W = 1280
TARGET_H = 720


# =====================================================================
# STAGE 5 INTERFACE: Swappable Layout Detector Base
# =====================================================================
class BaseLayoutDetector(ABC):
    """Abstract interface to decouple model architectures from pipeline execution."""

    @abstractmethod
    def load(self) -> None:
        """Initialize models, configure devices (MPS/CPU), and load weights."""
        pass

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Inference loop execution.
        Must return a list of dictionaries structured as:
            {
                "text": str,          # Text labels/confidence to overlay
                "poly": np.ndarray,   # Boundary coordinates, shape (N, 2), dtype=int32
                "label": str,         # Simplified taxonomy ("MATH", "TABLE", "DIAGRAM", "TEXT")
                "color": tuple        # BGR coordinate color
            }
        """
        pass


# =====================================================================
# BACKEND 1: YOLO Layout Detector (Highly Recommended)
# =====================================================================
class YOLOLayoutDetector(BaseLayoutDetector):
    """Strictly visual layout region box regression using standard YOLOv8."""

    def __init__(
        self,
        repo_id: str = "hantian/yolo-doclaynet",
        filename: str = "yolov8m-doclaynet.pt",
    ):
        self.repo_id = repo_id
        self.filename = filename
        self.model = None
        self.device = "cpu"

    def load(self):
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO

        print(
            f"[YOLOLayoutDetector] Downloading weights ({self.repo_id}/{self.filename})..."
        )
        weights_path = hf_hub_download(repo_id=self.repo_id, filename=self.filename)

        print("[YOLOLayoutDetector] Initializing YOLO model...")
        self.model = YOLO(weights_path)
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(
            f"[YOLOLayoutDetector] Loaded onto target hardware: {self.device.upper()}"
        )

    def detect(self, frame: np.ndarray) -> list[dict]:
        results = self.model.predict(
            frame, imgsz=640, conf=0.25, verbose=False, device=self.device
        )[0]

        discovered_regions = []
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            class_name = self.model.names[cls_id]
            xyxy = box.xyxy[0].cpu().numpy().astype(np.int32)
            score = float(box.conf[0].item())

            name_lower = class_name.lower()
            if "math" in name_lower or "formula" in name_lower:
                label = "MATH"
                color = (0, 200, 255)
            elif "table" in name_lower:
                label = "TABLE"
                color = (255, 255, 0)
            elif (
                "illus" in name_lower
                or "pic" in name_lower
                or "fig" in name_lower
                or "chart" in name_lower
            ):
                label = "DIAGRAM"
                color = (255, 100, 0)
            else:
                label = "TEXT"
                color = (0, 230, 0)

            x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
            poly_pts = np.array(
                [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32
            )

            discovered_regions.append(
                {
                    "text": f"{class_name.upper()} ({score:.1%})",
                    "poly": poly_pts,
                    "label": label,
                    "color": color,
                }
            )
        return discovered_regions


# =====================================================================
# BACKEND 2: PP-DocLayoutV3 (Transformers Engine)
# =====================================================================
class PPDocLayoutV3Detector(BaseLayoutDetector):
    """Irregular document segmenter. Outputs structural layout polygons."""

    def __init__(self, model_id: str = "PaddlePaddle/PP-DocLayoutV3_safetensors"):
        self.model_id = model_id
        self.model = None
        self.image_processor = None
        self.device = "cpu"

    def load(self):
        from transformers import AutoModelForObjectDetection

        try:
            from transformers import RTDetrImageProcessor as ImageProcessorClass
        except ImportError:
            from transformers import AutoImageProcessor as ImageProcessorClass

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(
            f"[PPDocLayoutV3Detector] Initializing {self.model_id} on {self.device.upper()}..."
        )

        self.image_processor = ImageProcessorClass.from_pretrained(self.model_id)
        self.model = AutoModelForObjectDetection.from_pretrained(self.model_id).to(
            self.device
        )
        self.model.eval()

    def detect(self, frame: np.ndarray) -> list[dict]:
        from PIL import Image

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        inputs = self.image_processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.image_processor.post_process_object_detection(
            outputs, target_sizes=torch.tensor([[h, w]], device=self.device)
        )[0]

        scores = results["scores"].cpu()
        labels = results["labels"].cpu()
        polygon_points_list = results.get("polygon_points", [])

        discovered_regions = []
        for idx, score in enumerate(scores):
            if score < 0.35:
                continue

            label_id = labels[idx].item()
            raw_label = self.model.config.id2label[label_id]

            name_lower = raw_label.lower()
            if "formula" in name_lower or "algorithm" in name_lower:
                label = "MATH"
                color = (0, 200, 255)
            elif "table" in name_lower:
                label = "TABLE"
                color = (255, 255, 0)
            elif "chart" in name_lower or "image" in name_lower or "pic" in name_lower:
                label = "DIAGRAM"
                color = (255, 100, 0)
            else:
                label = "TEXT"
                color = (0, 230, 0)

            # Extract irregular multi-point polygons if present, fallback to box
            if idx < len(polygon_points_list):
                poly_tensor = polygon_points_list[idx]
                if torch.is_tensor(poly_tensor):
                    poly_pts = poly_tensor.cpu().numpy().astype(np.int32)
                else:
                    poly_pts = np.array(poly_tensor, dtype=np.int32)
            else:
                box = results["boxes"][idx].cpu().numpy().astype(np.int32)
                poly_pts = np.array(
                    [
                        [box[0], box[1]],
                        [box[2], box[1]],
                        [box[2], box[3]],
                        [box[0], box[3]],
                    ],
                    dtype=np.int32,
                )

            discovered_regions.append(
                {
                    "text": f"{raw_label.upper()} ({score:.1%})",
                    "poly": poly_pts,
                    "label": label,
                    "color": color,
                }
            )
        return discovered_regions


# =====================================================================
# BACKEND 3: PaddleOCR-VL VLM (Native MLX-VLM Engine)
# =====================================================================
class PaddleOCRVLDetector(BaseLayoutDetector):
    """Autoregressive VLM layout grounding using the native 'Spotting:' chat template."""

    def __init__(self, model_id: str = "mlx-community/PaddleOCR-VL-1.5-8bit"):
        self.model_id = model_id
        self.model = None
        self.processor = None
        self.config = None

    def load(self):
        from mlx_vlm import load as load_mlx

        print(f"[PaddleOCRVLDetector] Loading native VLM on MLX: {self.model_id}...")
        self.model, self.processor = load_mlx(self.model_id)
        self.config = self.model.config

    def detect(self, frame: np.ndarray) -> list[dict]:
        import re

        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        from PIL import Image

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        formatted_prompt = apply_chat_template(
            self.processor, self.config, "Spotting:", num_images=1
        )
        gen_result = generate(
            self.model,
            self.processor,
            formatted_prompt,
            pil_img,
            max_tokens=512,
            verbose=False,
        )
        raw_text = gen_result.text if hasattr(gen_result, "text") else str(gen_result)

        # Parse XML coordinate token outputs matching LOC template [10]
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
                token_x, token_y = coords[j], coords[j + 1]
                abs_x = int((token_x / 1000.0) * w)
                abs_y = int((token_y / 1000.0) * h)
                abs_x = max(0, min(w, abs_x))
                abs_y = max(0, min(h, abs_y))
                poly_pts.append([abs_x, abs_y])

            pts_arr = np.array(poly_pts, dtype=np.int32)

            if (
                "$$" in content
                or "\\(" in content
                or "e^{" in content
                or "=" in content
            ):
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


# =====================================================================
# STAGE 5 CORE: Thread Orchestrator (Submits to whatever Backend is loaded)
# =====================================================================
class Stage5LayoutDiscovery(threading.Thread):
    """
    Spawns a generic worker thread.
    Synchronizes the mailbox and delegates calculations to the injected detector.
    """

    def __init__(
        self,
        detector: BaseLayoutDetector,
        target_w: int,
        target_h: int,
        on_regions_discovered,
    ):
        super().__init__(daemon=True)
        self.detector = detector
        self.target_w = target_w
        self.target_h = target_h
        self.on_regions_discovered = on_regions_discovered

        # Mailbox Synchronization
        self.mailbox_lock = threading.Lock()
        self.frame_mailbox: np.ndarray | None = None
        self.new_frame_event = threading.Event()
        self.is_running = True
        self.is_busy = False

    def submit_frame(self, frame: np.ndarray) -> bool:
        if self.is_busy:
            return False  # Skip execution drop frame to bypass latency backup

        with self.mailbox_lock:
            self.frame_mailbox = frame.copy()
        self.new_frame_event.set()
        return True

    def run(self):
        # Initialize selected layout detector backend
        self.detector.load()

        while self.is_running:
            self.new_frame_event.wait()
            self.new_frame_event.clear()

            with self.mailbox_lock:
                if self.frame_mailbox is None:
                    continue
                local_frame = self.frame_mailbox
                self.frame_mailbox = None

            self.is_busy = True
            start_time = time.time()

            try:
                # Perform layout discovery execution via the abstract backend interface
                discovered_regions = self.detector.detect(local_frame)
                latency = time.time() - start_time
                self.on_regions_discovered(discovered_regions, latency)

            except Exception as e:
                print(f"[Stage 5 Thread Error]: {e}")
            finally:
                self.is_busy = False

    def stop(self):
        self.is_running = False
        self.new_frame_event.set()


# =====================================================================
# STAGE 6: Hand-off Target (Tracker Registry Interface placeholder)
# =====================================================================
class Stage6TemporalRegistry:
    def __init__(self):
        self.lock = threading.Lock()
        self.current_regions = []
        self.last_latency = 0.0
        self.update_ready = False

    def receive_new_regions(self, regions: list[dict], latency: float):
        with self.lock:
            self.current_regions = regions
            self.last_latency = latency
            self.update_ready = True

    def get_regions(self) -> tuple[list[dict], float, bool]:
        with self.lock:
            ready = self.update_ready
            self.update_ready = False
            return self.current_regions, self.last_latency, ready


# =====================================================================
# Main Execution Pipeline Loop
# =====================================================================
def main() -> None:
    """Start the capture thread and run the pipeline loop."""
    import argparse

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
        choices=["yolo", "doclayoutv3", "paddleocrvl"],
        default="yolo",
        help="Stage 5 Layout Discovery backend model to run (default: yolo)",
    )
    args = parser.parse_args()

    frame_queue = capture.start(args.source)

    print("Loading Native Reconstruction Pipeline Models...")
    board_masker = BoardMasker()
    person_masker = PersonMasker()
    rectifier = Rectifier()
    reconstructor = BoardReconstructor()

    # Instantiate the selected swappable Stage 5 Layout Detector backend
    if args.model == "yolo":
        detector = YOLOLayoutDetector()
    elif args.model == "doclayoutv3":
        detector = PPDocLayoutV3Detector()
    elif args.model == "paddleocrvl":
        detector = PaddleOCRVLDetector()
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
    last_eval_latency = 0.0
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

            # Retrieve background layout detection evaluation results
            new_regions, latency, updated = registry.get_regions()
            if updated:
                active_regions = new_regions
                last_eval_latency = latency
                status_msg = f"Last Layout Latency: {last_eval_latency * 1000:.1f}ms"

            # -----------------------------------------------------------------
            # STAGE 5 Render results onto clean composite image
            # -----------------------------------------------------------------
            board_display = composite.copy()
            if active_regions:
                overlay = board_display.copy()
                for reg in active_regions:
                    poly = reg["poly"]
                    color = reg["color"]
                    text_slice = reg["text"]

                    cv2.fillPoly(overlay, [poly], color)
                    cv2.polylines(
                        board_display, [poly], isClosed=True, color=color, thickness=2
                    )

                    x, y = int(poly[:, 0].min()), int(poly[:, 1].min())
                    display_label = f"[{reg['label']}] {text_slice}"
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

            # Continuous auto-mode submission of the clean board
            if auto_mode:
                stage5_worker.submit_frame(composite)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                # Submit the reconstructed composite frame to Stage 5
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
