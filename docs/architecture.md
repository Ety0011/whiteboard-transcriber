# Whiteboard Transcriber — Software Architecture (Python)

## 1. Summary

This document specifies a cross-platform Python implementation of a real-time whiteboard transcription system. The design retains the proven 7-stage pipeline from the original architecture but replaces all Apple-specific frameworks with portable, well-documented Python libraries. The system targets accuracy over raw speed: a processing cycle of 1–2 seconds is acceptable because whiteboard content changes far more slowly than the camera frame rate (a new line appears roughly every 5–10 seconds).

The entire stack runs on any machine with Python 3.11+, a webcam, and optionally a CUDA GPU. No vendor-specific hardware (Neural Engine, Metal) is required.

---

## 2. Technology Mapping

| Concern | Original (Swift/Metal) | Python Replacement |
|---------|----------------------|-------------------|
| Camera capture | AVFoundation | OpenCV `cv2.VideoCapture` |
| Person segmentation | Vision `VNGeneratePersonSegmentationRequest` | MediaPipe Selfie Segmentation |
| Perspective warp | Metal compute shader | OpenCV `cv2.warpPerspective` |
| Background model | Custom Metal kernel (running median) | OpenCV `cv2.createBackgroundSubtractorMOG2` |
| Frame differencing | Accelerate / vImage | OpenCV `cv2.absdiff` + NumPy |
| Layout classification | YOLOv11n via CoreML | YOLOv11n via Ultralytics (PyTorch) |
| Text OCR (primary) | Vision `VNRecognizeTextRequest` | EasyOCR |
| Handwriting OCR (fallback) | TrOCR-small via CoreML | TrOCR-small via HuggingFace Transformers |
| Diagram vectorization | Accelerate Hough transform | OpenCV Hough + Contour analysis |
| Table detection | Metal kernel + Vision OCR | OpenCV Hough lines + EasyOCR |
| Concurrency | GCD / `@globalActor` | Python `threading` + `queue.Queue` |
| Output sync | `DistributedNotificationCenter` | `watchdog` file observer / direct write |
| UI | SwiftUI | Tkinter or web dashboard (optional) |

---

## 3. Pipeline Architecture

The 7-stage pipeline is preserved. The primary change is that stages run sequentially within a processing thread (no GPU kernel pipelining), with the camera feed decoupled via a thread-safe queue. At 1–2 seconds per cycle, this is well within the content change rate of a whiteboard.

```
┌──────────────────┐
│  Camera Thread    │  cv2.VideoCapture · 1080p @ 30 fps
│                   │  Grabs frames, puts latest into Queue(maxsize=1)
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Processing Thread │  Pulls frame from queue, runs stages 1–7
│  (sequential)     │  Skips cycle if no changes detected (Stage 4)
└──────┬───────────┘
       │
       ▼
  Stages 1 → 2 → 3 → 4 → [5 → 6 → 7]
                           ↑
                    only if changes detected
```

### 3.1 Stage 1 — Spatial Registration (~20 ms)

**Goal:** Correct perspective so the board maps to a flat rectangle.

**Method:**
1. Detect the whiteboard quadrilateral using `cv2.Canny` + `cv2.HoughLinesP`, or `cv2.findContours` on a thresholded frame to find the largest quadrilateral.
2. Alternatively, use ArUco markers placed on the board corners (`cv2.aruco.detectMarkers`) for robust, jitter-free detection.
3. Compute the homography with `cv2.getPerspectiveTransform` or `cv2.findHomography`.
4. Warp with `cv2.warpPerspective` to a canonical resolution (e.g., 1280×720).
5. Cache the homography and only recompute every N seconds or when the board moves significantly (measured by corner displacement).

### 3.2 Stage 2 — Person Segmentation (~30–50 ms)

**Goal:** Produce a binary mask of people occluding the board.

**Library:** MediaPipe Selfie Segmentation.

```python
import mediapipe as mp

mp_selfie = mp.solutions.selfie_segmentation
segmenter = mp_selfie.SelfieSegmentation(model_selection=1)  # landscape model

rgb_frame = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2RGB)
result = segmenter.process(rgb_frame)
person_mask = (result.segmentation_mask > 0.5).astype(np.uint8)
```

`model_selection=1` uses the landscape model which handles people at whiteboard distance (1–3 m). The output is a float mask thresholded to binary.

### 3.3 Stage 3 — Surface Reconstruction (~10 ms)

**Goal:** Maintain a clean, unobstructed view of the board.

**Method:** OpenCV's MOG2 background subtractor, configured for a slowly-changing background.

```python
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=16,
    detectShadows=False
)
```

For each frame:
1. Mask out person-region pixels (set to white / board color) before feeding to MOG2, so people don't corrupt the background model.
2. The background model (`bg_subtractor.getBackgroundImage()`) provides the clean composite.
3. Composite: use the current frame where no person is present; use the stored background where the person mask is active.

```python
composite = np.where(
    person_mask[:, :, np.axis] == 1,
    background_image,
    warped_frame
)
```

### 3.4 Stage 4 — Change Detection (~5 ms)

**Goal:** Find regions with new or modified ink.

**Method:**
1. Convert current and previous composite to grayscale.
2. `cv2.absdiff` to get the difference image.
3. `cv2.threshold` (adaptive or fixed) to binarize.
4. `cv2.morphologyEx` (open + close) to clean noise.
5. `cv2.findContours` + `cv2.boundingRect` to extract changed regions.
6. Filter by minimum area (e.g., > 400 px²).
7. Perceptual hash deduplication: compute `cv2.img_hash.pHash` (or `imagehash` library) for each region. Skip regions whose hash matches an already-processed region in the hash table.

If no regions pass the filter, skip stages 5–7 entirely.

### 3.5 Stage 5 — Layout Classification (~100–200 ms)

**Goal:** Classify each changed region as text, diagram, table, or equation.

**Model:** DocLayout-YOLO or YOLOv11n fine-tuned on whiteboard layout data.

```python
from ultralytics import YOLO

layout_model = YOLO("models/doclayout_yolo.pt")

for region in changed_regions:
    crop = composite[region.y:region.y+region.h, region.x:region.x+region.w]
    results = layout_model.predict(crop, imgsz=640, conf=0.25)
    region.label = results[0].boxes.cls  # text_block, diagram, table, equation
```

DocLayout-YOLO is purpose-built for document layout analysis and comes pre-trained on DocLayNet and D4LA datasets, making it immediately usable without custom training. Alternatively, a standard YOLOv11n can be fine-tuned on a whiteboard-specific dataset.

### 3.6 Stage 6 — Content Recognition (~500 ms–1.5 s)

This is the bottleneck stage. Sub-pipelines run sequentially (or can be parallelized with `concurrent.futures.ThreadPoolExecutor` for I/O-bound OCR calls).

#### 6a. Text OCR — EasyOCR (Primary)

```python
import easyocr

reader = easyocr.Reader(['en'], gpu=True)  # gpu=False if no CUDA

for region in text_regions:
    crop = composite[region.y:region.y+region.h, region.x:region.x+region.w]
    results = reader.readtext(crop)
    # results: list of (bbox, text, confidence)
    region.text = "\n".join([r[1] for r in results if r[2] > 0.4])
    region.low_confidence_lines = [r for r in results if r[2] < 0.65]
```

EasyOCR is chosen over Tesseract because it handles handwriting and scene text significantly better out of the box, and over PaddleOCR because it has simpler installation and better handwriting character error rates (CER of 0.16 vs PaddleOCR's 0.24 on handwritten notes in independent benchmarks).

#### 6b. Handwriting OCR — TrOCR (Fallback)

For lines where EasyOCR confidence is below 0.65:

```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten")
trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-handwritten")

def recognize_line(line_crop_bgr):
    img = Image.fromarray(cv2.cvtColor(line_crop_bgr, cv2.COLOR_BGR2RGB))
    pixel_values = processor(img, return_tensors="pt").pixel_values
    generated_ids = trocr_model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

TrOCR-small (~62M parameters) is the current state-of-the-art for handwritten text recognition. On a CPU it runs at ~200–400 ms per line; with CUDA, ~50–100 ms per line. It should only be invoked for low-confidence lines (typically 10–20% of text).

#### 6c. Diagram Vectorization — OpenCV Heuristics

```python
def vectorize_diagram(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30,
                             minLineLength=20, maxLineGap=10)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    shapes = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            shapes.append(("rectangle", approx))
        elif len(approx) > 6:
            shapes.append(("circle", cv2.minEnclosingCircle(contour)))
        else:
            shapes.append(("polygon", approx))
    
    # Attempt to emit Mermaid flowchart syntax if topology matches
    # Otherwise fall back to a textual description of shapes and connections
    return build_mermaid_or_description(shapes, lines)
```

#### 6d. Table Recognition

1. Detect horizontal and vertical lines via `cv2.HoughLinesP`.
2. Cluster lines to find row/column boundaries.
3. Extract each cell as a crop.
4. Run EasyOCR on each cell.
5. Assemble into Markdown table syntax.

### 3.7 Stage 7 — Document Assembly (~2 ms)

**Goal:** Merge recognized content into a Markdown file.

**Method:**
1. Map each content block to a grid position based on its centroid in board coordinates.
2. Deduplicate: if a new block has >85% text similarity (using `difflib.SequenceMatcher`) with an existing block within a 50-pixel radius, treat as update.
3. Serialize top-to-bottom, left-to-right into Markdown.
4. Infer headings from text size (larger bounding boxes relative to stroke count → `##`).
5. Write atomically: write to a temp file, then `os.replace()` to the output path.

---

## 4. Concurrency Model

```
Main Thread
├── Camera Thread (daemon)
│   └── cv2.VideoCapture.read() → Queue(maxsize=1)
│
├── Processing Thread (daemon)  
│   └── Pulls from queue, runs stages 1–7 sequentially
│   └── Stage 6 sub-pipelines can use ThreadPoolExecutor(max_workers=3)
│
└── UI Thread (optional)
    └── Tkinter mainloop or Flask/websocket server
```

The `Queue(maxsize=1)` with `put(block=False)` / `get()` ensures the processing thread always works on the latest frame, dropping stale ones automatically. This is the Python equivalent of the back-pressure mechanism in the original architecture.

Frame pacing: the processing thread simply runs as fast as it can. With a ~1–2 second cycle time and a board that changes every 5–10 seconds, there is no need for explicit frame skipping logic — the queue naturally discards intermediate frames.

---

## 5. Dependencies

```
# requirements.txt
opencv-python>=4.9.0
opencv-contrib-python>=4.9.0   # for ArUco, img_hash
mediapipe>=0.10.0
easyocr>=1.7.0
ultralytics>=8.3.0
transformers>=4.40.0
torch>=2.2.0
Pillow>=10.0.0
numpy>=1.26.0
imagehash>=4.3.0               # perceptual hashing for dedup
watchdog>=4.0.0                 # optional: file change notifications
```

All installable via `pip install -r requirements.txt`.

---

## 6. Project Structure

```
whiteboard-transcriber/
├── src/
│   ├── __init__.py
│   ├── main.py                # Entry point, thread orchestration
│   ├── capture.py             # Stage 0: camera thread, frame queue
│   ├── registration.py        # Stage 1: perspective correction
│   ├── segmentation.py        # Stage 2: person mask (MediaPipe)
│   ├── background.py          # Stage 3: surface reconstruction (MOG2)
│   ├── change_detection.py    # Stage 4: diff, threshold, dedup
│   ├── layout.py              # Stage 5: YOLO layout classification
│   ├── recognition.py         # Stage 6: OCR, diagrams, tables
│   ├── assembly.py            # Stage 7: Markdown emitter
│   ├── pipeline.py            # Orchestrates stages 1–7
│   └── utils.py               # Logging, config, hash table
├── models/                    # Pre-trained model weights
│   └── .gitkeep
├── tests/
│   ├── test_registration.py
│   ├── test_change_detection.py
│   ├── test_recognition.py
│   ├── fixtures/              # Test images of whiteboards
│   └── conftest.py
├── docs/
│   └── architecture.md        # This document
├── output/                    # Generated Markdown files
├── requirements.txt
├── pyproject.toml
├── CLAUDE.md
└── README.md
```

---

## 7. Latency Budget

| Stage | Estimated Time (CPU) | Estimated Time (GPU) | Notes |
|-------|---------------------|---------------------|-------|
| 1. Registration | ~20 ms | ~20 ms | OpenCV, CPU-bound |
| 2. Segmentation | ~50 ms | ~30 ms | MediaPipe, light model |
| 3. Background | ~10 ms | ~10 ms | MOG2, CPU-optimized |
| 4. Change Detection | ~5 ms | ~5 ms | NumPy + OpenCV |
| 5. Layout | ~200 ms | ~50 ms | YOLO inference |
| 6. Recognition | ~1000 ms | ~400 ms | EasyOCR + TrOCR fallback |
| 7. Assembly | ~2 ms | ~2 ms | String operations |
| **Total** | **~1.3 s** | **~0.5 s** | Per processing cycle |

A whiteboard line takes 5–10 seconds to write. At 1.3 s/cycle (CPU) or 0.5 s/cycle (GPU), the system processes content 4–20× faster than it appears. This is more than sufficient for real-time transcription.

---

## 8. Model Summary

| Model | Purpose | Size | Source | GPU Required? |
|-------|---------|------|--------|--------------|
| MediaPipe Selfie Segmentation (landscape) | Person masking | ~450 KB | Google MediaPipe | No |
| DocLayout-YOLO / YOLOv11n | Layout classification | ~5–13 MB | Ultralytics / OpenDataLab | Recommended |
| EasyOCR (English) | Primary text recognition | ~100 MB | JaidedAI | Recommended |
| TrOCR-small-handwritten | Handwriting fallback | ~250 MB | Microsoft/HuggingFace | Recommended |

Total model footprint: ~370 MB. All models are downloaded automatically on first run.

---

## 9. Accuracy Considerations

Since this project prioritizes accuracy over speed:

- **EasyOCR as primary** rather than Tesseract: EasyOCR uses a CRNN architecture that handles handwritten and scene text better than Tesseract's LSTM-based engine, which was designed for clean printed documents.
- **TrOCR as fallback** for difficult handwriting: the Vision Transformer encoder captures spatial relationships in handwriting that CRNN-based models miss. Using it selectively (only on low-confidence lines) keeps the latency impact manageable.
- **Confidence gating at 0.65:** lines below this threshold get a second opinion from TrOCR. This two-pass approach catches most OCR errors while avoiding the cost of running TrOCR on every line.
- **MOG2 with person masking:** feeding the background subtractor only unoccluded pixels prevents phantom "ink" from appearing when a person moves away from a board region.
- **Perceptual hash dedup:** prevents re-processing identical content, which would otherwise accumulate duplicate entries in the output document.

---

## 10. Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| MediaPipe segmentation misses arms/hands | Dilate the person mask by 10–15 px to create a safety margin around detected person boundaries |
| Board detection fails (no clear edges) | Fall back to ArUco markers; or allow manual corner selection on first frame |
| Lighting changes cause false change detections | Operate in HSV color space for change detection; threshold on saturation channel (markers are high-saturation, lighting changes are low-saturation) |
| EasyOCR slow on CPU without GPU | Batch regions and process in a single `readtext` call; consider GPU if available |
| TrOCR memory usage on low-RAM machines | Lazy-load the model only on first fallback invocation; unload after 60 s of inactivity |
| Multiple people moving simultaneously | MediaPipe handles multiple people natively; the mask is a union of all detected persons |

---

## 11. Future Enhancements

- **GPU acceleration:** If a CUDA GPU is available, EasyOCR and TrOCR run 3–5× faster with no code changes (just `gpu=True`).
- **PaddleOCR integration:** For multi-language support or printed text dominated boards, PaddleOCR can be swapped in as the primary OCR engine.
- **Web UI:** Replace Tkinter with a Flask + WebSocket dashboard for remote viewing of the live board and Markdown output.
- **VLM post-correction:** Use a small vision-language model to verify and correct OCR output with full visual context.
- **Multi-board stitching:** Support panoramic capture of multiple boards using OpenCV feature-based stitching.
