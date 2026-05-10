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
| Text region detection | Vision `VNRecognizeTextRequest` (bounding boxes) | CRAFT (`craft-text-detector`) |
| Semantic change detection | Custom Metal kernel (frame diff) | DINOv2-small (`facebook/dinov2-small`, HuggingFace Transformers) |
| Text OCR (primary) | Vision `VNRecognizeTextRequest` | TrOCR-small-handwritten (`microsoft/trocr-small-handwritten`) |
| Equation OCR | — | pix2tex (stubbed — `NotImplementedError`) |
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

### 3.4 Stage 4 — Region Detection (~200 ms CPU / ~80 ms GPU)

**Goal:** Locate text regions on the clean composite and assign stable spatial identifiers.

**Method:** CRAFT (Character Region Awareness For Text Detection) via the `craft-text-detector` package. CRAFT produces character-level heatmaps and affinity maps, from which bounding boxes are derived.

```python
from craft_text_detector import Craft

craft = Craft(output_dir=None, crop_type="poly", cuda=torch.cuda.is_available())

prediction_result = craft.detect_text(composite_rgb)
boxes = prediction_result["boxes"]  # list of (x, y, w, h) bounding rects
```

For each bounding box, a **spatial ID** is assigned based on the centroid's position in an 8×8 grid over the canonical 1280×720 canvas:

```python
def spatial_id(centroid_x, centroid_y, canvas_w=1280, canvas_h=720, cols=8, rows=8):
    col = int(centroid_x / canvas_w * cols)
    row = int(centroid_y / canvas_h * rows)
    return f"r{row:02d}_c{col:02d}"
```

Output: `List[Region]` where each `Region` carries `bbox` (x, y, w, h), `centroid`, and `spatial_id`.

Module: `src/region_detection.py`, entry point `process(composite: np.ndarray) -> List[Region]`.

### 3.5 Stage 5 — Change Detection (~50 ms CPU / ~20 ms GPU)

**Goal:** Determine which detected regions have actually changed since they were last processed, using semantic similarity rather than pixel-level diffing.

**Model:** DINOv2-small (`facebook/dinov2-small`, ~22 M parameters, ~80 MB). The CLS token embedding captures semantic content of a crop independently of minor lighting or perspective variation.

```python
from transformers import AutoImageProcessor, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
model = AutoModel.from_pretrained("facebook/dinov2-small")

def embed(crop_bgr: np.ndarray) -> np.ndarray:
    img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0].numpy()  # CLS token
```

**State machine per region (keyed by `spatial_id`):**

| Condition | State | Action |
|-----------|-------|--------|
| No stored embedding | **MUTATION** | Treat as new; send downstream; store embedding |
| `similarity ≥ 0.95` | **IDENTICAL** | Skip (noise, no real change) |
| `0.2 ≤ similarity < 0.95` | **MUTATION** | Content changed; send downstream; update stored embedding |
| `similarity < 0.2` | **CLEARANCE** | Region erased; mark empty; clear stored embedding |

Only **MUTATION** regions are returned and passed to Stage 6.

Module: `src/change_detection.py`, entry point `process(regions: List[Region], composite: np.ndarray, state_store: dict) -> List[Region]`.

### 3.6 Stage 6 — Content Recognition (~400 ms CPU / ~100 ms GPU)

Processes all MUTATION regions from Stage 5. Region type is determined by a stub classifier — default is `RegionType.TEXT`; equation detection is deferred.

#### 6a. Text OCR — TrOCR (Primary)

TrOCR-small-handwritten is the primary OCR engine for all text regions. It is a Vision Transformer + GPT-2 encoder-decoder trained on handwritten text and outperforms EasyOCR on whiteboard-style handwriting.

```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten")
trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-handwritten")

def recognize_region(crop_bgr: np.ndarray) -> str:
    img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    pixel_values = processor(img, return_tensors="pt").pixel_values
    generated_ids = trocr_model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

Both models are loaded once at startup. On CPU: ~200–400 ms per region; on CUDA: ~50–100 ms per region.

#### 6b. Equation OCR — pix2tex (Stub)

```python
def recognize_equation(crop_bgr: np.ndarray) -> str:
    raise NotImplementedError("pix2tex equation recognition not yet implemented")
```

When the region classifier eventually labels a region as `RegionType.EQUATION`, this stub raises `NotImplementedError`. Do not silently fall back to TrOCR for equations — the stub makes the gap explicit.

### 3.7 Stage 7 — Document Assembly (~2 ms)

**Goal:** Maintain a Markdown document where each region maps to a stable entry, updated in-place as the board changes.

**Method:**
1. Each region's `spatial_id` is the document key. Entries are ordered top-to-bottom, left-to-right by spatial ID.
2. **MUTATION:** insert or update the entry for that `spatial_id` with the new recognized text.
3. **CLEARANCE:** mark the entry with strikethrough (`~~text~~`) or remove it entirely — configurable via `CLEARANCE_STYLE` in `utils.py`.
4. No difflib deduplication needed — DINOv2 similarity gating upstream already prevents re-processing identical content.
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
opencv-contrib-python>=4.9.0   # for ArUco markers
mediapipe>=0.10.0
craft-text-detector>=0.4.2     # Stage 4: CRAFT text region detection
transformers>=4.40.0           # Stage 5: DINOv2; Stage 6: TrOCR
torch>=2.2.0
Pillow>=10.0.0
numpy>=1.26.0
scikit-learn>=1.4.0            # Stage 5: cosine_similarity
pix2tex>=0.1.2                 # Stage 6: equation OCR (stubbed)
watchdog>=4.0.0                # optional: file change notifications
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
│   ├── region_detection.py    # Stage 4: CRAFT text regions + spatial IDs
│   ├── change_detection.py    # Stage 5: DINOv2 embeddings, state machine
│   ├── recognition.py         # Stage 6: TrOCR (primary), pix2tex stub
│   ├── assembly.py            # Stage 7: Markdown emitter, MUTATION/CLEARANCE
│   ├── pipeline.py            # Orchestrates stages 1–7
│   └── utils.py               # Logging, config, state store
├── models/                    # Pre-trained model weights
│   └── .gitkeep
├── tests/
│   ├── test_registration.py
│   ├── test_region_detection.py
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
| 4. Region Detection | ~200 ms | ~80 ms | CRAFT inference |
| 5. Change Detection | ~50 ms | ~20 ms | DINOv2 CLS, one pass per region |
| 6. Recognition | ~400 ms | ~100 ms | TrOCR per MUTATION region |
| 7. Assembly | ~2 ms | ~2 ms | String operations |
| **Total** | **~730 ms** | **~250 ms** | Per processing cycle (when changes present) |

Stages 6–7 are skipped entirely for IDENTICAL regions. In practice, only a subset of regions will be in MUTATION state per cycle, keeping Stage 6 latency well below the worst-case estimate.

---

## 8. Model Summary

| Model | Purpose | Size | Source | GPU Required? |
|-------|---------|------|--------|--------------|
| MediaPipe Selfie Segmentation (landscape) | Person masking | ~450 KB | Google MediaPipe | No |
| CRAFT | Text region detection | ~100 MB | craft-text-detector | Recommended |
| DINOv2-small (`facebook/dinov2-small`) | Semantic change detection | ~80 MB | HuggingFace | Recommended |
| TrOCR-small-handwritten (`microsoft/trocr-small-handwritten`) | Primary OCR | ~250 MB | Microsoft/HuggingFace | Recommended |
| pix2tex | Equation OCR (stub) | ~250 MB | pix2tex | Recommended |

Total model footprint: ~680 MB (pix2tex deferred — not loaded until equation detection is implemented). All models download automatically on first run.

---

## 9. Accuracy Considerations

Since this project prioritizes accuracy over speed:

- **TrOCR as primary OCR:** TrOCR-small-handwritten is a Vision Transformer trained specifically on handwritten text, giving lower character error rates on whiteboard content than CRNN-based engines like EasyOCR or Tesseract.
- **DINOv2 for change detection:** pixel-level frame differencing (absdiff + threshold) produces many false positives from lighting changes and minor camera movement. DINOv2 CLS embeddings capture semantic content, so IDENTICAL regions are correctly skipped even with surface-level pixel variation.
- **CRAFT for region detection:** CRAFT is robust to perspective distortion and variable stroke widths — both common on whiteboards. It returns tighter bounding boxes than contour-based approaches, reducing background noise fed to TrOCR.
- **Spatial ID deduplication:** the `spatial_id` grid ensures that an updated region overwrites its existing document entry rather than appending a duplicate. This is the document-level equivalent of what perceptual hashing did per-frame in the previous design.
- **MOG2 with person masking:** feeding the background subtractor only unoccluded pixels prevents phantom "ink" from appearing when a person moves away from a board region.

---

## 10. Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| MediaPipe segmentation misses arms/hands | Dilate the person mask by 10–15 px to create a safety margin around detected person boundaries |
| Board detection fails (no clear edges) | Fall back to ArUco markers; or allow manual corner selection on first frame |
| CRAFT misses low-contrast or very thin text | Lower CRAFT's `text_threshold` parameter; pre-process with CLAHE to enhance local contrast |
| DINOv2 falsely marks IDENTICAL when content is similar but changed (e.g., single digit edit) | Lower the IDENTICAL threshold from 0.95; tune with fixture images |
| DINOv2 CLEARANCE false positive (board glare clears region embedding) | Add hysteresis: require CLEARANCE in N consecutive frames before acting |
| TrOCR memory usage on low-RAM machines | Load TrOCR model once at startup; do not lazy-load (startup cost acceptable; per-frame reload is not) |
| Multiple people moving simultaneously | MediaPipe handles multiple people natively; the mask is a union of all detected persons |
| pix2tex called before implementation | `recognize_equation` raises `NotImplementedError` explicitly — surfaced as a pipeline error, not a silent fallback |

---

## 11. Future Enhancements

- **GPU acceleration:** If a CUDA GPU is available, EasyOCR and TrOCR run 3–5× faster with no code changes (just `gpu=True`).
- **PaddleOCR integration:** For multi-language support or printed text dominated boards, PaddleOCR can be swapped in as the primary OCR engine.
- **Web UI:** Replace Tkinter with a Flask + WebSocket dashboard for remote viewing of the live board and Markdown output.
- **VLM post-correction:** Use a small vision-language model to verify and correct OCR output with full visual context.
- **Multi-board stitching:** Support panoramic capture of multiple boards using OpenCV feature-based stitching.
