# Whiteboard Transcriber — Software Architecture & Model Selection

## 1. Executive Summary

This document specifies the software architecture for a real-time whiteboard transcription system running entirely on-device on Apple M4 silicon (24 GB unified memory, 16-core Neural Engine at 38 TOPS). The design distributes work across three compute domains — CPU, GPU (Metal), and Neural Engine (via CoreML) — inside a multi-stage, event-driven pipeline that targets 100–200 ms end-to-end latency per processing cycle.

The architecture is split into seven pipeline stages, each mapped to the most appropriate hardware unit, and coordinated through a lock-free ring buffer that decouples producers from consumers. Every model recommended below has been selected for its ability to export to CoreML, its inference profile on Apple Silicon, and its suitability for the specific sub-task.

---

## 2. High-Level Pipeline Architecture

The system is organized as a directed acyclic graph of processing stages. Each stage reads from one or more input buffers and writes to one or more output buffers. Stages run concurrently on dedicated GCD serial queues (or Swift structured concurrency tasks), with back-pressure enforced by dropping stale frames rather than blocking upstream producers.

```
┌──────────────┐
│  Camera Feed  │  AVCaptureSession · 1080p @ 30 fps
│  (AVFoundation)│  CMSampleBuffer → CVPixelBuffer
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│ Stage 1: Spatial  │  Homography estimation + perspective warp
│ Registration      │  Metal compute shader (GPU)
└──────┬───────────┘
       │  Rectified frame (board-aligned)
       ▼
┌──────────────────┐
│ Stage 2: Person   │  VNGeneratePersonSegmentationRequest
│ Segmentation      │  Neural Engine (via Vision framework)
└──────┬───────────┘
       │  Per-pixel alpha mask
       ▼
┌──────────────────┐
│ Stage 3: Surface  │  Running median background model
│ Reconstruction    │  Metal compute shader (GPU)
└──────┬───────────┘
       │  Clean composite board image
       ▼
┌──────────────────┐
│ Stage 4: Change   │  Frame differencing + adaptive threshold
│ Detection         │  Accelerate / vImage (CPU SIMD)
└──────┬───────────┘
       │  Dirty-region bounding boxes
       ▼
┌──────────────────┐
│ Stage 5: Layout   │  YOLOv11n (CoreML, Neural Engine)
│ Classification    │  → text / diagram / table regions
└──────┬───────────┘
       │  Typed region proposals
       ▼
┌───────────────────────────────┐
│ Stage 6: Recognition          │
│  ┌─────────────────────────┐  │
│  │ 6a. Text regions        │  │  Apple Vision VNRecognizeTextRequest (.fast)
│  │     → OCR               │  │  Neural Engine
│  ├─────────────────────────┤  │
│  │ 6b. Text regions (HW)   │  │  TrOCR-small (CoreML, fallback for
│  │     → Handwriting OCR   │  │  low-confidence lines)
│  ├─────────────────────────┤  │
│  │ 6c. Diagram regions     │  │  Vectorization heuristics
│  │     → SVG primitives    │  │  CPU (Accelerate)
│  ├─────────────────────────┤  │
│  │ 6d. Table regions       │  │  Grid-line detection + cell OCR
│  │     → Markdown table    │  │  Metal + Vision
│  └─────────────────────────┘  │
└──────────┬────────────────────┘
           │  Structured content blocks
           ▼
┌──────────────────┐
│ Stage 7: Document │  Spatial merge + deduplication
│ Assembly          │  Markdown emitter → live file sync
└──────────────────┘
```

---

## 3. Stage-by-Stage Design

### 3.1 Stage 1 — Spatial Registration (GPU, ~5 ms)

**Goal:** Correct for camera angle, vibration, and minor board shifts so that every frame maps to a stable, fronto-parallel coordinate system.

**Method:**

- On the first frame (and periodically every ~10 s), detect the four corners of the whiteboard using a Canny edge detector followed by a Hough line intersection, or, if the board has a distinctive border, using `VNDetectRectanglesRequest`.
- Compute a perspective homography (3×3 matrix) mapping the detected quadrilateral to a canonical rectangle.
- Apply the warp via a Metal compute kernel (`MPSImageBilinearScale` or a custom kernel using `MTLComputeCommandEncoder`). The GPU processes this in well under 5 ms for a 1080p frame.
- Between re-detections, track inter-frame motion with sparse optical flow (`VNTrackOpticalFlowRequest`) and accumulate a delta homography. An exponential moving average smooths jitter.

**Why Metal, not CPU:** The perspective warp is a per-pixel operation on a 1920×1080 image (~2 M pixels). Metal's GPU shaders process this in a single dispatch; the CPU would need ~15 ms even with NEON SIMD.

---

### 3.2 Stage 2 — Person Segmentation (Neural Engine, ~12 ms)

**Goal:** Produce a per-pixel mask distinguishing people (and their arms, markers, erasers) from the whiteboard surface.

**Model:** Apple's built-in `VNGeneratePersonSegmentationRequest` (Vision framework).

**Configuration:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `qualityLevel` | `.balanced` | ~12 ms on ANE; `.fast` is ~3 ms but edge quality drops noticeably around arms/hands. `.accurate` is ~30 ms, which would consume too much of the latency budget. |
| `outputPixelFormat` | `kCVPixelFormatType_OneComponent8` | Single-channel 8-bit mask is sufficient for alpha compositing. |

**Why this model:** Apple's Vision person segmentation runs natively on the Neural Engine with zero conversion overhead. It is a stateful request, meaning it temporally smooths the mask across frames when used with `VNSequenceRequestHandler`, which is exactly what a streaming pipeline needs. No third-party model matches this integration depth on macOS.

**Fallback:** For edge cases where non-person foreground objects occlude the board (e.g., a projector cart rolled in front), a secondary lightweight semantic segmentation model (MobileNetV3-based, ~4 MB CoreML) can be trained on a small custom dataset to detect common classroom objects. This is only triggered when the person mask covers less than an expected minimum area but the frame-difference map shows large occlusions.

---

### 3.3 Stage 3 — Surface Reconstruction (GPU, ~3 ms)

**Goal:** Maintain a persistent, unobstructed composite image of the whiteboard surface, filling in regions currently hidden behind people.

**Method — Running Median Background Model:**

1. Maintain a buffer of the last N = 30 rectified frames (~1 second at 30 fps) in a circular `MTLBuffer`.
2. For each pixel, maintain a **per-pixel age counter** (time since last observed as "background") and a running median of the recent background-classified values.
3. Each frame, use the person-segmentation mask from Stage 2 to classify pixels:
   - Pixels **outside** the mask (background) update the running median model.
   - Pixels **inside** the mask (foreground/occluded) are **not** used to update the model.
4. The composite output image is constructed by: for each pixel, if the current frame pixel is background, use it directly; otherwise, use the stored median value.

This is implemented as a single Metal compute kernel that reads the mask, the current frame, and the model buffer, and writes the composite. At 1080p, this runs in ~3 ms on the M4 GPU.

**Why a running median and not a Gaussian Mixture Model:** GMMs are more robust to multi-modal backgrounds (e.g., a flickering screen), but the whiteboard surface is approximately unimodal. The running median is simpler, faster, and more resistant to outlier corruption (e.g., a hand briefly touching the same pixel in many consecutive frames).

---

### 3.4 Stage 4 — Change Detection (CPU, ~2 ms)

**Goal:** Identify which rectangular regions of the composite board image have changed since the last processing cycle, so only new or modified ink is sent downstream.

**Method:**

1. Convert the current composite and the previously-processed composite to grayscale (Accelerate `vImageConvert`).
2. Compute an absolute difference image.
3. Apply an adaptive threshold (Accelerate `vImageDilate` + `vImageThreshold`) to produce a binary change mask.
4. Run connected-component analysis on the binary mask to extract bounding rectangles of changed regions. Small regions below a minimum area threshold (e.g., < 400 px²) are suppressed as noise.
5. Each bounding rectangle is tagged with a **content hash** (a fast perceptual hash of the region's pixel data). Before dispatching downstream, the hash is compared against a hash table of already-processed regions. If a match is found (content already digitized), the region is skipped. This is the core mechanism for **redundancy control**.

**Why CPU and not GPU:** The differencing and thresholding operate on a single-channel downscaled image (960×540 is sufficient). Accelerate's SIMD routines saturate the M4's performance cores in ~2 ms, and the result stays in unified memory for immediate access by Stage 5 without a GPU→CPU copy.

---

### 3.5 Stage 5 — Layout Classification (Neural Engine, ~8 ms)

**Goal:** Classify each changed region as *text*, *geometric diagram*, or *table*.

**Model: YOLOv11-nano, fine-tuned on a whiteboard layout dataset.**

| Property | Value |
|----------|-------|
| Architecture | YOLOv11n (Ultralytics) |
| Input resolution | 640 × 640 |
| Classes | `text_block`, `diagram`, `table`, `equation` |
| Export format | CoreML (`.mlpackage`), FP16, Neural Engine target |
| Inference latency | ~6–8 ms on M4 ANE via CoreML |
| Model size | ~5.4 MB |

**Why YOLOv11n:** YOLO26 (the latest generation) offers NMS-free end-to-end detection, but the CoreML export path is still maturing. YOLOv11n has proven CoreML compatibility, achieves strong mAP on document layout benchmarks (DocLayNet), and its nano variant fits comfortably within the latency budget. Benchmarks show YOLOv11 models consistently outperform YOLOv8 on document layout analysis while being more parameter-efficient.

**Training data:** Fine-tune on a custom dataset of ~2,000 annotated whiteboard photos. Augmentations should include perspective distortion, varied lighting, marker colors, and partial erasure. Pre-training on DocLayNet provides a strong initialization for the text/table/diagram distinction.

**Alternative (if a single model is preferred):** The Ultralytics pipeline also supports segmentation heads (YOLOv11n-seg), which would provide pixel-level region masks rather than axis-aligned bounding boxes, at the cost of ~3 ms additional latency.

---

### 3.6 Stage 6 — Content Recognition (Neural Engine + CPU, ~40–80 ms)

This is the most computationally expensive stage and is internally parallelized: text, diagram, and table regions are dispatched to independent recognition sub-pipelines that run concurrently.

#### 6a. Printed-Style & Clear Handwriting OCR — Apple Vision

**API:** `VNRecognizeTextRequest` (Vision framework)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `recognitionLevel` | `.fast` | ~15 ms per region on ANE. Sufficient for clear handwriting and printed markers. |
| `usesLanguageCorrection` | `true` | Corrects common OCR errors using built-in language model. |
| `recognitionLanguages` | `["en"]` (configurable) | Set to match lecture language. |

Each text region detected in Stage 5 is cropped from the composite image and submitted as a separate `VNImageRequestHandler`. The Vision framework returns an array of `VNRecognizedTextObservation` objects, each with bounding boxes and confidence scores.

**Confidence gating:** If the top candidate's confidence for a line is below 0.65, the line is routed to the TrOCR fallback (6b) for a second opinion.

#### 6b. Difficult Handwriting OCR — TrOCR-small (Fallback)

**Model:** `microsoft/trocr-small-handwritten`, converted to CoreML via `coremltools`.

| Property | Value |
|----------|-------|
| Architecture | ViT encoder (BEiT-small) + GPT-2 decoder |
| Parameters | ~62 M |
| CoreML size | ~130 MB (FP16) |
| Inference | ~50–70 ms per text-line crop on ANE |

TrOCR is the current state-of-the-art open-source model for handwritten text recognition. The `small` variant balances accuracy and speed: the `base` and `large` variants are more accurate but would exceed the latency budget for a fallback path.

**Conversion to CoreML:**

```python
import coremltools as ct
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-handwritten")
# Trace encoder and decoder separately, export as MLProgram
# Use ct.convert(..., compute_units=ct.ComputeUnit.ALL) to enable ANE
```

**When to invoke:** Only for lines where Apple Vision's confidence is below the threshold. In practice, this fallback fires for ~10–20% of handwritten lines (e.g., highly cursive writing, unusual abbreviations, mathematical notation mixed with text).

#### 6c. Diagram Vectorization — Heuristic Pipeline (CPU)

For regions classified as `diagram`, the goal is to produce a structured description (SVG or Mermaid syntax) rather than rasterized text.

**Pipeline:**

1. **Edge detection:** Canny edges on the cropped region (Accelerate/vImage).
2. **Line detection:** Progressive probabilistic Hough transform to extract line segments.
3. **Shape fitting:** Group line segments into primitives — rectangles, circles (via Hough circles), arrows (line + triangle head), and free-form curves (Douglas-Peucker simplification).
4. **Topology extraction:** Build an adjacency graph of shapes and connecting lines/arrows.
5. **Output:** Emit as either Mermaid diagram syntax (if the topology matches a flowchart or tree pattern) or inline SVG in the Markdown output.

This is entirely CPU-bound using Accelerate and runs in ~10–20 ms for a typical diagram region. No ML model is required — geometric heuristics are more reliable for the structured, high-contrast shapes found on whiteboards than a general-purpose vision model.

#### 6d. Table Recognition — Grid Detection + Cell OCR

1. **Grid detection:** Apply Hough lines to the table region. Cluster horizontal and vertical lines to infer row/column boundaries. This runs as a Metal compute kernel (~2 ms).
2. **Cell extraction:** Crop each cell from the composite image.
3. **Cell OCR:** Submit each cell to `VNRecognizeTextRequest` (same as 6a). Cells are batched and processed in parallel.
4. **Output:** Assemble into a Markdown table (`| col1 | col2 |` syntax).

---

### 3.7 Stage 7 — Document Assembly (CPU, ~2 ms)

**Goal:** Merge recognized content into a single, continuously-updated Markdown document that reflects the spatial layout of the whiteboard.

**Method:**

1. **Spatial grid mapping:** Divide the board coordinate space into a logical grid (e.g., 3 columns × N rows, where row boundaries are inferred from vertical spacing between content blocks).
2. **Block placement:** Each recognized content block (text paragraph, diagram, table) is assigned a grid cell based on its centroid in board coordinates.
3. **Deduplication:** Before inserting a new block, compare its content hash and spatial position against existing blocks. If a block with >85% text similarity (Levenshtein ratio) exists within a 50-pixel radius, treat it as an update rather than an insertion.
4. **Markdown emission:** The grid is serialized top-to-bottom, left-to-right into Markdown. Headings are inferred from text size (larger strokes → `##`), diagrams are embedded as code blocks (Mermaid) or inline SVG, and tables use standard Markdown table syntax.
5. **File sync:** The Markdown file is written atomically (`FileManager.replaceItemAt`) and a `DistributedNotificationCenter` notification is posted so that any listening UI (e.g., a SwiftUI preview pane) can refresh.

---

## 4. Concurrency & Scheduling Model

### 4.1 Threading Architecture

```
┌─────────────────────────────────────────────────────┐
│                   GCD / Swift Concurrency            │
├──────────┬──────────┬──────────┬────────────────────┤
│ Camera Q │ Vision Q │ Metal Q  │ Assembly Q         │
│ (serial) │ (serial) │ (serial) │ (serial)           │
│          │          │          │                    │
│ Stage 1  │ Stage 2  │ Stage 3  │ Stage 7            │
│          │ Stage 5  │          │                    │
│          │ Stage 6a │          │                    │
│          │ Stage 6b │          │                    │
├──────────┴──────────┴──────────┴────────────────────┤
│ Accelerate / vImage  (CPU, any core)                │
│ Stage 4 · Stage 6c · Stage 6d (Hough)              │
└─────────────────────────────────────────────────────┘
```

- **Camera Queue:** Receives `CMSampleBuffer` from `AVCaptureVideoDataOutput` delegate. Extracts `CVPixelBuffer`, submits to Stage 1, and immediately returns. If Stage 1 is still processing the previous frame, the new frame is dropped (back-pressure via atomic flag).
- **Vision Queue:** Handles all `VNRequest` submissions. Vision framework internally dispatches to the Neural Engine; the queue serializes request creation.
- **Metal Queue:** Owns the `MTLCommandQueue`. Stages 1 and 3 enqueue Metal command buffers here.
- **Assembly Queue:** Serializes writes to the Markdown document and deduplication state.

### 4.2 Frame Pacing & Back-Pressure

The camera delivers 30 fps, but the full pipeline only needs to complete a cycle every 100–200 ms (5–10 Hz). The strategy:

- Process every 3rd frame (10 Hz effective rate) under normal conditions.
- If the change detector (Stage 4) reports no meaningful changes, skip all downstream stages entirely (idle power savings).
- If the pipeline falls behind (Stage 6 taking >200 ms due to many regions), automatically drop to every 6th frame (5 Hz) until the backlog clears.

---

## 5. Memory Budget

| Component | Allocation | Notes |
|-----------|-----------|-------|
| Frame ring buffer (30 frames, 1080p, BGRA) | ~240 MB | Stored in `MTLBuffer` (unified memory) |
| Background model (running median) | ~8 MB | Single-channel float per pixel |
| YOLOv11n CoreML model | ~5.4 MB | Loaded once at startup |
| TrOCR-small CoreML model | ~130 MB | Loaded lazily on first fallback |
| Vision framework models (person seg, OCR) | ~50 MB | Managed by OS, shared with other apps |
| Working buffers (masks, diffs, crops) | ~100 MB | Peak allocation during Stage 6 |
| Application code + state | ~50 MB | |
| **Total peak** | **~584 MB** | Well within the 24 GB budget |

The system uses less than 3% of available unified memory at peak, leaving ample headroom for the OS and other applications. The largest single allocation (frame ring buffer) can be reduced to ~80 MB by storing frames at half resolution (960×540) for the background model and only warping to full resolution for the current frame's recognition pass.

---

## 6. Model Summary & Conversion Matrix

| Stage | Model / Engine | Source | Target Runtime | Conversion Path | Est. Latency |
|-------|---------------|--------|---------------|----------------|-------------|
| 1 | Custom Metal kernel | — | GPU (Metal) | Native | ~5 ms |
| 2 | Apple Person Segmentation | Vision.framework | Neural Engine | Built-in | ~12 ms |
| 3 | Custom Metal kernel | — | GPU (Metal) | Native | ~3 ms |
| 4 | Accelerate / vImage | — | CPU (SIMD) | Native | ~2 ms |
| 5 | YOLOv11n | Ultralytics PyTorch | Neural Engine (CoreML) | `coremltools` + `ultralytics export format=coreml` | ~8 ms |
| 6a | Apple Text Recognition | Vision.framework | Neural Engine | Built-in | ~15 ms / region |
| 6b | TrOCR-small-handwritten | HuggingFace PyTorch | Neural Engine (CoreML) | `coremltools.convert()` from traced PyTorch | ~60 ms / line |
| 6c | Heuristic (Canny + Hough) | — | CPU (Accelerate) | Native | ~15 ms / region |
| 6d | Hough lines + Vision OCR | — | GPU + Neural Engine | Native | ~20 ms / table |
| 7 | Document assembly logic | — | CPU | Native Swift | ~2 ms |

---

## 7. Software Stack & Dependencies

### 7.1 Apple Frameworks (No Third-Party Dependencies)

| Framework | Usage |
|-----------|-------|
| **AVFoundation** | Camera capture (`AVCaptureSession`, `AVCaptureVideoDataOutput`) |
| **Vision** | Person segmentation, text recognition, rectangle detection, optical flow |
| **Metal / MetalPerformanceShaders** | Perspective warp, background model update, grid-line detection |
| **Accelerate / vImage** | Frame differencing, thresholding, connected components, Hough transform |
| **CoreML** | YOLOv11n inference, TrOCR inference |
| **CoreImage** | Alpha compositing (mask blending), color space conversions |
| **Foundation** | File I/O, `DistributedNotificationCenter`, atomic writes |

### 7.2 Build-Time Dependencies

| Tool | Purpose |
|------|---------|
| `coremltools` (Python) | Convert PyTorch models to `.mlpackage` |
| `ultralytics` (Python) | Train and export YOLOv11n |
| `transformers` (Python) | Load TrOCR weights for conversion |

### 7.3 Recommended Language & Runtime

- **Swift 6.x** with structured concurrency (`async/await`, `TaskGroup`) for the pipeline orchestration.
- **Metal Shading Language** for GPU compute kernels.
- **Python** only at build time for model conversion, not at runtime.

---

## 8. UI Synchronization

The background service writes the Markdown file to a known path and signals the UI via:

1. **`DistributedNotificationCenter`** — lightweight inter-process notification for a menubar/companion app.
2. **Combine `PassthroughSubject`** — if the UI is in-process (e.g., a SwiftUI sidebar), a Combine publisher emits on every document update, debounced to 200 ms.
3. **File system events (`DispatchSource.makeFileSystemObjectSource`)** — a file watcher on the output `.md` file allows any external editor (Obsidian, VS Code) to live-reload.

A companion SwiftUI view renders the Markdown in real-time using `AttributedString` or a lightweight Markdown renderer (e.g., `swift-markdown` + `NSTextView`).

---

## 9. Latency Budget Breakdown

| Stage | Target | Runs On | Pipelined? |
|-------|--------|---------|-----------|
| 1. Spatial Registration | 5 ms | GPU | Yes — overlaps with Stage 2 of previous frame |
| 2. Person Segmentation | 12 ms | ANE | Yes — overlaps with Stage 3 of previous frame |
| 3. Surface Reconstruction | 3 ms | GPU | Yes |
| 4. Change Detection | 2 ms | CPU | Sequential gate (determines if rest runs) |
| 5. Layout Classification | 8 ms | ANE | Sequential after 4 |
| 6. Recognition (parallel) | 40–80 ms | ANE + CPU | Internal parallelism across regions |
| 7. Document Assembly | 2 ms | CPU | Sequential after 6 |
| **Total (sequential worst case)** | **~112 ms** | | |
| **Total (pipelined steady state)** | **~90 ms** | | Within 100–200 ms target |

Stages 1–3 are pipelined: while Stage 2 processes frame N, Stage 1 is already warping frame N+1, and Stage 3 is compositing frame N−1. The effective throughput bottleneck is Stage 6 (recognition), which runs at ~40–80 ms depending on how many regions changed. With the 10 Hz processing cadence, there is a comfortable 100 ms budget per cycle, and the pipeline meets it even at the upper bound.

---

## 10. Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Person segmentation misses arms/hands holding markers, leaking foreground into background model | Use `.balanced` quality level which captures limbs more accurately; additionally, mask pixels that are within 50 px of any detected person edge are excluded from background updates. |
| TrOCR CoreML conversion introduces accuracy regression | Validate converted model against the IAM test set; accept ≤1% character error rate increase vs. PyTorch baseline. Use FP16 (not INT8) to preserve accuracy. |
| Camera vibration causes registration drift over long sessions | Periodic re-detection of board corners every 10 s; if corner detection fails (board partially occluded), hold the last good homography and rely on optical flow deltas only. |
| Marker colors too close to whiteboard background | Operate in HSV color space for change detection; whiteboard surface is high-value, low-saturation, while markers are high-saturation. Threshold on saturation channel rather than luminance. |
| Concurrent ANE requests (person seg + YOLO + OCR) contend for Neural Engine | Serialize ANE-bound requests on a single Vision queue. The pipeline design ensures at most one ANE model runs at a time per frame cycle. |

---

## 11. Future Enhancements

- **On-device VLM post-correction:** As smaller vision-language models (e.g., Florence-2 at ~230 M parameters, already available as CoreML) become faster, a final post-processing pass could use a VLM to verify and correct OCR output with full visual context, reducing character error rates to near-human levels.
- **MLX-native inference:** For stages currently using CoreML, migrating to MLX (Apple's open-source ML framework) would enable GPU-based inference with zero-copy unified memory access and avoid CoreML's 2–4× dispatch overhead on small operations. MLX already supports YOLO26 natively on Apple Silicon with up to 2× faster inference than PyTorch MPS.
- **Incremental handwriting recognition:** Rather than processing completed text lines, track pen strokes in real-time and feed partial stroke sequences to a character-level RNN, producing text predictions before the writer lifts the marker.
- **Multi-board stitching:** Support panoramic capture of multiple whiteboards or a single board wider than the camera's field of view, using feature-based stitching across overlapping frames.
