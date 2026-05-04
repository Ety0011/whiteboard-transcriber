# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Status

This repository is in **design phase**. The implementation does not exist yet — only the architecture specification (`docs/architecture.md`) and Python build-time tooling.

## Development Environment

The project uses a Nix flake for reproducible environments. With `direnv` installed:

```bash
direnv allow          # loads the Nix flake automatically
pip install -r requirements.txt  # Python build-time tools (model conversion only)
```

Without `direnv`:

```bash
nix develop           # enter the Nix development shell manually
```

Python 3.13 is the configured version. Python is used **only at build time** for converting PyTorch models to CoreML — not at runtime.

## Architecture Overview

The planned implementation is a **Swift 6.x + Metal macOS application** targeting Apple M4 silicon (24 GB unified memory, 16-core Neural Engine). It processes live camera input through a 7-stage pipeline targeting 100–200 ms end-to-end latency.

### Pipeline Stages

| Stage | What it does | Hardware | Latency |
|-------|-------------|----------|---------|
| 1. Spatial Registration | Perspective-corrects the camera frame to a flat board view | GPU (Metal) | ~5 ms |
| 2. Person Segmentation | Masks out people/arms using `VNGeneratePersonSegmentationRequest` | Neural Engine | ~12 ms |
| 3. Surface Reconstruction | Running median background model to fill in occluded pixels | GPU (Metal) | ~3 ms |
| 4. Change Detection | Frame differencing + hashing to find only new/modified ink | CPU (Accelerate) | ~2 ms |
| 5. Layout Classification | YOLOv11n CoreML model classifies regions as text/diagram/table | Neural Engine | ~8 ms |
| 6. Recognition (parallel) | Per-region content extraction (see below) | ANE + CPU | ~40–80 ms |
| 7. Document Assembly | Spatial grid merge → atomic Markdown file write | CPU | ~2 ms |

Stages 1–3 are **pipelined** (frame N+1 enters Stage 1 while frame N is in Stage 2). Stage 4 is a gate: if no regions changed, stages 5–7 are skipped entirely.

### Recognition Sub-pipeline (Stage 6)

- **6a Text (clear):** `VNRecognizeTextRequest` with `.fast` recognition level
- **6b Handwriting fallback:** TrOCR-small CoreML model, triggered when Vision confidence < 0.65
- **6c Diagrams:** Canny + Hough heuristics → Mermaid/SVG output (no ML model)
- **6d Tables:** Hough grid detection + batched cell OCR

### Concurrency Model

Four GCD serial queues: Camera, Vision, Metal, Assembly. The Vision queue serializes all ANE requests to avoid Neural Engine contention. Back-pressure is handled by dropping frames, not blocking.

### Key Apple Frameworks

`AVFoundation` (capture) · `Vision` (segmentation, OCR, optical flow) · `Metal` / `MetalPerformanceShaders` (GPU compute) · `Accelerate` / `vImage` (CPU SIMD) · `CoreML` (YOLO, TrOCR) · `CoreImage` · `Foundation`

## Project Structure

```
WhiteboardTranscriber/
├── Sources/
│   ├── App/                # SwiftUI app entry point, settings, menubar
│   ├── Pipeline/           # PipelineOrchestrator, frame pacing, back-pressure
│   ├── Capture/            # AVFoundation camera session + configuration
│   ├── Registration/       # Stage 1: corner detection, homography, warp
│   ├── Segmentation/       # Stage 2: person mask via Vision framework
│   ├── Background/         # Stage 3: running median model (Metal kernel)
│   ├── ChangeDetection/    # Stage 4: diff, threshold, connected components, hash dedup
│   ├── Layout/             # Stage 5: YOLOv11n region classifier
│   ├── Recognition/        # Stage 6: OCR, TrOCR fallback, diagram vectorizer, tables
│   ├── Assembly/           # Stage 7: spatial grid, Markdown emitter, file sync
│   └── Shared/             # Extensions, Metal utilities, logging, common types
├── Shaders/                # .metal compute kernels (warp, background, grid detection)
├── Models/                 # .mlpackage CoreML models (git-lfs or .gitignore'd)
├── Scripts/                # Python model conversion scripts
│   ├── convert_yolo.py
│   └── convert_trocr.py
├── Tests/                  # Per-stage unit tests with fixture images
├── docs/
│   └── architecture.md     # Full design spec, rationale, latency budgets
├── Package.swift
├── flake.nix
└── CLAUDE.md
```

## Key Commands

```bash
swift build                         # compile the package
swift test                          # run all unit tests
swift test --filter RegistrationTests  # run a single test suite
python Scripts/convert_yolo.py      # export YOLOv11n → Models/yolo11n_layout.mlpackage
python Scripts/convert_trocr.py     # export TrOCR-small → Models/trocr_small.mlpackage
```

## Coding Rules

- Every pipeline stage conforms to the `PipelineStage` protocol
- Use `async/await` and `TaskGroup` for concurrency — no raw GCD unless interfacing directly with AVFoundation or Metal command queues
- Metal kernels live in `Shaders/`; their Swift wrappers live in the matching `Sources/` module
- All `MTLBuffer` allocations use `.storageModeShared` (unified memory — no CPU↔GPU copies)
- Use `VNSequenceRequestHandler` (not `VNImageRequestHandler`) for streaming Vision requests — it enables temporal smoothing
- Each stage must be testable in isolation using fixture images — no integration-only code
- Keep functions short and single-purpose; prefer composition over inheritance
- Commit messages: conventional commits (`feat:`, `fix:`, `refactor:`, `perf:`, `test:`, `docs:`)
- Always work on a feature branch — never commit directly to `main`

## Model Conversion (Python Build Step)

Two models require conversion from PyTorch → CoreML before the Swift app can use them:

**YOLOv11n** (layout classification, ~5.4 MB):
```bash
python Scripts/convert_yolo.py
# internally: yolo export model=yolov11n.pt format=coreml imgsz=640 half=True nms=False
```

**TrOCR-small** (handwriting fallback, ~130 MB FP16):
```bash
python Scripts/convert_trocr.py
# internally: trace encoder/decoder → ct.convert(..., compute_units=ct.ComputeUnit.ALL)
```

Use FP16 (not INT8) for TrOCR to stay within ≤1% CER regression vs. the PyTorch baseline.

## Output Format

The pipeline writes a single continuously-updated Markdown file. UI synchronization uses `DistributedNotificationCenter` (inter-process), Combine `PassthroughSubject` (in-process SwiftUI), or `DispatchSource` file system events (external editors like Obsidian/VS Code).

## Warnings

- **ANE serialization:** At most one Neural Engine model may run at a time. The Vision queue enforces this — never dispatch CoreML predictions from other queues.
- **AVCaptureSession threading:** `startRunning()` and `stopRunning()` must be called on a background thread, never the main thread.
- **Metal command buffers:** Always call `.commit()` after `.endEncoding()` — an encoded-but-uncommitted buffer silently does nothing.
- **TrOCR is a fallback only:** Do not route all text through it. At ~60 ms per line it will blow the latency budget. Use Apple Vision OCR as the primary path.
- **Person segmentation quality:** Use `.balanced`, not `.fast` — the latter drops arm/hand edges which corrupts the background model.
- **Frame drops are expected:** The back-pressure system intentionally drops frames. Never queue unbounded work — check the atomic processing flag before submitting.

## Architecture Reference

Full design details, model rationale, memory budget, latency breakdown, and risk mitigations are in `docs/architecture.md`.
