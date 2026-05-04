# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

The planned implementation is a **Swift 6.x + Metal macOS application** targeting Apple M4 silicon. It processes live camera input through a 7-stage pipeline targeting 100–200 ms end-to-end latency.

### Pipeline Stages

| Stage | What it does | Hardware | Latency |
|-------|-------------|----------|---------|
| 1. Spatial Registration | Perspective-corrects the camera frame to a flat board view | GPU (Metal) | ~5 ms |
| 2. Person Segmentation | Masks out people/arms using `VNGeneratePersonSegmentationRequest` | Neural Engine | ~12 ms |
| 3. Surface Reconstruction | Running median background model to "fill in" occluded pixels | GPU (Metal) | ~3 ms |
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

## Model Conversion (Python Build Step)

Two models require conversion from PyTorch → CoreML before the Swift app can use them:

**YOLOv11n** (layout classification, ~5.4 MB):
```bash
yolo export model=yolov11n.pt format=coreml
```

**TrOCR-small** (handwriting fallback, ~130 MB FP16):
```python
import coremltools as ct
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-handwritten")
# Trace encoder and decoder separately, export as MLProgram
# ct.convert(..., compute_units=ct.ComputeUnit.ALL)
```

Use FP16 (not INT8) for TrOCR to stay within ≤1% CER regression vs. the PyTorch baseline.

## Output Format

The pipeline writes a single continuously-updated Markdown file. UI synchronization uses `DistributedNotificationCenter` (inter-process), Combine `PassthroughSubject` (in-process SwiftUI), or `DispatchSource` file system events (external editors like Obsidian/VS Code).

## Architecture Reference

Full design details, rationale, latency budgets, and risk mitigations are in [docs/architecture.md](docs/architecture.md).
