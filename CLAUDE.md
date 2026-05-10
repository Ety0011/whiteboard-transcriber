# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Status

This repository is **pivoting from a Swift/Metal design to a Python implementation**. The original Swift scaffold has been abandoned. The architecture specification (`docs/architecture.md`) describes the Python pipeline. Ignore any Swift, Metal, or CoreML references in the repo — they are legacy artifacts.

## What

Real-time whiteboard-to-Markdown transcription system. Captures a camera feed, removes people, reconstructs the board surface, detects changes, classifies layout regions, runs OCR, and emits a continuously-updated Markdown document.

## Why

University project (1-month timeline). Prioritizes **accuracy over speed** — a 1–2 second processing cycle is fine because whiteboard content changes every 5–10 seconds. Cross-platform (not Apple-specific). All processing on-device.

## Tech Stack

- **Language:** Python 3.11+
- **Computer vision:** OpenCV (`cv2`)
- **Person segmentation:** MediaPipe Selfie Segmentation
- **Text region detection:** CRAFT (`craft-text-detector`)
- **Change detection:** DINOv2-small (`facebook/dinov2-small`, HuggingFace Transformers)
- **OCR:** TrOCR-small-handwritten (`microsoft/trocr-small-handwritten`, HuggingFace Transformers)
- **Equation OCR:** pix2tex (stubbed — `NotImplementedError`)
- **Similarity:** scikit-learn `cosine_similarity`
- **Concurrency:** `threading` + `queue.Queue`
- **Testing:** pytest
- **Package management:** pip + requirements.txt

## Development Environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Project Structure

```
whiteboard-transcriber/
├── src/
│   ├── main.py                # Entry point, thread orchestration
│   ├── capture.py             # Camera thread, frame queue
│   ├── registration.py        # Stage 1: perspective correction
│   ├── segmentation.py        # Stage 2: person mask (MediaPipe)
│   ├── background.py          # Stage 3: MOG2 surface reconstruction
│   ├── region_detection.py    # Stage 4: CRAFT text regions + spatial IDs
│   ├── change_detection.py    # Stage 5: DINOv2 embeddings, state machine
│   ├── recognition.py         # Stage 6: TrOCR (primary), pix2tex stub
│   ├── assembly.py            # Stage 7: Markdown emitter, MUTATION/CLEARANCE
│   ├── pipeline.py            # Orchestrates stages 1–7
│   └── utils.py               # Config, logging, hash table
├── models/                    # Downloaded model weights (.gitignore'd)
├── tests/
│   ├── fixtures/              # Test images of whiteboards
│   └── test_*.py              # One test file per stage
├── docs/
│   └── architecture.md
├── output/                    # Generated Markdown files
├── requirements.txt
├── CLAUDE.md
└── README.md
```

## Key Commands

```bash
python -m pytest                              # run all tests
python -m pytest tests/test_region_detection.py  # single test file
python src/main.py                            # run the pipeline (requires webcam)
python src/main.py --input video.mp4          # run on a video file (for testing)
```

## Architecture Overview

7-stage pipeline, same logical design as the architecture doc. All stages run sequentially in a processing thread. Camera runs in a separate daemon thread with a `Queue(maxsize=1)` for back-pressure (latest frame only).

| Stage | What | Library |
|-------|------|---------|
| 1. Registration | Perspective warp to flat board | OpenCV |
| 2. Segmentation | Person mask | MediaPipe |
| 3. Background | Clean board composite via MOG2 | OpenCV |
| 4. Region Detection | CRAFT text bounding boxes + spatial IDs | craft-text-detector |
| 5. Change Detection | DINOv2 CLS cosine similarity, state machine (IDENTICAL / MUTATION / CLEARANCE) | transformers, scikit-learn |
| 6. Recognition | TrOCR (primary text), pix2tex stub (equations) | transformers, pix2tex |
| 7. Assembly | Spatial Markdown, update/clearance tracking, atomic write | Python stdlib |

Stage 5 is the gate: only MUTATION regions pass to stages 6–7.

## Coding Rules

- Each stage is a module in `src/` with a clear `process()` function
- Functions take NumPy arrays (images) as input and return structured results — no global mutable state
- Use type hints on all function signatures
- Keep each module independently testable with fixture images — no camera required for tests
- Use `logging` module, not `print()`, for debug output
- Write docstrings on every public function
- Commit messages: conventional commits (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`)
- Always work on a feature branch — never commit directly to `main`
- One module per file — don't combine stages

## Coding Style

- Follow PEP 8
- Max line length: 100 characters
- Imports: stdlib → third-party → local, separated by blank lines
- Use `pathlib.Path` for file paths, not string concatenation
- NumPy arrays are `np.ndarray`, images are BGR uint8 (OpenCV convention) unless documented otherwise

## Warnings

- **DINOv2 first load is slow** (~3–5 s, downloads ~80 MB on first run). Initialize once at startup.
- **CRAFT outputs character-level heatmaps.** Use `craft_text_detector.get_prediction` then `craft_text_detector.get_boxes` — do not call the raw model forward pass directly.
- **TrOCR is now the primary OCR** (not a fallback). At ~200–400 ms per crop on CPU, batch regions where possible. Load `TrOCRProcessor` + `VisionEncoderDecoderModel` once at startup.
- **pix2tex is stubbed.** Any `RegionType.EQUATION` call raises `NotImplementedError`. Do not silently route equations to TrOCR.
- **DINOv2 cosine similarity thresholds (0.95 / 0.2):** tune with fixture images before hardcoding.
- **MediaPipe expects RGB input.** Always `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` before calling `segmenter.process()`. Forgetting this produces garbage masks silently.
- **MOG2 background model needs person masking.** Feed the subtractor a frame where person pixels are replaced with white/board color, otherwise people standing still corrupt the background model.
- **`Queue(maxsize=1)` drops frames intentionally.** Use `put_nowait()` with a try/except or `get()` the old frame first. This is correct behavior, not a bug.
- **Model files are large.** Do not commit them to git. Add `models/` to `.gitignore`. EasyOCR and TrOCR download weights automatically on first run.

## Architecture Reference

Full design spec, latency budgets, model details, and risk mitigations: `docs/architecture.md`
