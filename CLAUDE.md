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
- **Layout detection:** DocLayout-YOLO or YOLOv11n (Ultralytics)
- **OCR (primary):** EasyOCR
- **OCR (fallback):** TrOCR-small-handwritten (HuggingFace Transformers)
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
│   ├── change_detection.py    # Stage 4: diff, threshold, dedup
│   ├── layout.py              # Stage 5: YOLO layout classification
│   ├── recognition.py         # Stage 6: OCR, diagrams, tables
│   ├── assembly.py            # Stage 7: Markdown emitter
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
python -m pytest tests/test_change_detection.py  # single test file
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
| 4. Change Detection | Diff + dedup hash | OpenCV + NumPy |
| 5. Layout | Region classification (text/diagram/table) | Ultralytics YOLO |
| 6. Recognition | OCR + diagram vectorization + table extraction | EasyOCR, TrOCR, OpenCV |
| 7. Assembly | Markdown output with spatial layout | Python stdlib |

Stage 4 is a gate: if nothing changed, stages 5–7 are skipped.

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

- **EasyOCR is slow to initialize** (~3–5 seconds loading models). Create the `Reader` once at startup, not per frame.
- **TrOCR is a fallback only.** At ~200–400 ms per line on CPU, routing all text through it will make the pipeline unusably slow. Only invoke for EasyOCR lines with confidence < 0.65.
- **MediaPipe expects RGB input.** Always `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` before calling `segmenter.process()`. Forgetting this produces garbage masks silently.
- **MOG2 background model needs person masking.** Feed the subtractor a frame where person pixels are replaced with white/board color, otherwise people standing still corrupt the background model.
- **`Queue(maxsize=1)` drops frames intentionally.** Use `put_nowait()` with a try/except or `get()` the old frame first. This is correct behavior, not a bug.
- **Model files are large.** Do not commit them to git. Add `models/` to `.gitignore`. EasyOCR and TrOCR download weights automatically on first run.

## Architecture Reference

Full design spec, latency budgets, model details, and risk mitigations: `docs/architecture.md`
