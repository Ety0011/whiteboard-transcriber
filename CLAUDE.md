# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## What

Real-time whiteboard-to-Markdown transcription system. Captures a camera feed, removes people, reconstructs the board surface, detects layout regions, identifies which regions changed, runs OCR on changed regions only, and emits a continuously-updated Markdown document.

University project (1-month timeline). Prioritizes **accuracy over speed** — a 1–2 second processing cycle is fine because whiteboard content changes every 5–10 seconds. Cross-platform. All processing on-device.

## Tech Stack

- **Language:** Python 3.11+
- **Computer vision:** OpenCV (`cv2`)
- **Person segmentation:** MediaPipe Selfie Segmentation
- **Layout detection:** PaddleOCR PP-DocLayout (via `paddleocr`)
- **OCR:** PaddleOCR PP-OCRv5 (via `paddleocr`)
- **Change detection:** DINOv2-base (ViT-B/14, 86M params) — per-region embedding similarity
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
│   ├── layout.py              # Stage 4: PP-DocLayout region detection
│   ├── change_detection.py    # Stage 5: DINOv2 embedding comparison
│   ├── recognition.py         # Stage 6: PP-OCRv5 + Markdown output on changed regions
│   ├── pipeline.py            # Orchestrates stages 1–6, merges Markdown fragments to file
│   └── utils.py               # Config, logging, helpers
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

6-stage pipeline. All stages run sequentially in a processing thread. Camera runs in a separate daemon thread with a `Queue(maxsize=1)` for back-pressure (latest frame only).

| Stage | What | Library |
|-------|------|---------|
| 1. Registration | Perspective warp to flat board | OpenCV |
| 2. Segmentation | Person mask | MediaPipe |
| 3. Background | Clean board composite via MOG2 | OpenCV |
| 4. Layout | Region detection (text/table/diagram/formula) | PaddleOCR (PP-DocLayout) |
| 5. Change Detection | DINOv2 embedding per region, cosine similarity gate | PyTorch (DINOv2-base) |
| 6. Recognition | OCR + Markdown on changed regions only | PaddleOCR (PP-StructureV3) |

PP-StructureV3 (stage 6) outputs Markdown directly. `pipeline.py` merges per-region fragments and writes the final file — no separate assembly stage needed.

### Change Detection Design (Stage 5)

The change detection gate sits **after** layout detection, not before. This enables per-region change tracking instead of whole-frame diffing.

**How it works:**
1. Stage 4 produces a list of detected regions (bounding boxes + class labels).
2. Each region is cropped and passed through DINOv2-base to produce a 768-dim embedding vector.
3. Each embedding is compared (cosine similarity) against the stored embedding for the spatially nearest region from the previous cycle.
4. If `cosine_similarity > threshold` (start with 0.92, tune empirically), the region is considered unchanged — skip OCR.
5. Changed regions proceed to Stage 6. Their new embeddings replace the stored ones.

**Region matching across frames:**
Regions are matched by IoU (intersection over union) of bounding boxes between consecutive frames, not by index order. A new region with no IoU match to any previous region is always treated as changed.

**Why DINOv2 over pixel hashing:**
- Robust to lighting changes, camera micro-movements, shadows
- Semantic: minor pixel noise doesn't trigger false changes
- Erasing and rewriting genuinely shifts the embedding
- DINOv2-base runs ~20–40ms per crop on CPU for small region crops

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

- **PaddleOCR is slow to initialize** (~3–5 seconds loading models). Create the `PaddleOCR` instance once at startup, not per frame.
- **DINOv2 model loading takes a few seconds.** Load once at startup. Use `torch.no_grad()` for all inference — you never train.
- **MediaPipe expects RGB input.** Always `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` before calling `segmenter.process()`. Forgetting this produces garbage masks silently.
- **DINOv2 expects RGB input, normalized.** Convert from BGR, resize to 224×224 (or multiple of 14), normalize with ImageNet mean/std. Don't feed it raw OpenCV arrays.
- **MOG2 background model needs person masking.** Feed the subtractor a frame where person pixels are replaced with white/board color, otherwise people standing still corrupt the background model.
- **`Queue(maxsize=1)` drops frames intentionally.** Use `put_nowait()` with a try/except or `get()` the old frame first. This is correct behavior, not a bug.
- **Model files are large.** Do not commit them to git. Add `models/` to `.gitignore`. PaddleOCR and DINOv2 download weights automatically on first run.
- **PaddlePaddle is a separate ML framework from PyTorch.** Both will be installed. PaddleOCR uses PaddlePaddle; DINOv2 uses PyTorch. This is intentional — they don't conflict but the install is large.

## Architecture Reference

Full design spec, latency budgets, model details, and risk mitigations: `docs/architecture.md`
