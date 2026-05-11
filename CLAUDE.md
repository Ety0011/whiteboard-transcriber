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
│   ├── text_detection.py      # Stage 5: PP-OCRv5 text line detection within regions
│   ├── change_detection.py    # Stage 6: DINOv2 embedding comparison per text line
│   ├── recognition.py         # Stage 7: PP-OCRv5 recognition on changed lines only
│   ├── pipeline.py            # Orchestrates stages 1–7, merges Markdown fragments to file
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

7-stage pipeline. All stages run sequentially in a processing thread. Camera runs in a separate daemon thread with a `Queue(maxsize=1)` for back-pressure (latest frame only).

| Stage | What | Library |
|-------|------|---------|
| 1. Registration | Perspective warp to flat board | OpenCV |
| 2. Segmentation | Person mask | MediaPipe |
| 3. Background | Clean board composite via MOG2 | OpenCV |
| 4. Layout | Region detection (text/table/figure/formula) | PaddleOCR (`RT-DETR-H_layout_17cls`) |
| 5. Text Detection | Text line bounding boxes within each region | PaddleOCR (`PP-OCRv5_det_server`) |
| 6. Change Detection | DINOv2 embedding per text line, cosine similarity gate | PyTorch (DINOv2-base) |
| 7. Recognition | OCR on changed lines only, emit Markdown | PaddleOCR (PP-OCRv5 recognition) |

`pipeline.py` merges per-line Markdown fragments and writes the final file.

### Change Detection Design (Stage 6)

The change detection gate operates at **text line level**, not region level. This means a growing region (professor adds a new line) only re-OCRs the new/changed lines — existing lines are skipped.

**How it works:**
1. Stage 5 produces text line bounding boxes within each region.
2. Each line crop is passed through DINOv2-base to produce a 768-dim embedding vector.
3. Each embedding is compared (cosine similarity) against the stored embedding for the matching line from the previous cycle.
4. Lines are matched across frames by IoU of bounding boxes. A line with no IoU match is always treated as new.
5. If `cosine_similarity > threshold` (start with 0.92, tune empirically), the line is unchanged — skip OCR.
6. Changed or new lines proceed to Stage 7. Their embeddings replace the stored ones.

**Why line-level, not region-level:**
A region grows as the professor writes. Region-level diffing would re-OCR the entire region every cycle. Line-level diffing only re-OCRs the new line at the bottom — existing lines are cached.

**Why DINOv2 over pixel hashing:**
- Robust to lighting changes, camera micro-movements, shadows
- Semantic: minor pixel noise doesn't trigger false changes
- Erasing and rewriting genuinely shifts the embedding

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

- **PP-OCRv5 detection returns polygons**, not rectangles. Convert to axis-aligned bboxes using min/max of the 4 point coordinates before storing or cropping.
- **PaddleOCR is slow to initialize** (~3–5 seconds loading models). Create all PaddleOCR instances once at startup, not per frame.
- **DINOv2 model loading takes a few seconds.** Load once at startup. Use `torch.no_grad()` for all inference — you never train.
- **MediaPipe expects RGB input.** Always `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` before calling `segmenter.process()`. Forgetting this produces garbage masks silently.
- **DINOv2 expects RGB input, normalized.** Convert from BGR, resize to 224×224 (or multiple of 14), normalize with ImageNet mean/std. Don't feed it raw OpenCV arrays.
- **MOG2 background model needs person masking.** Feed the subtractor a frame where person pixels are replaced with white/board color, otherwise people standing still corrupt the background model.
- **`Queue(maxsize=1)` drops frames intentionally.** Use `put_nowait()` with a try/except or `get()` the old frame first. This is correct behavior, not a bug.
- **Model files are large.** Do not commit them to git. Add `models/` to `.gitignore`. PaddleOCR and DINOv2 download weights automatically on first run.
- **PaddlePaddle is a separate ML framework from PyTorch.** Both will be installed. PaddleOCR uses PaddlePaddle; DINOv2 uses PyTorch. This is intentional — they don't conflict but the install is large.

## Architecture Reference

Full design spec, latency budgets, model details, and risk mitigations: `docs/architecture.md`
