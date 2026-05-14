# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## What

Real-time whiteboard-to-Markdown transcription system. Captures a camera feed, removes people, reconstructs the board surface, tracks persistent text regions over time, runs OCR only when regions stabilize, and emits a continuously-updated Markdown document.

University project (1-month timeline). Prioritizes **accuracy over speed** — a 1–2 second processing cycle is fine because whiteboard content changes every 5–10 seconds. Cross-platform. All processing on-device.

## Tech Stack

- **Language:** Python 3.11+
- **Computer vision:** OpenCV (`cv2`)
- **Person segmentation:** MediaPipe Selfie Segmentation
- **Text detection:** PaddleOCR `PP-OCRv5_det_server` (via `paddleocr`)
- **Layout classification:** PaddleOCR `RT-DETR-H_layout_17cls` — once per stable region, not every frame
- **Change detection:** DINOv2-base (ViT-B/14, 86M params) — crop similarity on re-stabilized regions
- **OCR recognition:** PaddleOCR `PP-OCRv5_rec` (via `paddleocr`)
- **Text diffing:** Python `difflib` — patches Markdown on re-stabilized regions
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
│   ├── main.py                   # Entry point, thread orchestration
│   ├── capture.py                # Camera thread, frame queue
│   ├── board_detector.py         # Stage 1: SAM 3 whiteboard localization → corners
│   ├── person_masker.py          # Stage 2: MediaPipe person mask (raw frame)
│   ├── rectifier.py              # Stage 3: perspective warp of frame + mask
│   ├── board_reconstructor.py    # Stage 4: distance-weighted EMA board model
│   ├── text_detection.py         # Stage 5: PP-OCRv5 raw text line boxes every frame
│   ├── tracker.py                # Stage 6: region lifecycle state machine
│   ├── recognition.py            # Stage 7: layout classify + OCR on newly stable regions
│   ├── pipeline.py               # Orchestrates stages 1–7, writes Markdown to disk
│   └── utils.py                  # Config, logging, helpers
├── models/                       # Downloaded model weights (.gitignore'd)
├── tests/
│   ├── fixtures/                 # Test images of whiteboards
│   └── test_*.py                 # One test file per stage
├── docs/
│   └── architecture.md
├── output/                       # Generated Markdown files
├── requirements.txt
├── CLAUDE.md
└── README.md
```

## Key Commands

```bash
python -m pytest                          # run all tests
python -m pytest tests/test_tracker.py   # single test file
python src/main.py                        # run the pipeline (requires webcam)
python src/main.py --input video.mp4      # run on a video file (for testing)
```

## Architecture Overview

7-stage pipeline. The whiteboard is modeled as a **persistent evolving document**, not a sequence of independent frames. All stages run sequentially in a processing thread. Camera runs in a separate daemon thread with a `Queue(maxsize=1)` for back-pressure (latest frame only).

Stages 1 and 2 both operate on the **raw camera frame** (before any warp) because their models (SAM 3 and MediaPipe) were trained on natural camera images. Stage 3 then warps both the frame and the person mask into the canonical board coordinate system.

| Stage | What | Library |
|-------|------|---------|
| 1. Board Detection | SAM 3 → whiteboard corners (async, periodic) | Ultralytics SAM 3 |
| 2. Person Masking | Person mask on raw frame | MediaPipe |
| 3. Rectification | Perspective warp of frame + mask → canonical view | OpenCV |
| 4. Board Reconstruction | Distance-weighted EMA board model | OpenCV |
| 5. Text Detection | Raw text line boxes every frame | PaddleOCR (`PP-OCRv5_det_server`) |
| 6. Region Tracker | Lifecycle state machine, persistence, stability | Pure Python + DINOv2-base |
| 7. Recognition | Layout classify + OCR on newly stable regions, Markdown patch | PaddleOCR |

---

## Region Tracker Design (Stage 6)

This is the core of the system. The tracker maintains a registry of persistent `Region` objects across frames. The whiteboard is treated as a persistent document — regions are long-lived entities, not per-frame detections.

### Region Data Structure

```python
class Region:
    id: int
    bbox: tuple[int, int, int, int]      # x1, y1, x2, y2 — EMA-smoothed
    confidence: float
    state: RegionState                    # NEW | GROWING | STABLE | OCR_DONE | ERASED
    first_seen: float                     # timestamp
    last_seen: float
    last_modified: float
    stable_frames: int
    missing_frames: int
    ocr_text: str | None                  # cached OCR result
    ocr_confidence: float | None
    last_stable_crop: np.ndarray | None   # crop at last stabilization
```

### State Machine

```
NEW → GROWING → STABLE → OCR_DONE
                              ↓ (content changed)
                           GROWING → STABLE → OCR_DONE  (text diff + patch)
                              ↓ (missing too long)
                           ERASED → remove from Markdown
```

| State | Meaning | Transition |
|-------|---------|------------|
| NEW | Just appeared | → GROWING after 2–5 consecutive visible frames |
| GROWING | Actively changing | → STABLE after `stable_frames > 20` (tune empirically) |
| STABLE | Unchanged, eligible for OCR | → OCR_DONE after OCR runs |
| OCR_DONE | OCR result cached | → GROWING if DINOv2 cosine(current, last_stable_crop) < 0.92 |
| ERASED | Gone from board | Remove region after `missing_frames > 20` |

Any bbox change or content change resets `stable_frames = 0`.

### Detection Matching

Each frame, raw detections from Stage 5 are matched to existing regions:

```
score = 0.7 * IoU + 0.3 * centroid_similarity
```

- `score > 0.4` → update existing region
- No match → create new region in NEW state

### Bounding Box Smoothing

Raw detector output jitters between frames. Apply EMA:

```
smoothed_bbox = 0.2 * detected_bbox + 0.8 * previous_bbox
```

### Region Persistence

Do not remove a region immediately when its detection disappears — it may be a transient occlusion or lighting flicker. Increment `missing_frames` instead. Only transition to ERASED after `missing_frames > 20`.

### Re-stabilization and Text Diffing

When a region cycles OCR_DONE → GROWING → STABLE again (professor modified something):

1. Run DINOv2 on current crop vs `last_stable_crop` to confirm content actually changed.
2. Re-OCR the full region crop.
3. Diff new text against `region.ocr_text` using `difflib.unified_diff`.
4. Patch the Markdown document with only the changed lines (additions, removals, edits).
5. Update `region.ocr_text` and `region.last_stable_crop`.

Markdown updates are surgical — existing content is never blindly overwritten.

### Layout Classification

`RT-DETR-H_layout_17cls` runs **once per region**, only when it first reaches STABLE. It classifies the region as text / table / formula / figure, which determines how Stage 6 OCRs it. It does NOT run every frame.

---

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
- **DINOv2 expects RGB input, normalized.** Convert from BGR, resize to a multiple of 14, normalize with ImageNet mean/std. Don't feed it raw OpenCV arrays.
- **Board reconstruction uses distance-weighted EMA, not MOG2.** The learning rate drops to zero near person pixels, so occluded board regions are frozen at their last known value rather than corrupted by the person's appearance.
- **`Queue(maxsize=1)` drops frames intentionally.** Use `put_nowait()` with a try/except or `get()` the old frame first. This is correct behavior, not a bug.
- **Model files are large.** Do not commit them to git. Add `models/` to `.gitignore`. PaddleOCR and DINOv2 download weights automatically on first run.
- **PaddlePaddle is a separate ML framework from PyTorch.** Both will be installed. PaddleOCR uses PaddlePaddle; DINOv2 uses PyTorch. This is intentional — they don't conflict but the install is large.

## Architecture Reference

Full design spec, latency budgets, model details, and risk mitigations: `docs/architecture.md`
