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
│   ├── change_detection.py       # Gate: skip stages 5–7 if nothing changed
│   ├── layout.py                 # PP-DocLayout region classification
│   ├── text_detector.py          # Stage 5: PP-OCRv5 raw text line boxes every frame
│   ├── tracker.py                # Stage 6: region lifecycle state machine
│   ├── recognizer.py             # Stage 7: OCR on newly stable regions, text diff
│   ├── assembly.py               # Stage 8: spatial ordering → Markdown document
│   ├── document.py               # WhiteboardDoc — persistent Markdown output model
│   ├── pipeline.py               # Orchestrates stages 1–8, writes Markdown to disk
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
| — Change Detection | Gate: skip 5–7 if board unchanged | OpenCV |
| 5. Text Detection | Raw text line boxes every frame | PaddleOCR (`PP-OCRv5_det_server`) |
| 6. Region Tracker | Lifecycle state machine, persistence, stability | Pure Python + DINOv2-base |
| 7. Recognition | OCR on newly stable regions, text diff + patch | PaddleOCR |
| 8. Document Assembly | Spatial ordering → Markdown document | Python `difflib` |

---

## Region Tracker Design (Stage 6)

This is the core of the system. The tracker maintains a registry of persistent `Region` objects across frames. The whiteboard is treated as a persistent document — regions are long-lived entities, not per-frame detections.

### Region Data Structure

```python
class Region:
    id: int
    bbox: np.ndarray                      # shape (4,) int32: x1, y1, x2, y2 — EMA-smoothed
    confidence: float
    state: RegionState                    # CANDIDATE | STABILIZING | STABLE | MISSING | REMOVED
    first_seen: float                     # time.monotonic() timestamps
    last_seen: float
    last_modified: float
    ocr_text: str | None                  # set by Recognizer via tracker.mark_ocr_done()
    ocr_confidence: float | None
    last_stable_crop: np.ndarray | None   # BGR uint8 crop at last stabilization
    last_stable_center: np.ndarray | None # shape (2,) float64 centroid at stabilization
    last_stable_embedding: torch.Tensor | None  # DINOv2 CLS token at stabilization
    line_bboxes: list[np.ndarray]         # sub-line bboxes in board coordinates
```

### State Machine

```
CANDIDATE → STABILIZING → STABLE ←→ STABILIZING (drift resets)
                                ↓ (unmatched > grace)
                             MISSING → REMOVED (purged after retention period)
                                ↑ (re-matched)
                           STABILIZING
```

| State | Meaning | Transition |
|-------|---------|------------|
| CANDIDATE | Just appeared | → STABILIZING after `stabilizing_time_threshold` s of presence |
| STABILIZING | Building stability | → STABLE after `stable_time_threshold` s without drift |
| STABLE | Stable; OCR runs here | → STABILIZING if centroid drifts > `drift_threshold_px` (ocr_text cleared) |
| MISSING | Unmatched too long | → STABILIZING on re-match; → REMOVED after `missing_time_threshold` s |
| REMOVED | Awaiting purge | Deleted from registry after `removed_time_threshold` s |

OCR text is **metadata**, not a lifecycle state. The Recognizer calls `tracker.mark_ocr_done(region, text, confidence)` to record it on STABLE regions.

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

Unmatched regions are not removed immediately — they transition to MISSING after `grace_time_threshold` seconds. Only removed after `missing_time_threshold` additional seconds. This tolerates transient occlusion and lighting flicker.

### Re-stabilization and Text Diffing

When a STABLE region's centroid drifts beyond `drift_threshold_px` from `last_stable_center` (professor modified something), it resets to STABILIZING and `ocr_text` is cleared. Once it re-stabilizes:

1. Re-OCR the region crop using `line_bboxes` for line-level granularity.
2. Diff new text against the previous `ocr_text` using `difflib.unified_diff`.
3. Patch `WhiteboardDoc.blocks[region_id]` with only the changed lines.
4. Call `tracker.mark_ocr_done(region, text, confidence)` to update Region metadata.

Markdown updates are surgical — existing content is never blindly overwritten.

### Layout Classification

`PP-DocLayout_plus-L` (via `layout.py`) classifies regions as text / table / formula / figure. It runs on the board composite image periodically (async child process, same pattern as text detection). Its output determines which regions Stage 7 OCRs and how. It does NOT run every frame.

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
