# CLAUDE.md — Whiteboard Transcriber Runbook

> **Lecture Historian** — a real-time CV/VLM pipeline that captures the *evolution* of whiteboard content across a lecture session. Every entity written, erased, or corrected is preserved in an append-only temporal ledger and synthesised into structured Markdown output.

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Running the Pipeline](#2-running-the-pipeline)
3. [Architecture Overview](#3-architecture-overview)
4. [8-Stage Pipeline](#4-8-stage-pipeline)
5. [Entity State Machine](#5-entity-state-machine)
6. [Subprocess Design](#6-subprocess-design)
7. [Project Structure](#7-project-structure)
8. [Testing](#8-testing)
9. [Development Rules](#9-development-rules)
10. [Engineering Constraints](#10-engineering-constraints)

---

## 1. Environment Setup

### Prerequisites

- macOS on Apple Silicon (M-series). MPS is required for model inference.
- [Nix](https://nixos.org/) with flakes enabled.
- [direnv](https://direnv.net/) (recommended).

### Bootstrap

```bash
# Allow direnv to activate the Nix dev shell automatically on cd
direnv allow

# Or enter the dev shell manually
nix develop

# Install Python dependencies into the Nix-managed venv
pip install -r requirements.txt
```

The Nix flake (`flake.nix`) pins **Python 3.13** and manages the `.venv` via `venvShellHook`. If the venv Python version mismatches the flake version, delete `.venv` and reload the shell.

### Model Weights

Place the following model files under `models/` before running:

| File | Source |
|------|--------|
| `models/sam3.1_multiplex.pt` | Ultralytics SAM 3.1 |
| `models/selfie_segmenter.tflite` | MediaPipe Selfie Segmenter |

HuggingFace models (`stepfun-ai/GOT-OCR-2.0-hf`, `mlx-community/PaddleOCR-VL-1.5-8bit`) and PaddleOCR (`PP-OCRv5_server_det`) are downloaded automatically on first run.

---

## 2. Running the Pipeline

All commands are run from the project root. The entry point is `src/main.py`.

```bash
# Live webcam, default settings
python src/main.py

# Video file input
python src/main.py recording.mp4

# Custom output directory
python src/main.py --output-dir /tmp/lecture recording.mp4

# Swap layout detector
python src/main.py --detector hdbscan recording.mp4
python src/main.py --detector aabbtree recording.mp4

# Swap OCR backend
python src/main.py --transcriber got recording.mp4
python src/main.py --transcriber mock recording.mp4   # dev/test (no model load)

# Verbose logging (propagates LOG_LEVEL=DEBUG to all subprocesses)
python src/main.py --debug recording.mp4

# Adjust display window width
python src/main.py --display-width 1280
```

### CLI Reference

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `source` | positional | webcam | Video/image file path |
| `--detector` | `unionfind\|hdbscan\|aabbtree\|singlelinkage` | `unionfind` | Stage 5/6 layout grouping strategy |
| `--transcriber` | `mock\|got\|paddlevl` | `paddlevl` | Stage 7 OCR backend |
| `--output-dir` | path | `output/` | Directory for `live.md` and `lecture_history.md` |
| `--display-width` | int | `960` | Preview window width in pixels |
| `--debug` | flag | off | Set root log level to DEBUG across all processes |

### Keyboard Controls (Live Window)

| Key | Action |
|-----|--------|
| `q` | Quit |
| `w` | Toggle board corner overlay |
| `p` | Toggle person mask overlay |
| `t` | Toggle text-block overlay |
| `r` | Toggle entity tracker overlay |

### Output Files

Both files are written atomically (tmp + rename) on every OCR result:

- **`output/live.md`** — spatial snapshot of all currently visible entities, sorted top-to-bottom.
- **`output/lecture_history.md`** — full chronological ledger including erased content, with a TOC and collapsible revision history per entity.

---

## 3. Architecture Overview

The pipeline is built around three principles:

**Non-blocking.** Every heavy model (SAM 3.1, PaddleOCR, VLM) runs in an isolated `multiprocessing.Process`. The main loop never waits for a model — it submits work and immediately returns the latest cached result.

**Append-only.** The ledger never deletes. Erasure is recorded as a timestamp, not a deletion. Every OCR correction is a new version appended to the entity's history.

**Coordinate-locked.** All geometry from Stage 3 onward is expressed exclusively in the canonical **1920×1080 rectified space**. Raw camera-space coordinates are never propagated downstream of the rectifier.

### Data Flow Summary

```
Camera / File
    │
    ▼
[Stage 0: capture.py]           Frame queue (maxsize=1, always latest)
    │
    ▼
[Stage 1: BoardMasker]          Async SAM 3.1 → binary board mask
[Stage 2: PersonMasker]         Sync MediaPipe → binary person mask
    │
    ▼
[Stage 3: Rectifier]            Homography → 1920×1080 rectified frame + mask
    │
    ▼
[Stage 4: BoardReconstructor]   Distance-weighted EMA → clean board composite
    │
    ├──▶ [Stage 5/6: Discovery] Async PaddleOCR + grouper → list[Block]
    │
    ▼
[Registry]                      Block → SemanticEntity lifecycle (state machine)
    │
    ├──▶ [Stage 7: TranscriptionWorker] Async VLM → OCR text per entity
    │
    ▼
[Stage 8: Ledger]               Append-only record → live.md + lecture_history.md
```

---

## 4. 8-Stage Pipeline

### Stage 0 — Frame Capture (`capture.py`)

Reads from webcam or file in a background thread. Uses a `Queue(maxsize=1)` — stale frames are dropped automatically, so the main loop always processes the latest frame.

### Stage 1 — Board Masking (`board/masker.py`)

SAM 3.1 runs in a dedicated subprocess with a ~5s cadence. Takes a raw camera frame, returns a binary H×W uint8 mask (1=board, 0=background). Outputs `None` between cycles — the rectifier uses its cached homography when `None` is received. Does **not** perform corner extraction or homography computation.

### Stage 2 — Person Masking (`board/person_masker.py`)

MediaPipe selfie segmenter runs synchronously every frame (~5ms). Returns a binary H×W uint8 mask (1=person). The person mask ensures that pixels under or near the body are never updated in Stage 4, preserving board content under occlusion.

### Stage 3 — Rectification (`board/rectifier.py`)

Owns all geometric computation. When a new board mask arrives from Stage 1, it extracts four corners via convex-hull approximation + `approxPolyDP`, orders them TL/TR/BR/BL, and computes a perspective homography to a canonical 1920×1080 rectangle. The homography is cached and reused every frame. Both the raw frame and person mask are warped to the rectified space.

**Homography update trigger:** ≥2 corners shift >50px *or* the new quad is larger than cached (area ratio ≥0.98).

### Stage 4 — Board Reconstruction (`board/reconstructor.py`)

Maintains a clean board composite using distance-weighted EMA:

```
lr(x) = max_lr × (dist(x, person_mask) / falloff_distance) ^ power
composite = (1 - lr) × composite + lr × frame
```

Pixels near or under the person mask have `lr≈0` and are frozen at their last known value. When no person is detected, a uniform EMA (`lr = max_lr`) is applied directly, skipping the expensive `distanceTransform`.

### Stage 5/6 — Layout Detection (`layout/worker.py`, `layout/`)

`LayoutWorker` manages the `layout-detector` subprocess. Inside the worker:
1. `TextLineDetector` runs PaddleOCR `PP-OCRv5_server_det` synchronously, returning a list of `TextLine` objects (bbox + confidence).
2. A `BaseTextLineClusterer` strategy clusters lines into `Block` objects.

Four grouping strategies are available:

| Strategy | Class | Behaviour |
|----------|-------|-----------|
| `unionfind` | `UnionFindClusterer` | Asymmetric v/h dilation over median line height; early-break on y-gap; width-ratio guard against header absorption |
| `hdbscan` | `HDBSCANClusterer` | Scale-invariant anisotropic distance; noise lines become singleton blocks |
| `aabbtree` | `AABBTreeClusterer` | Greedy agglomerative merge via min-heap + AABB engulfment veto |
| `singlelinkage` | `SingleLinkageClusterer` | Obstacle-vetoed agglomerative merge; nearest-point distance cap |

### Stage 7 — OCR Transcription (`ocr/worker.py`, `ocr/`)

`TranscriptionWorker` manages the `transcription-worker` subprocess. The worker accepts `(entity_id, crop)` pairs from an input queue (`maxsize=10`) and writes `TranscriptionResult` objects to an output queue (`maxsize=30`). Three backends:

| Backend | Class | Notes |
|---------|-------|-------|
| `paddlevl` | `PaddleVLTranscriber` | `mlx-community/PaddleOCR-VL-1.5-8bit` via MLX. Default. |
| `got` | `GotTranscriber` | `stepfun-ai/GOT-OCR-2.0-hf` via HuggingFace, float16 on MPS. CLAHE preprocessing applied. |
| `mock` | `MockTranscriber` | Returns `[mock OCR]`. No model loaded. Use for testing. |

### Stage 8 — Ledger Synthesis (`ledger.py`)

`Ledger` maintains an append-only in-memory record of every entity. On every OCR result or erasure event, it atomically overwrites both output files (write to `.tmp`, then `rename`). The history file includes a generated TOC and collapsible `<details>` revision blocks for versioned entities.

---

## 5. Entity State Machine

The `Registry` (`registry.py`) tracks every detected layout block as a `SemanticEntity` through a 4-state lifecycle.

```
         new block detected
               │
               ▼
        ┌─────────────┐
        │ STABILIZING │ ◀─── centroid drift detected
        └──────┬──────┘
               │ stable_time_threshold elapsed (default: 10s)
               ▼
        ┌─────────────┐
        │  INFERRING  │ ──── crop submitted to VLM worker
        └──────┬──────┘
               │ OCR result received (state must still be INFERRING)
               ▼
        ┌─────────────┐
        │   ACTIVE    │
        └──────┬──────┘
               │ block absent for erase_grace_period (default: 1s)
               ▼
        ┌─────────────┐
        │   ERASED    │ ──── tombstone retained for 3s, then pruned
        └─────────────┘
```

**Key invariants:**

- An entity transitions STABILIZING → INFERRING only after `stable_time_threshold` seconds with no centroid drift exceeding `drift_threshold_px` (default: 50px).
- If drift is detected on an INFERRING or ACTIVE entity, it resets to STABILIZING. Any pending OCR result for that entity is discarded (state check on result receipt).
- `EntityUpdate.entities` contains only non-ERASED entities. Newly-erased entities are reported separately in `EntityUpdate.newly_erased` and simultaneously removed from `pending_ocr` in the main loop.
- Entity identity is spatial: the same text written in a new location gets a new ID; a correction in-place preserves the existing ID and appends a version.

---

## 6. Subprocess Design

Three independent worker processes run throughout a session:

| Process name | Owner class | Model | Queue design |
|---|---|---|---|
| `sam3-board-masker` | `BoardMasker` | SAM 3.1 | in: maxsize=1, out: maxsize=1 (drop-old pattern) |
| `layout-detector` | `LayoutWorker` | PaddleOCR | in: maxsize=1, out: maxsize=1 (drop-old pattern) |
| `transcription-worker` | `TranscriptionWorker` | VLM backend | in: maxsize=10, out: maxsize=30 |

**Drop-old pattern** (used by board masker and layout): the output queue holds at most one result. Before publishing, the worker drains any stale result with `get_nowait()` before `put_nowait()`. The main loop always receives the freshest available result.

**Logging in workers:** Workers call `logging.basicConfig(level=_level, format=...)` before any imports, then call `logging_config.suppress_worker_noise()` to set third-party loggers to WARNING. The level is inherited via the `LOG_LEVEL` environment variable set by `--debug`.

---

## 7. Project Structure

```
whiteboard-transcriber/
├── flake.nix                   # Nix dev environment (Python 3.13, venv)
├── requirements.txt            # Python dependencies
├── models/                     # Local model weights (not committed)
├── output/                     # Generated output (not committed)
├── tests/                      # Pytest test suite
└── src/
    ├── main.py                 # Entry point — pipeline orchestrator + UI
    ├── capture.py              # Stage 0: frame ingestion thread
    ├── logging_config.py       # Third-party noise suppression
    ├── registry.py             # Entity state machine + SemanticEntity
    ├── ledger.py               # Stage 8: append-only ledger + file synthesis
    ├── renderer.py             # OpenCV overlay rendering (display only)
    ├── board/                  # Stages 1–4: visual surface pipeline
    │   ├── masker.py           # Stage 1: SAM 3.1 async subprocess
    │   ├── person_masker.py    # Stage 2: MediaPipe sync per-frame
    │   ├── rectifier.py        # Stage 3: homography + warp to 1920×1080
    │   └── reconstructor.py    # Stage 4: distance-weighted EMA composite
    ├── layout/                 # Stages 5–6: text detection + grouping
    │   ├── base.py             # BaseLayoutDetector ABC
    │   ├── block.py            # Block dataclass
    │   ├── clusterer.py        # BaseTextLineClusterer ABC
    │   ├── text_detector.py    # TextLine dataclass + PaddleOCR detection
    │   ├── block_detector.py   # Composes TextLineDetector + clusterer strategy
    │   ├── worker.py           # LayoutWorker subprocess manager
    │   ├── union_find.py       # Grouping: asymmetric dilation + Union-Find
    │   ├── hdbscan.py          # Grouping: anisotropic HDBSCAN
    │   ├── aabb_tree.py        # Grouping: greedy agglomerative + AABB veto
    │   └── single_linkage.py   # Grouping: obstacle-vetoed agglomeration
    └── ocr/                    # Stage 7: VLM transcription
        ├── base.py             # BaseTranscriber ABC + TranscriptionResult
        ├── worker.py           # TranscriptionWorker subprocess manager
        ├── got.py              # GotTranscriber (GOT-OCR 2.0, HuggingFace)
        ├── paddle_vl.py        # PaddleVLTranscriber (PaddleOCR-VL-1.5, MLX)
        └── mock.py             # MockTranscriber (no model, for testing)
```

---

## 8. Testing

```bash
# Run all tests
pytest tests/

# Run a specific test module
pytest tests/test_rectifier.py

# Run with verbose output
pytest -v tests/

# Run with debug logging
pytest -s tests/
```

Use `--transcriber mock` during manual integration testing — it bypasses model loading and returns immediately, allowing full pipeline validation without GPU/memory overhead.

---

## 9. Development Rules

### Typing

All function signatures must have complete type annotations. No `Any` without a comment justifying it. Use `from __future__ import annotations` for forward references.

```python
# Correct
def tick(self, blocks: list[Block], frame_shape: tuple[int, int]) -> EntityUpdate: ...

# Wrong
def tick(self, blocks, frame_shape): ...
```

### Docstrings

Google-style docstrings are mandatory for all public functions and classes. Private helpers warrant a one-line docstring unless they are trivially obvious. Inline comments are reserved for non-obvious logic (algorithmic invariants, workarounds, non-obvious constraints) — do not comment self-explanatory code.

```python
# Correct: explains a non-obvious invariant
# Lines are sorted by y1; once the vertical gap exceeds 2×v_expand
# all subsequent j values are further away and can be skipped.
if by1 - ay2 > v_expand * 2.0:
    break

# Wrong: restates the code
x = x + 1  # increment x
```

### Model Loading

Model weights must never be loaded in `__init__`. Constructors are picklable config containers only. Load in a `load()` method called inside the worker subprocess after unpickling.

```python
# Correct
class MyDetector(BaseLayoutDetector):
    def __init__(self, threshold: float = 0.6) -> None:
        self._threshold = threshold
        self._model = None  # loaded in load()

    def load(self) -> None:
        self._model = load_model(...)

# Wrong: loads model in __init__ — breaks pickling
class MyDetector(BaseLayoutDetector):
    def __init__(self) -> None:
        self._model = load_model(...)
```

### Subprocess Communication

Use the drop-old queue pattern for single-result producers (board masker, layout detector). Use bounded queues (`maxsize > 1`) only for pipelines that must not drop results (transcription worker). Never use `queue.get()` with a blocking call in the main loop.

### Adding a New Grouper

1. Subclass `BaseTextLineClusterer` from `layout/clusterer.py`.
2. Implement `cluster(lines: list[TextLine]) -> list[Block]`.
3. Add the module to `layout/__init__.py` exports.
4. Register in `main.py`'s `detector_factories` dict and add the choice to `--detector`.

### Adding a New Transcriber Backend

1. Subclass `BaseTranscriber` from `ocr/base.py`.
2. Implement `load()` and `transcribe(crop: np.ndarray) -> str`.
3. Add the module to `ocr/__init__.py` exports.
4. Register in `main.py`'s `transcriber_factories` dict and add the choice to `--transcriber`.

---

## 10. Engineering Constraints

### Coordinate Space

All bounding boxes, centroids, and homography points from Stage 3 onward are expressed exclusively in the **1920×1080 rectified coordinate space**. Raw camera-space coordinates must never be passed to Stage 4 or beyond. The rectifier is the single source of truth for all geometry.

### Non-Blocking Main Loop

The main loop in `main.py` must never block on a model call. Every model interaction uses a non-blocking `put_nowait` / `get_nowait` pattern. If a worker is busy, the main loop proceeds with the last cached result.

### Append-Only Ledger

The `Ledger` class never deletes entries. Mark erasures with a timestamp; append new versions rather than overwriting. File writes use atomic rename (`path.tmp` → `path`) to prevent partial reads by external markdown viewers.

### Subprocess Log Initialisation

Every worker function must call `logging.basicConfig(...)` **before** importing any model library, then call `logging_config.suppress_worker_noise()` to silence third-party loggers. The log level is controlled by the `LOG_LEVEL` environment variable (set by `--debug` in main before workers spawn).

### Memory Budget (Apple Silicon M4, 24GB)

| Allocation | Budget |
|---|---|
| SAM 3.1 + PaddleOCR (CV resident) | ~9 GB |
| VLM worker (GOT-OCR 2.0 float16 or PaddleVL-1.5 8-bit) | ~11 GB |
| OS + frame queues + system overhead | ~4 GB |

If unified memory approaches 22GB, reduce VLM inference frequency before degrading SAM 3.1. Tracking integrity takes priority over OCR throughput.
