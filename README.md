# Real-Time Whiteboard Transcription

Real-time computer vision pipeline that captures the *evolution* of whiteboard content across a lecture session. Every entity written, erased, or corrected is preserved in an append-only temporal ledger and synthesised into structured Markdown output.

Revisions are appended as new versions under the same entity ID — the ledger never overwrites or deletes.

---

## How it works

1. **Segments** the board surface from background and people (SAM 3.1 + MediaPipe)
2. **Rectifies** perspective to a canonical 1920×1080 frame
3. **Composites** a clean board image, freezing pixels occluded by the presenter
4. **Detects** text regions (PaddleOCR `PP-OCRv5_server_det`)
5. **Tracks** each region across frames through a 4-state lifecycle: STABILIZING → INFERRING → ACTIVE → ERASED
6. **Transcribes** stable regions via VLM (`PaddleOCR-VL-1.5-8bit` on MLX)
7. **Writes** two Markdown files atomically on every update:
   - `output/live.md` — spatial snapshot of currently visible content
   - `output/lecture_history.md` — full chronological ledger with revision history per entity

Everything runs non-blocking: three worker subprocesses (SAM, PaddleOCR, VLM) run independently. The main loop never stalls waiting for a model.

### Pipeline

```
Camera / File
    │
    ▼
[S1] capture/video.py          Frame queue (maxsize=1, always latest)
    │
    ▼
[S2] BoardSegmenter            Async SAM 3.1 → board mask (~5s cadence)
[S3] PersonSegmenter           Sync MediaPipe → person mask (~5ms/frame)
    │
    ▼
[S4] Rectifier                 Homography → 1920×1080 rectified space
    │
    ▼
[S5] BoardCompositor           Distance-weighted EMA → clean board
    │
    ├──▶ [S6] TextLineDetector  Async PaddleOCR → list[TextLine]
    ├──▶ [S7] BlockGrouping     Single-linkage clustering → list[Block]
    │
    ▼
[S8] NoteTracker               Block → Note lifecycle (state machine)
    │
    ├──▶ [S9] TranscriptionWorker  Async VLM → TranscriptionResult
    ▼
[S10] Ledger                   Append-only record → live.md + lecture_history.md
```

### Memory budget (Apple Silicon M4, 24 GB)

| Allocation | Budget |
|---|---|
| SAM 3.1 + PaddleOCR | ~9 GB |
| VLM worker (PaddleVL-1.5 8-bit) | ~11 GB |
| OS + overhead | ~4 GB |

---

## Setup

### Prerequisites

- macOS on Apple Silicon (M-series) — MPS required
- Python 3.13

### Install

```bash
pip install -r requirements.txt
```

### Model weights

Place the following files under `models/` before running:

| File | Source |
|------|--------|
| `models/sam3.1_multiplex.pt` | Ultralytics SAM 3.1 — **gated model, requires access request at [ultralytics.com](https://www.ultralytics.com/)** |
| `models/selfie_segmenter.tflite` | MediaPipe Selfie Segmenter |

HuggingFace model (`mlx-community/PaddleOCR-VL-1.5-8bit`) and PaddleOCR (`PP-OCRv5_server_det`) download automatically on first run.

---

## Running

```bash
# Live webcam
python src/main.py

# Video file
python src/main.py recording.mp4

# Custom output directory
python src/main.py --output-dir /tmp/lecture recording.mp4

# Mouse-drawable canvas (no camera needed)
python src/main.py --canvas

# Verbose logging
python src/main.py --debug recording.mp4

# Adjust window width
python src/main.py --display-width 1280
```

### Keyboard controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `space` | Pause / resume |
| `c` | Clear canvas (`--canvas` mode only) |
| `w` | Toggle board corner overlay |
| `p` | Toggle person mask overlay |
| `t` | Toggle text-block overlay |
| `r` | Toggle entity tracker overlay |

---

## Demo

Run the pipeline on the included lecture recording:

```bash
python src/main.py videos/YTDown_YouTube_Can-x-3-i_Media_6LVOMaQwDmU_001_1080p.mp4
```

Results are written live to `output/live.md` (current board snapshot) and `output/lecture_history.md` (full session ledger).

