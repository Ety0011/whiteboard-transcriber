# CLAUDE.md — Temporal Semantic Whiteboard Ledger

This is the master specification for this project. All implementation decisions must align with this document.

---

## 1. Project Vision: The "Lecture Historian"

This system is not a scanner; it is a **Lecture Historian**. It captures the **evolution of knowledge** across a session. By utilizing Hierarchical Visual Grounding and Vision-Language Models (VLMs), it transforms a physical whiteboard into an append-only **Temporal Ledger**. Even when a professor erases the board, the information is preserved, timestamped, and semantically integrated into a final **Chronological Study Guide**.

---

## 2. Tech Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Board Sensing** | **SAM 3.1 (Semantic Text Prompt)** | Async board region segmentation via `text=["whiteboard"]` — board mask drives homography in Stage 3. |
| **Person Sensing** | **MediaPipe Selfie Segmenter** | Sync per-frame person mask — warped to rectified space and used by Stage 4. |
| **Neural Surface** | **Spatial-Glare EMA** | Distance-weighted EMA composite + spatial glare suppression (brightness + Laplacian). |
| **Spatial Anchors** | **PaddleOCR PP-OCRv5_server_det** | Detects line-level `TEXT_LINE` anchors. |
| **Grouping** | **Union-Find IoU Clustering** | Clusters anchors into "Semantic Entities" via pairwise IoU on expanded bboxes. |
| **The Brain** | **GOT-OCR 2.0 (Masked Crop)** | High-fidelity VLM OCR/LaTeX on per-entity masked crops. float16 via HuggingFace Transformers on MPS. |
| **Memory** | **Temporal Event Ledger** | Append-only registry with semantic versioning. |

---

## 3. Hardware Target: MacBook Pro M4 — 24GB Unified Memory

All inference runs on Apple Silicon via **MPS (Metal Performance Shaders)**.

**Memory Partitioning:**
- **CV Resident (9GB):** SAM 3.1 + PaddleOCR — always resident in unified memory.
- **VLM Worker (11GB):** GOT-OCR 2.0 (float16) — runs in isolated process.
- **OS / Buffers (4GB):** Frame queues, system overhead.

**Non-Blocking Architecture:** SAM 3.1, PaddleOCR, and GOT-OCR 2.0 each run as independent `multiprocessing.Process` workers. The main loop reads from each model's result queue and always uses the latest cached result. No stage waits on any model.

---

## 4. The 8-Stage Architecture

### Stage 1: Board Masking (SAM 3.1) — Async

SAM 3.1 runs in a background process (`board_masker.py`), segmenting the whiteboard region each cycle (~10s cadence). Outputs a raw board mask in camera-frame space. Corner extraction and homography computation are **not** done here — that is Stage 3's responsibility.

### Stage 2: Person Masking (MediaPipe) — Sync

MediaPipe selfie segmenter runs synchronously every frame (`person_masker.py`). Outputs a person mask in camera-frame space at ~5ms per frame. No background process — always fresh for the current frame.

**Gesture Suppression:** Person mask pixels never contribute to the board model, even when the occluder is motionless.

### Stage 3: Rectification

Owns all geometry for the pipeline. Each time a new board mask arrives from Stage 1, corners are extracted via contour approximation and the homography is recomputed and cached. Every frame, the cached homography warps both the raw frame and the person mask to the canonical **1920×1080** fronto-parallel view.

> **Coordinate Space Rule:** All stages from Stage 3 onward operate exclusively in the 1920×1080 rectified coordinate space.

### Stage 4: Specular-Free Board Reconstruction

Maintains a **Clean Board Composite** — the "Gold Standard" image fed to the VLM.

**Two-layer approach:**
1. **Distance-Weighted EMA (base layer):** `lr(x) = max_lr × (dist(x) / falloff_distance) ^ power`. Pixels under/near the person mask have lr≈0 and are frozen at their last known composite value, preserving written content under occlusions.
2. **Spatial Glare Detection (suppression layer):** Glare pixels are identified per-frame as `brightness ≥ 248 AND |Laplacian| < 15`. They are excluded from the EMA update (composite retains the pre-glare value) and inpainted using `cv2.inpaint` (Telea, zero-VRAM classical CPU inpainting).

### Stage 5: Anchor Discovery (PaddleOCR PP-OCRv5_server_det) — Async

Runs in a background process. Detects every individual line on the board as a `TEXT_LINE` **Spatial Anchor**. Anchors are the atomic unit of the Ledger — each has a bounding box in rectified 1920×1080 space and a confidence score.

### Stage 6: Hierarchical Grouping (Union-Find IoU Clustering)

Analyzes the set of Spatial Anchors and groups them into **Semantic Entities** using pairwise Union-Find clustering over expanded bbox IoU. Handles multi-column layouts by scanning all open groups.

- A cluster of `TEXT_LINE` anchors that are spatially adjacent (same paragraph or derivation) → one Semantic Entity.
- Each Semantic Entity is the unit that enters the Entity State Machine (Section 5) and is submitted to the VLM.

### Stage 7: Grounded Brain (GOT-OCR 2.0) — Async

Runs as a dedicated `multiprocessing.Process` (`transcriber.py`). Model: `stepfun-ai/GOT-OCR-2.0-hf` via HuggingFace Transformers, float16 on MPS. Input queue `maxsize=10`, output queue `maxsize=30` — multiple entities per frame are accepted without dropping.

**Masked Crop:** Rather than coordinate-prompting the model, each entity is passed as a **masked crop** — a white canvas with only that entity's constituent anchor pixels copied onto it. Adjacent entities are blanked out, preventing hallucinations from background noise.

Before inference, every crop is preprocessed with **CLAHE** (Contrast Limited Adaptive Histogram Equalization) on the L channel to maximise contrast regardless of marker quality. CLAHE is inlined in `transcriber.py` (`_preprocess_crop`).

### Stage 8: Synthesis (The Ledger)

Generates two output files from the UUID Ledger:
- **`live.md`** — spatial snapshot of all currently `ACTIVE` entities, updated after every VLM result.
- **`lecture_history.md`** — full chronological ledger. Every entity is present, including `ERASED` ones. Corrections appear as: `→ [HH:MM] Original: "[old text]"`.

---

## 5. The Heart: The Entity State Machine

The Ledger tracks each **Semantic Entity** through a strict 4-state lifecycle. Transition logic is handled by `anchor_service/entity_registry.py` (`EntityRegistry`).

| State | Definition | Transition Trigger |
| :--- | :--- | :--- |
| **STABILIZING** | Ink is settling; entity present but not yet stable. | New entity detected, or edit/drift detected on existing entity. |
| **INFERRING** | Crop captured and submitted to GOT-OCR 2.0. | `stable_time_threshold` elapsed with no significant change. |
| **ACTIVE** | OCR complete; entity visible and live in outputs. | VLM result received and written to Ledger. |
| **ERASED** | Anchors absent from clean board composite. | Entity not matched by any anchor group in current frame. |

**Edit detection:** If an `ACTIVE` or `INFERRING` entity's centroid drifts beyond `drift_threshold_px`, it resets to `STABILIZING` with its UUID preserved and `ocr_text` cleared.

---

## 6. Project Structure

```text
src/
├── main.py                 # Pipeline orchestrator — async model coordination & UI
├── capture.py              # Frame ingestion (Queue maxsize=1, always latest frame)
├── board_service/
│   ├── board_masker.py     # Stage 1: SAM 3.1 (async) — board region mask
│   ├── person_masker.py    # Stage 2: MediaPipe (sync) — person mask per frame
│   ├── rectifier.py        # Stage 3: Corner extraction + H cache + warp to 1920×1080
│   └── reconstructor.py    # Stage 4: Distance-weighted EMA + spatial glare suppression
├── anchor_service/
│   ├── detector.py         # Stage 5: PaddleOCR PP-OCRv5_server_det (async) — TEXT_LINE anchors
│   ├── grouper.py          # Stage 6: Union-Find IoU clustering + masked crop helper
│   └── entity_registry.py  # Entity lifecycle manager + state machine (STABILIZING → ERASED)
├── brain_service/
│   └── transcriber.py      # Stage 7: GOT-OCR 2.0 (async HF process) + _preprocess_crop (CLAHE)
└── ledger_service/
    ├── registry.py         # Append-only session ledger (LedgerRegistry)
    └── assembly.py         # Stage 8: live.md + lecture_history.md synthesis
```

---

## 7. Implementation Rules

**All Models Non-Blocking.** SAM 3.1, PaddleOCR, and GOT-OCR 2.0 each run as independent `multiprocessing.Process` workers with input and output queues. The main loop submits work and polls results; it never blocks. If a model is busy, the main loop continues with the last cached result.

**Accuracy-First Preprocessing.** Before any VLM inference, crops receive CLAHE contrast enhancement on the L channel. The VLM must see crisp, high-contrast content regardless of lighting.

**Append-Only History.** When an entity enters `ERASED`, its `erased_at` timestamp is written. It moves to the "Archives" section of `lecture_history.md` but is never deleted from the Ledger.

**Spatial Identity.** An entity's UUID is tied to its Spatial Anchor Group, not its content. If a professor erases a formula and rewrites the same formula in a different position, it receives a **new UUID**. If the professor corrects a typo in-place, the **UUID is preserved** and the Ledger records a VERSIONED event.

**Surgical Versioning.** A single-anchor change triggers re-inference of only the affected entity, not the entire board. The correction appears as a diff in `lecture_history.md`.

**Coordinate Integrity.** All geometry — bounding boxes, anchor coordinates, homography points — must be expressed in the 1920×1080 rectified coordinate space. Raw frame coordinates are never passed downstream of Stage 3.

---

## 8. Critical Warnings

**VRAM Contention.** If unified memory usage approaches 22GB, reduce GOT-OCR 2.0 inference frequency before degrading SAM 3.1. Tracking integrity takes priority over OCR throughput.

**Ghosting Trap.** Low-quality erasers leave faint residue. Stage 4's glare detector (`|Laplacian| < 15`) must not confuse faint eraser smudge with glare — both are low-frequency. If the brightness threshold (≥ 248) is set too low, smudge gets treated as glare and inpainted, creating phantom clean regions. Tune brightness threshold upward if ghosting occurs.

**Coordinate Drift.** Homography recompute triggers when ≥2 corners shift >50px or the new quad is larger than the cached one (area ratio ≥0.98). Allowing drift to accumulate will misalign anchor coordinates with the VLM crops.
