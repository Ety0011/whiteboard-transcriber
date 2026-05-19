# CLAUDE.md — Temporal Semantic Whiteboard Ledger

This is the master specification for this project. All implementation decisions must align with this document.

---

## 1. Project Vision: The "Lecture Historian"

This system is not a scanner; it is a **Lecture Historian**. It captures the **evolution of knowledge** across a session. By utilizing Hierarchical Visual Grounding and Vision-Language Models (VLMs), it transforms a physical whiteboard into an append-only **Temporal Ledger**. Even when a professor erases the board, the information is preserved, timestamped, and semantically integrated into a final **Chronological Study Guide**.

---

## 2. Tech Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Foundation** | **SAM 3.1 (Video Mode)** | Real-time board corner tracking & pixel-perfect person/shadow matting. |
| **Neural Surface** | **Temporal-Variance EMA** | Distance-weighted EMA composite + temporal variance filter for specular suppression. |
| **Spatial Anchors** | **Grounded-DINO 2.0** | Detects line-level "Functional Units" (TEXT_LINE, MATH_UNIT). |
| **Grouping** | **Spatial Graph Transformer** | Clusters anchors into "Semantic Entities" based on spatial logic. |
| **The Brain** | **GOT-OCR 2.0 (Point-Prompted)** | High-fidelity VLM OCR/LaTeX via coordinate-grounding. INT4 quantized via MLX. |
| **Identity** | **DINOv3 Embeddings** | Stability verification and content-shift detection across write-erase cycles. |
| **Memory** | **Temporal Event Ledger** | Append-only UUID registry with semantic versioning. |

---

## 3. Hardware Target: MacBook Pro M4 — 24GB Unified Memory

All inference runs on Apple Silicon via **MPS (Metal Performance Shaders)** and **MLX**.

**Memory Partitioning:**
- **CV Resident (9GB):** SAM 3.1 + Grounded-DINO 2.0 + DINOv3 — always resident in unified memory.
- **VLM Worker (11GB):** GOT-OCR 2.0 (INT4 quantized) — runs in isolated process.
- **OS / Buffers (4GB):** Frame queues, system overhead.

**Non-Blocking Architecture:** SAM 3.1, Grounded-DINO 2.0, and GOT-OCR 2.0 each run as independent `multiprocessing.Process` workers. The main loop reads from each model's result queue and always uses the latest cached result. No stage waits on any model.

---

## 4. The 7-Stage Architecture

### Stage 1 & 2: Dynamic Tracking & Matting (SAM 3.1) — Async

SAM 3.1 runs in **Video Tracking Mode** in a background process. It simultaneously:
- Locks onto board corners with sub-pixel precision, even through camera micro-vibrations.
- Segments all foreground occlusions: the professor, arms, markers, **and shadows**.
- Produces a 16-bit alpha mask ("Body Mask") used by Stage 4 for inpainting and by Stage 6 for gesture rejection.

**Gesture Suppression:** Body Mask pixels never contribute to the board model, even when the occluder is motionless.

### Stage 3: Anchor-Refined Rectification

Warps every frame to a canonical **1920×1080** fronto-parallel view. Uses OpenCV perspective transform with the latest board corners from Stage 1. When Spatial Anchors from Stage 5 are available, they serve as additional control points to micro-correct homography drift between corner updates, neutralizing camera vibrations.

> **Coordinate Space Rule:** All stages from Stage 3 onward operate exclusively in the 1920×1080 rectified coordinate space.

### Stage 4: Specular-Free Neural Reconstruction

Maintains a **Clean Board Composite** — the "Gold Standard" image fed to the VLM.

**Two-layer approach:**
1. **Distance-Weighted EMA (base layer):** Each pixel's learning rate is proportional to its distance from the Body Mask. Pixels under/near a person update slowly and are inpainted from the last known clean state. This is the same mechanism as the existing `board_reconstructor.py`.
2. **Temporal Variance Filter (new layer):** If a pixel's intensity varies rapidly across frames *without* a corresponding change in the Body Mask, it is classified as a specular highlight or glare artifact and is suppressed in crops sent to the VLM.

**Ghosting Defense:** A high-pass filter distinguishes "Real Ink" (sharp, high-frequency edges) from "Eraser Smudge" (low-frequency residue). Only pixels above the high-pass threshold are treated as active content.

### Stage 5: Anchor Discovery (Grounded-DINO 2.0) — Async

Runs in a background process. Detects every individual mark as a **Spatial Anchor** in one of two categories:
- `TEXT_LINE` — a line of handwritten natural language.
- `MATH_UNIT` — a mathematical expression, equation, or formula fragment.

Anchors are the atomic unit of the Ledger. Each has a bounding box in rectified space and a category label.

### Stage 6: Hierarchical Grouping (Spatial Graph Transformer)

Analyzes the set of Spatial Anchors and groups them into **Semantic Entities**.

- A group of `MATH_UNIT` anchors stacked vertically beneath a `TEXT_LINE` header → one Semantic Entity (e.g., "The Euler Derivation").
- A paragraph of `TEXT_LINE` anchors → one Semantic Entity.
- Each Semantic Entity is the unit that enters the Entity State Machine (Section 5) and is submitted to the VLM.

### Stage 7: Grounded Brain (GOT-OCR 2.0) — Async

Runs as a dedicated MLX process. Receives an **Entity Crop** — a high-resolution region of the Clean Board Composite — along with the bounding-box coordinates of the constituent Spatial Anchors.

**Point-Prompting:** Anchor coordinates are passed directly to GOT-OCR 2.0 to force the model to attend only to the relevant writing, preventing hallucinations from background noise or adjacent entities.

**Structured Output:**
- `MATH_UNIT` entities → LaTeX.
- `TEXT_LINE` entities → Markdown.

Before inference, every crop is preprocessed with **CLAHE** (Contrast Limited Adaptive Histogram Equalization) and the Stage 4 glare mask to ensure the VLM sees maximum contrast regardless of marker quality.

### Stage 8: Synthesis (The Ledger)

Generates two output files from the UUID Ledger:
- **`live.md`** — spatial snapshot of all currently `ACTIVE` entities, updated after every VLM result.
- **`lecture_history.md`** — full chronological ledger. Every entity is present, including `ERASED` ones. Corrections appear as diff blocks: `→ Correction at HH:MM: "[old]" → "[new]"`.

---

## 5. The Heart: The Entity State Machine

The Ledger tracks each **Semantic Entity** through a strict 7-state lifecycle. Transition logic is handled by `anchor_service/state_machine.py`.

| State | Definition | Transition Trigger |
| :--- | :--- | :--- |
| **DISCOVERED** | New anchor cluster found by Stage 5. | Stage 6 creates a new Semantic Entity. |
| **STABILIZING** | Pixels are constant; DINOv3 confirms no feature drift. | $N$ consecutive frames below movement threshold. |
| **READABLE** | Stage 4 confirms no glare or occlusion over the entity. | Quality check passes AND Body Mask does not overlap entity. |
| **INFERRING** | Entity crop submitted to GOT-OCR 2.0. | Non-blocking submission to VLM queue. |
| **ACTIVE** | Content OCR'd; entity is visible on the physical board. | VLM result received and written to Ledger. |
| **VERSIONED** | A subset of anchors in the entity changed (e.g., a typo corrected). | Sub-pixel shift or new anchor within existing group. |
| **ERASED** | All anchors in the group match the background color. | Stage 5 confirms anchors absent for $M$ frames. |

**Gesture Rejection:** While the Body Mask overlaps an entity, the stabilization timer is **frozen**. The entity cannot transition from `STABILIZING` to `READABLE` until the occlusion clears.

**VERSIONED semantics:** Only the changed anchors trigger re-inference. The UUID is preserved. The Ledger records an "Update" event with the diff. The original text is retained in `lecture_history.md`.

---

## 6. Project Structure

```text
src/
├── main.py                 # Pipeline orchestrator — async model coordination & UI
├── capture.py              # Frame ingestion (Queue maxsize=1, always latest frame)
├── board_service/
│   ├── tracker.py          # Stage 1-2: SAM 3.1 (async) — corner tracking + body/shadow matting
│   ├── rectifier.py        # Stage 3: Sub-pixel perspective warp to 1920×1080
│   └── reconstructor.py    # Stage 4: Distance-weighted EMA + temporal variance glare filter
├── anchor_service/
│   ├── detector.py         # Stage 5: Grounded-DINO 2.0 (async) — TEXT_LINE & MATH_UNIT
│   ├── grouper.py          # Stage 6: Spatial Graph Transformer entity clustering
│   └── state_machine.py    # Entity lifecycle manager (DISCOVERED → ERASED)
├── brain_service/
│   ├── vlm_worker.py       # Stage 7: GOT-OCR 2.0 (async MLX process)
│   └── preprocessor.py     # CLAHE & glare suppression for VLM crops
├── ledger_service/
│   ├── registry.py         # UUID Ledger — source of truth, append-only
│   └── assembly.py         # Stage 8: live.md + lecture_history.md synthesis
└── utils/
    ├── hardware.py         # Apple Silicon MPS/MLX memory management
    └── types.py            # Dataclasses: LedgerEntry, EntityState, AnchorGroup, AnchorType
```

---

## 7. Implementation Rules

**All Models Non-Blocking.** SAM 3.1, Grounded-DINO 2.0, and GOT-OCR 2.0 each run as independent `multiprocessing.Process` workers with input and output queues. The main loop submits work and polls results; it never blocks. If a model is busy, the main loop continues with the last cached result.

**Accuracy-First Preprocessing.** Before any VLM inference, crops receive CLAHE + high-pass filtering. The VLM must see crisp, high-contrast content regardless of lighting.

**Append-Only History.** When an entity enters `ERASED`, its `erased_at` timestamp is written. It moves to the "Archives" section of `lecture_history.md` but is never deleted from the Ledger.

**Spatial Identity.** An entity's UUID is tied to its Spatial Anchor Group, not its content. If a professor erases a formula and rewrites the same formula in a different position, it receives a **new UUID**. If the professor corrects a typo in-place, the **UUID is preserved** and the Ledger records a VERSIONED event.

**Surgical Versioning.** A single-anchor change triggers re-inference of only the affected entity, not the entire board. The correction appears as a diff in `lecture_history.md`.

**Coordinate Integrity.** All geometry — bounding boxes, anchor coordinates, homography points — must be expressed in the 1920×1080 rectified coordinate space. Raw frame coordinates are never passed downstream of Stage 3.

---

## 8. Critical Warnings

**VRAM Contention.** If unified memory usage approaches 22GB, reduce GOT-OCR 2.0 inference frequency before degrading SAM 3.1. Tracking integrity takes priority over OCR throughput.

**Ghosting Trap.** Low-quality erasers leave faint residue. Stage 4's high-pass filter must be tuned to distinguish genuine ink (sharp, high-frequency signal) from eraser smudge (blurry, low-frequency residue). Failing to do so will cause phantom entities in the Ledger.

**Coordinate Drift.** Any corner displacement >2px between consecutive frames must trigger a Stage 3 homography recompute. Allowing drift to accumulate will misalign anchor coordinates with the VLM crops.

**Prompt Integrity.** VLM prompts must explicitly instruct GOT-OCR 2.0 to output LaTeX for `MATH_UNIT` anchors and Markdown for `TEXT_LINE` anchors. Omitting this causes mixed-format output that breaks `assembly.py` synthesis.
