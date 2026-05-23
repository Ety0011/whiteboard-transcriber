A0 academic research poster, portrait orientation (841 × 1189 mm, ~1:1.41 ratio), 2K resolution. Swiss graphic design school style — clean, minimal, structured, zero decorative flourishes. White background. Three zones: full-width header, two-column body. Professional typeset quality. No AI artifacts. No lorem ipsum. Render ALL quoted text verbatim — no paraphrasing, no substitutions, no word changes.

---

## COLOR PALETTE (USI brand)

- Black #000000 — section rule borders, primary headings
- White #FFFFFF — background, label text on dark fills
- Faculty accent blue #0057A8 — section title left-border bars, badge fills only
- Light gray #F4F4F4 — card backgrounds
- Pipeline nodes follow a rainbow gradient top-to-bottom (Google Material palette). Same-stage nodes share a color stop:
    Node 1  (Video Feed)                      — Google Red    #EA4335, white text
    Nodes 2–4 (Board/Person/Perspective)      — Google Orange #FA7B17, white text
    Node 5  (Surface Reconstruction)          — Google Yellow #F9AB00, dark text #1a1a1a
    Nodes 6–7 (Text Detection/Block Grouping) — Google Green  #34A853, white text
    Node 8  (Entity Registry)                 — Google Blue   #1A73E8, white text
    Node 9  (OCR Transcription)               — Indigo        #5C6BC0, white text
    Node 10 (Ledger Synthesis)                — Google Purple #9334E6, white text
- Red tint #FFF0F0 with thin #ef4444 border — problem callout chip backgrounds
- Terminal pane backgrounds: dark charcoal #1E1E2E
- Body text: dark gray #333333
- State machine colors match codebase renderer.py exactly: STABILIZING #FFA500, INFERRING #FFC800, ACTIVE #22C55E, ERASED #DC2626

---

## TYPOGRAPHY

- Font family: Akzidenz-Grotesk (fallback: Helvetica Neue or Inter). Clean sans-serif throughout.
- Monospace only inside terminal panes.
- Main title: very large, bold, black.
- Section titles: bold, medium-large, black, with a 4px solid #0057A8 left-border bar and 8px left padding.
- Body text: 11pt, #333333.
- Node labels: bold white (or bold dark text where fill is light).
- Badge text: small, white, on #0057A8 fill.
- Annotations: small italic #555555, to the right of each node.
- Callout icons: small solid #0057A8.

---

## ZONE 1 — HEADER (full width, ~95mm tall)

White background. Two rows.

**Top row:** Left-aligned: small rectangular placeholder box, dashed border, labeled "USI logo — insert from desk.usi.ch" (approx 40×15mm). To the right of logo, two lines stacked:
- Line 1 — very large bold black: "Real-Time Whiteboard Transcription"
- Line 2 — medium italic gray: "A system that watches a whiteboard like a historian — recording every idea written, corrected, and erased."

**Bottom row:** thin 1px #0057A8 horizontal rule, then three columns of small text beneath it:
- LEFT: "USI Università della Svizzera italiana / Faculty of Informatics"
- CENTER: "Etienne Orio · Sara Mashhadi Alizadeh"
- RIGHT: "Computer Vision · M.Sc. Artificial Intelligence · May 28, 2026"

---

## ZONE 2 — LEFT COLUMN (40% width, below header)

---

### Section A — "The Problem"

Section title with blue left border: "The Problem"

Body text:
"A professor walks up to a whiteboard. They write, explain, erase, correct, and move on. By the end of the lecture the board looks nothing like it did at the start. A photo of the final board captures the conclusion — it misses everything that happened in between. We wanted to capture the full story."

Three callout chips, each full-width, pill-shaped, light red-tinted fill #FFF0F0, thin red border, small red ✕ icon on the left:

1. "Students spend all their time copying and miss the actual explanation."
2. "When the professor erases to make room, that content is gone forever."
3. "There's no way to replay how an idea developed, only its final form survives."

---

### Section B — "How It Works"

Section title with blue left border: "How It Works"

Intro line (body text):
"A camera feeds a 10-stage real-time pipeline. Models run in isolated subprocesses — the main loop never blocks on inference."

**Vertical flowchart.** 10 rounded-rectangle nodes, connected by downward arrows. Each node: bold white label centered inside, small italic gray annotation to the right, model badge (small white text on #0057A8 pill) below the label where specified. Consistent width. Tight vertical spacing.

Node 1 — Google Red #EA4335 fill, white text:
  Label: "Video Feed"
  Annotation: "Queue(maxsize=1) — stale frames dropped. Always the freshest."

Node 2 — Google Orange #FA7B17 fill, white text:
  Label: "Board Segmentation"
  Badge: "SAM 3.1"
  Annotation: "Segments whiteboard region. Async subprocess · ~5 s cadence."

Node 3 — Google Orange #FA7B17 fill, white text:
  Label: "Person Segmentation"
  Badge: "MediaPipe Selfie Segmenter"
  Annotation: "Tracks lecturer's body every frame. Sync · ~5 ms."

Node 4 — Google Orange #FA7B17 fill, white text:
  Label: "Perspective Correction"
  Annotation: "Homography → canonical 1920×1080. All downstream geometry locked to this space."

Node 5 — Google Yellow #F9AB00 fill, dark text #1a1a1a:
  Label: "Surface Reconstruction"
  Annotation: "Distance-weighted EMA. Pixels under body frozen at last known value."

Node 6 — Google Green #34A853 fill, white text:
  Label: "Text Line Detection"
  Badge: "PaddleOCR PP-OCRv5_server_det"
  Annotation: "Detects every text line on clean composite. Async subprocess."

Node 7 — Google Green #34A853 fill, white text:
  Label: "Block Grouping"
  Annotation: "Clusters lines into blocks. Pluggable strategy: Union-Find · HDBSCAN · AABB-Tree."

Node 8 — Google Blue #1A73E8 fill, white text:
  Label: "Entity Registry"
  Annotation: "Tracks blocks across frames. IoU + centroid matching. EMA bbox. Stable 10 s → dispatch OCR."

Node 9 — Indigo #5C6BC0 fill, white text:
  Label: "OCR Transcription"
  Badge: "PaddleOCR-VL-1.5  (default)  ·  GOT-OCR 2.0  (alt)"
  Annotation: "VLM reads each stable entity crop. Async subprocess."

Node 10 — Google Purple #9334E6 fill, white text:
  Label: "Ledger Synthesis"
  Annotation: "Re-renders live.md + lecture_history.md from in-memory ledger. Atomic overwrite (tmp → rename). In-memory record never deletes — only accumulates."

---

## ZONE 3 — RIGHT COLUMN (60% width, below header)

---

### Panel 1 — "System in Action" (~35% right-column height)

Section title with blue left border: "System in Action"

Large dashed-border rectangle, #F4F4F4 fill, centered italic text:
"[ INSERT SCREENSHOT HERE — live pipeline with colour-coded entity bounding boxes: orange=STABILIZING · yellow=INFERRING · green=ACTIVE ]"

---

### Panel 2 — "Output Files" (~30% right-column height)

Section title with blue left border: "Output Files"

Two side-by-side terminal-style panes. Each pane: charcoal #1E1E2E background, thin #0057A8 top border accent, monospace font, subtle drop shadow.

Left pane:
- Header bar: "📄 live.md" in white monospace, left #0057A8 accent stripe
- Body: dashed-border box, light gray italic text: "[ INSERT SCREENSHOT HERE — live.md: current board snapshot, one block per active entity, sorted top-to-bottom ]"
- Caption below (small italic black): "Always up to date. Reflects exactly what is visible on the board right now."

Right pane:
- Header bar: "📖 lecture_history.md" in white monospace, left #0057A8 accent stripe
- Body: dashed-border box, light gray italic text: "[ INSERT SCREENSHOT HERE — lecture_history.md: full session ledger with TOC, timestamps, and collapsible revision history per entity ]"
- Caption below (small italic black): "Nothing is lost. Erased content is timestamped and stays in the in-memory ledger. Corrections appear as numbered revisions."

---

### Panel 3 — "Every Block Has a Lifecycle" (~30% right-column height)

Section title with blue left border: "Every Block Has a Lifecycle"

Body text:
"Once detected, a block lives through four states. Identity is spatial — a correction in-place is a new version of the same entity; a rewrite at a new location is a new entity entirely."

Render this diagram faithfully, using the ASCII layout below as the exact structural reference. Replace ASCII boxes with styled rounded-rectangle nodes filled with the specified colors. Replace ASCII arrows with clean vector arrows. Render all labels verbatim.

```
  ┌───────────────┐
  │  STABILIZING  │ ◀─────────────────────┐
  └───────┬───────┘   drift > 50 px       │
          │                               │
          │  stable for 10 s              │ drift > 50 px
          ▼                               │ (edit in place)
  ┌───────────────┐                       │
  │   INFERRING   │ ──────────────────────┤
  └───────┬───────┘                       │
          │                               │
          │  OCR result received          │
          ▼                               │
  ┌───────────────┐                       │
  │    ACTIVE     │ ──────────────────────┘
  └───────┬───────┘
          │
          │  absent for 1 s
          ▼
  ┌───────────────┐
  │    ERASED     │
  └───────────────┘
```

Node colors (fill, white bold label):
- STABILIZING: orange #FFA500
- INFERRING: golden #FFC800, dark text
- ACTIVE: bright green #22C55E, dark text
- ERASED: red #DC2626

Small italic annotation to the right of each node:
- STABILIZING: "new block, or drift reset"
- INFERRING: "crop sent to VLM"
- ACTIVE: "text written to ledger"
- ERASED: "archived with timestamp · pruned after 3 s"

Small note below diagram, italic #555555:
"Any movement — including edits to already-transcribed blocks — resets the stability clock."

---


**Final hard constraints:** Render every quoted string verbatim — no AI paraphrasing, no substitutions, no extra words. Placeholder boxes must display their dashed border and label text legibly at poster scale. No decorative elements beyond what is specified above. No lorem ipsum anywhere. Consistent grid alignment throughout all four zones. The overall aesthetic must feel like a Swiss typographic poster — clean, structured, confident, minimal.
