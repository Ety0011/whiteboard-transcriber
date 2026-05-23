# Board Screenshot Specification

Image to place in `poster/screenshots/pipeline_live.png`.
Generate with GPT Image 2.0 using the prompt below.

---

## Image Generation Prompt

A photorealistic screenshot of a computer vision pipeline running on a clean, flat-on whiteboard. The whiteboard fills the entire frame, perfectly rectangular with no perspective distortion, no person visible, bright white surface with faint marker residue suggesting recent use. The image looks like a raw camera feed processed through a homography correction — clinical, technical, not artistic.

On the whiteboard, handwritten in black marker, are five text blocks in different positions spread naturally across the surface. Each block is outlined by a solid or dashed colored rectangle — a CV bounding box overlay drawn by software, not by hand. Each rectangle has a small label tag in the top-left corner in the same color as the box, using a monospace font, showing the entity ID and state name.

**Block 1 — top-left, ERASED:**
The text has been completely erased. No letters visible. Inside the box is only a faint diffuse dark cloud — a soft, slightly grayish smudge on the white surface, the kind of residue left after a thorough wipe. No streaks, no readable text, just a subtle dark haze. Bounding box is a dashed rectangle in color #DC2626 (red). Label tag reads: `#1 ERASED`

**Block 2 — top-center, ACTIVE:**
Text: "Forward pass: make a prediction" — clean dark handwriting, naturally wrapping across two lines. Solid bounding box in color #22C55E (green). On the top-left corner of the box, sitting on top of the bounding box border, two rectangular badges with sharp square corners, side by side on the same horizontal line with a small gap between them. First badge: green (#22C55E) background, white text, reads `#2 ACTIVE`. Second badge immediately to its right: neutral gray background, white text, reads `1 revision`. Both badges are exactly the same height, same square corners, same internal horizontal and vertical padding — they look like two identical flat rectangular chips from the same design system, differing only in background color and text. Neither badge is taller or shorter than the other. No rounded corners anywhere.

**Block 3 — middle-left, INFERRING:**
Text: "Loss: measure how wrong" — clean dark handwriting, on a single line. Solid bounding box in color #FFC800 (amber/yellow). Label tag reads: `#3 INFERRING`

**Block 4 — middle-right, ACTIVE:**
Text: "w = w - lr * grad" — clean dark handwriting, all on a single line, written like a simple equation. Solid bounding box in color #22C55E (green). Label tag reads: `#4 ACTIVE`

**Block 5 — bottom-center, STABILIZING:**
Text: "Repeat millions of times" — clean handwriting, slightly fresher ink as if just written, naturally wrapping across two lines. Solid bounding box in color #FFA500 (orange). Label tag reads: `#5 STABILIZING`

The bounding box colors must be exactly: STABILIZING boxes are #FFA500 (orange), INFERRING boxes are #FFC800 (amber), ACTIVE boxes are #22C55E (medium green), ERASED boxes are #DC2626 (red). The label tags use the same color as their box. The ERASED box is dashed; all others are solid.

The overall aesthetic is a developer tool UI overlay — precise, geometric colored rectangles on a clean white surface. No humans, no room background, no shadows from a person. The whiteboard fills the frame edge to edge. Lighting is flat and even. Photorealistic, not illustrated.

Generate the image in landscape orientation, aspect ratio 16:9, high resolution.

---

## Narrative Context

This image shows the pipeline mid-lecture on the topic "How does a neural network learn?".

The board tells this story:
- Professor first wrote "Training is iterative" (now erased — Block 1)
- Then wrote "Forward pass: make a prediction", originally had "It guesses" — corrected in place (Block 2, ACTIVE, 1 revision)
- Added "Loss: measure how wrong" — VLM still processing (Block 3, INFERRING, not in ledger yet)
- Added "w = w - lr * grad" (Block 4, ACTIVE)
- Just wrote "Repeat millions of times", not yet stable (Block 5, STABILIZING, not in ledger yet)

Ledger contains: Block 1 (erased), Block 2 (active, 1 revision), Block 4 (active).
Blocks 3 and 5 are not yet in the ledger — no OCR result received.
Erased entries appear in history as plain entries — no marker, no special styling.

## Corresponding Output Files

### live.md
Only non-erased ledger entries, sorted top-to-bottom. Block 1 (erased), 3 (inferring), 5 (stabilizing) excluded.

```markdown
# Whiteboard

Forward pass: make a prediction

---

w = w - lr * grad
```

### lecture_history.md
All ledger entries chronologically: blocks 1, 2, 4. Block 1 appears as a plain entry — no ERASED marker.

```markdown
# Lecture Notes

## Contents

- [09:03](#ent1-0903): Training is iterative
- [09:07](#ent2-0907): Forward pass: make a prediction
- [09:18](#ent4-0918): w = w - lr * grad

---

## 09:03 {#ent1-0903}

Training is iterative

---

## 09:07 {#ent2-0907}

Forward pass: make a prediction

<details><summary>1 revision</summary>

1. It guesses

</details>

---

## 09:18 {#ent4-0918}

w = w - lr * grad
```
