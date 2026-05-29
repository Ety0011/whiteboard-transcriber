# Presentation Script — Real-Time Whiteboard Transcription

**Format:** Poster session, 3–4 min before live demo + Q&A.  
**Tone:** Conversational, direct. Speak to the person, not the poster.

---

## 1. Welcome + Title (15 s)

> "Welcome — I'm presenting *Real-Time Whiteboard Transcription*: a system that captures the full evolution of a lecture, everything written, corrected, and erased, as it happens."

---

## 2. The Problem (30 s)

> "Here's the core tension in any lecture: when a professor is at the board explaining something, you have two choices — you can pay attention, or you can copy. You can't really do both at the same time."

> "We wanted to eliminate that trade-off. Our project transcribes in real time what the professor writes, so the student is free to just listen and understand."

---

## 3. The Solution: A 10-Stage Pipeline (2 min 30 s)

Point to the flowchart on the poster.

> "Our solution is built on a 10-stage real-time pipeline. Let me walk you through it."

---

**Stage 1 — Video Feed**

> "Everything starts with a video stream from a camera pointed at the whiteboard. Simple input, but the design choice here matters: we use a queue that holds only one frame at a time. Old frames are dropped automatically, so the rest of the pipeline always works with the freshest image."

---

**Stage 2 — Board Segmentation**

> "This stage is responsible for identifying the region of the image that represents the whiteboard — isolating it from the wall, the frame, the room around it."

---

**Stage 3 — Person Segmentation**

> "This does the same thing, but for the person. It produces a mask of wherever the lecturer's body is in the frame."

---

**Stage 4 — Perspective Correction**

> "A camera is rarely positioned perfectly in front of the board. So this stage corrects the perspective and crops to the board region, giving us a clean, rectified view at a fixed resolution. Everything from here on works in that canonical space."

---

**Stage 5 — Surface Reconstruction**

> "This is where we solve the occlusion problem. The professor is always standing in front of something. So we maintain a virtual memory of the board — a running composite built by continuously updating it with the regions where the person is *not* present. The pixels under the body stay frozen at their last known value. The result is a clean, unoccluded image of the full board at all times."

---

**Stage 6 — Text Line Detection**

> "Now that we have a clean, rectified, unoccluded view of the board, we can start reading it. This stage detects all the regions that contain text."

---

**Stage 7 — Block Grouping**

> "Then we group those text regions together into logical blocks — paragraphs, headings, equations. Lines that belong together become one unit."

---

**Stage 8 — Entity Registry** *(pause here, this is the core)*

> "This is the heart of the system. Every block gets inserted into a registry, and from that point on we monitor how it evolves over time. The registry is what decides what actually gets transcribed — a block has to prove it's stable before we trust it."

Point to the state machine diagram.

> "A new block starts as *Stabilizing*. If it stays in the same position for ten seconds without significant movement, it advances to *Inferring* — we send it to the OCR model. When the transcription comes back, it becomes *Active*, and the text is recorded. If the block disappears from the board for more than a second, it's marked *Erased*, with a timestamp. And any movement at any stage — even a small correction — resets the clock."

---

**Stage 9 — OCR Transcription** *(brief)*

> "The actual transcription is done by a vision-language model running in a background subprocess — it reads the image crop and returns the text."

---

**Stage 10 — Ledger Synthesis**

> "Finally, everything gets archived. The full history of every block seen during the session — including everything that was erased — is written to *lecture_history.md*. And the current state of the board, right now, is always available in *live.md*."

---

## 4. Hand Off to Demo (15 s)

> "Let me show you what this looks like in practice."

*[Start demo.]*

---

## Q&A Prompts (for evaluator questions)

**Why wait 10 seconds before transcribing?**
> "VLMs are slow. If we transcribed every frame we'd build a queue that never drains. The stability gate means we only transcribe content the professor intends to keep."

**Why subprocesses rather than threads?**
> "Python's GIL throttles CPU-bound work. And models like MLX and PaddleOCR hold large amounts of GPU memory — subprocess isolation means each one loads once, crashes in one don't affect the rest, and the main loop's latency stays predictable."

**What if the perspective changes mid-lecture?**
> "The board masker runs every five seconds and updates the homography if it detects the camera or board has moved — specifically if at least two corners shift more than 50 pixels or the new quadrilateral is significantly larger than the cached one."

**Why two output files?**
> "Different use cases. live.md is for the student following along now. lecture_history.md is for reviewing later — it has every block in order, with timestamps, revisions, and everything that was erased."

---

*Estimated time to demo: 3 min 15 s at a comfortable pace.*
