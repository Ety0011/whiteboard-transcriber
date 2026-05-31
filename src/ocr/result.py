"""TranscriptionResult — OCR output type shared between ocr and tracker."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class TranscriptionResult:
    """OCR result produced by the transcription worker subprocess.

    Attributes:
        note_id:    NoteTracker ID of the note whose crop was transcribed.
        generation: Dispatch generation counter. Matched against Note.ocr_gen
                    in NoteTracker._commit_ocr_result to discard stale results
                    that arrive after a note has drifted and been re-dispatched.
        text:       Recognised text (and/or LaTeX) returned by the VLM backend.
    """

    note_id: int
    generation: int
    text: str
