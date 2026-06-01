"""Visualization layer for the whiteboard pipeline.

Owns all OpenCV drawing logic and overlay toggle state. Pipeline code
in main.py stays free of display concerns.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from layout import Block
from tracker import Note, NoteState

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Color palettes (BGR)
# ---------------------------------------------------------------------------


_STATE_COLORS: dict[NoteState, tuple[int, int, int]] = {
    NoteState.STABILIZING: (0, 165, 255),
    NoteState.INFERRING: (0, 200, 255),
    NoteState.ACTIVE: (94, 197, 34),    # #22C55E
    NoteState.ERASED: (38, 38, 220),    # #DC2626
}

_CORNER_LABELS = ["TL", "TR", "BR", "BL"]

# ---------------------------------------------------------------------------
# Pure drawing functions (stateless)
# ---------------------------------------------------------------------------


def _apply_mask_overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Blend a translucent red overlay onto pixels where mask == 1."""
    if not mask.any():
        return frame
    overlay = frame.copy()
    overlay[mask == 1] = (0, 0, 220)
    return cv2.addWeighted(frame, 0.65, overlay, 0.35, 0)


def _draw_corners(
    frame: np.ndarray,
    corners: np.ndarray | None,
    sx: float = 1.0,
    sy: float = 1.0,
) -> np.ndarray:
    """Draw the board quad outline and labeled corner circles onto *frame*."""
    if corners is None:
        return frame

    pts = (corners * np.array([sx, sy], dtype=np.float32)).astype(np.int32)
    cv2.polylines(
        frame,
        [pts.reshape(-1, 1, 2)],
        isClosed=True,
        color=(0, 0, 220),
        thickness=3,
        lineType=cv2.LINE_AA,
    )
    for i, (x, y) in enumerate(pts):
        cv2.circle(frame, (int(x), int(y)), 12, (0, 200, 0), -1, cv2.LINE_AA)
        cv2.putText(
            frame,
            _CORNER_LABELS[i],
            (int(x) + 14, int(y) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return frame


_ANCHOR_COLOR = (255, 165, 0)  # sky blue (BGR)


def _draw_blocks(
    frame: np.ndarray,
    blocks: list[Block],
    sx: float = 1.0,
    sy: float = 1.0,
) -> np.ndarray:
    """Draw translucent line-level bbox fills for all blocks onto *frame*."""
    overlay = frame.copy()
    for block in blocks:
        for line in block.lines:
            x1 = int(line.bbox[0] * sx)
            y1 = int(line.bbox[1] * sy)
            x2 = int(line.bbox[2] * sx)
            y2 = int(line.bbox[3] * sy)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), _ANCHOR_COLOR, -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), _ANCHOR_COLOR, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
    return frame


def _draw_notes(
    frame: np.ndarray,
    notes: list[Note],
    sx: float = 1.0,
    sy: float = 1.0,
) -> np.ndarray:
    """Draw note bboxes and state labels colour-coded by NoteState."""
    overlay = frame.copy()
    for ent in notes:
        x1 = int(ent.bbox[0] * sx)
        y1 = int(ent.bbox[1] * sy)
        x2 = int(ent.bbox[2] * sx)
        y2 = int(ent.bbox[3] * sy)
        color = _STATE_COLORS.get(ent.state, (255, 255, 255))

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        label = f"#{ent.id} {ent.state.value}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(frame, (x1, y1), (x1 + tw + 4, y1 + th + 4), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 2, y1 + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
    return frame


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


class Renderer:
    """Owns overlay toggle state and renders both pipeline display windows."""

    def __init__(self, display_width: int = 960) -> None:
        self._display_width = display_width
        self.show_corners = True
        self.show_mask = True
        self.show_blocks = True
        self.show_tracker = True

    def render_board_panel(
        self,
        composite: np.ndarray,
        blocks: list[Block],
        notes: list[Note],
        layout_busy: bool,
    ) -> np.ndarray:
        """Board panel scaled to display_width. Returns RGB (W,H,3) for pygame.

        Resizes composite to display resolution before drawing overlays — 4× fewer
        pixels to copy and process than drawing at the native 1920×1080.
        """
        h, w = composite.shape[:2]
        target_h = int(h * self._display_width / w)
        # cv2.resize allocates the output; overlays are drawn directly on it.
        board = cv2.resize(composite, (self._display_width, target_h))
        sx = self._display_width / w
        sy = target_h / h
        if self.show_blocks:
            _draw_blocks(board, blocks, sx, sy)
        if self.show_tracker:
            _draw_notes(board, notes, sx, sy)
        cv2.circle(
            board,
            (board.shape[1] - 30, 30),
            10,
            (0, 165, 255) if layout_busy else (0, 255, 0),
            -1,
        )
        rgb = cv2.cvtColor(board, cv2.COLOR_BGR2RGB)
        return rgb.swapaxes(0, 1)

    def render_raw_panel(
        self,
        frame: np.ndarray,
        person_mask: np.ndarray | None,
        cached_corners: np.ndarray | None,
        sam_busy: bool,
    ) -> np.ndarray:
        """Raw camera frame with overlays, scaled to display_width. Returns RGB (W,H,3) for pygame."""
        target_h = int(frame.shape[0] * self._display_width / frame.shape[1])
        raw = cv2.resize(frame, (self._display_width, target_h))
        sx = self._display_width / frame.shape[1]
        sy = target_h / frame.shape[0]
        if self.show_mask and person_mask is not None:
            small_mask = cv2.resize(
                person_mask,
                (self._display_width, target_h),
                interpolation=cv2.INTER_NEAREST,
            )
            raw = _apply_mask_overlay(raw, small_mask)
        if self.show_corners:
            raw = _draw_corners(raw, cached_corners, sx, sy)
        cv2.circle(
            raw,
            (raw.shape[1] - 30, 30),
            10,
            (0, 165, 255) if sam_busy else (0, 255, 0),
            -1,
        )
        rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        return rgb.swapaxes(0, 1)

    def handle_key(self, key: int) -> bool:
        """Handle overlay toggle keys [w/p/t/r]. Returns True if key was consumed."""
        if key == ord("w"):
            self.show_corners = not self.show_corners
            log.info("[w] Corners → %s", "ON" if self.show_corners else "OFF")
        elif key == ord("p"):
            self.show_mask = not self.show_mask
            log.info("[p] Mask → %s", "ON" if self.show_mask else "OFF")
        elif key == ord("t"):
            self.show_blocks = not self.show_blocks
            log.info("[t] Blocks → %s", "ON" if self.show_blocks else "OFF")
        elif key == ord("r"):
            self.show_tracker = not self.show_tracker
            log.info("[r] Entities → %s", "ON" if self.show_tracker else "OFF")
        else:
            return False
        return True
