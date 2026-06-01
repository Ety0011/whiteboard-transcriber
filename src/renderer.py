"""Visualization layer — overlay state and pygame surface rendering.

Owns all OpenCV drawing logic and overlay toggle state. Converts BGR numpy
frames to pygame Surfaces. Surfaces are reused across frames via blit_array
to avoid per-frame allocation.
"""

from __future__ import annotations

import cv2
import numpy as np
import pygame

from layout import Block
from tracker import Note, NoteState

# ---------------------------------------------------------------------------
# Color palettes (BGR)
# ---------------------------------------------------------------------------

_STATE_COLORS: dict[NoteState, tuple[int, int, int]] = {
    NoteState.STABILIZING: (0, 165, 255),
    NoteState.INFERRING:   (0, 200, 255),
    NoteState.ACTIVE:      (94, 197, 34),
    NoteState.ERASED:      (38, 38, 220),
}
_CORNER_LABELS = ["TL", "TR", "BR", "BL"]
_ANCHOR_COLOR = (255, 165, 0)

# ---------------------------------------------------------------------------
# BGR → pygame Surface
# ---------------------------------------------------------------------------


def _bgr_to_surface(bgr: np.ndarray, cache: pygame.Surface | None) -> pygame.Surface:
    """Convert a BGR (H, W, 3) numpy array to a pygame Surface.

    On first call or dimension change a new surface is allocated via
    make_surface. On subsequent same-size calls blit_array updates the surface
    in-place, avoiding per-frame heap allocation.

    Args:
        bgr: BGR uint8 image of shape (H, W, 3).
        cache: Previously returned Surface to reuse, or None.

    Returns:
        pygame.Surface of size (W, H) with updated pixel data.
    """
    h, w = bgr.shape[:2]
    # Both ops are zero-copy numpy views: channel flip + axis swap.
    rgb_wh3 = bgr[:, :, ::-1].swapaxes(0, 1)
    if cache is None or cache.get_size() != (w, h):
        return pygame.surfarray.make_surface(rgb_wh3)
    pygame.surfarray.blit_array(cache, rgb_wh3)
    return cache


# ---------------------------------------------------------------------------
# Pure drawing helpers (stateless, operate on BGR numpy arrays)
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
    """Owns overlay toggle state and renders pipeline panels as pygame Surfaces.

    Each render method resizes the input to display_width, draws overlays at
    that resolution (4× fewer pixels than native 1920×1080), then converts to
    a cached pygame.Surface. The surface is reused across frames via blit_array
    so no heap allocation occurs after the first call.

    Args:
        display_width: Target pixel width for all rendered panels.
    """

    def __init__(self, display_width: int = 960) -> None:
        self._display_width = display_width
        self.show_corners = True
        self.show_mask = True
        self.show_blocks = True
        self.show_tracker = True
        self._raw_cache: pygame.Surface | None = None
        self._board_cache: pygame.Surface | None = None

    def raw_surface(
        self,
        frame: np.ndarray,
        person_mask: np.ndarray | None,
        cached_corners: np.ndarray | None,
        sam_busy: bool,
    ) -> pygame.Surface:
        """Render raw camera frame with overlays as a pygame Surface.

        Args:
            frame: BGR uint8 camera frame.
            person_mask: Binary H×W mask (1=person) or None.
            cached_corners: Board quad corners in camera space, or None.
            sam_busy: True while SAM inference is in flight.

        Returns:
            pygame.Surface at (display_width, proportional height), ready to blit.
        """
        target_h = int(frame.shape[0] * self._display_width / frame.shape[1])
        raw = cv2.resize(frame, (self._display_width, target_h))
        sx = self._display_width / frame.shape[1]
        sy = target_h / frame.shape[0]
        if self.show_mask and person_mask is not None:
            small_mask = cv2.resize(
                person_mask, (self._display_width, target_h),
                interpolation=cv2.INTER_NEAREST,
            )
            raw = _apply_mask_overlay(raw, small_mask)
        if self.show_corners:
            raw = _draw_corners(raw, cached_corners, sx, sy)
        cv2.circle(raw, (raw.shape[1] - 30, 30), 10,
                   (0, 165, 255) if sam_busy else (0, 255, 0), -1)
        self._raw_cache = _bgr_to_surface(raw, self._raw_cache)
        return self._raw_cache

    def board_surface(
        self,
        composite: np.ndarray,
        blocks: list[Block],
        notes: list[Note],
        layout_busy: bool,
    ) -> pygame.Surface:
        """Render board composite with overlays as a pygame Surface.

        Args:
            composite: BGR uint8 1920×1080 board composite.
            blocks: Detected text blocks for overlay.
            notes: Tracked notes for overlay.
            layout_busy: True while layout detector subprocess is in flight.

        Returns:
            pygame.Surface at (display_width, proportional height), ready to blit.
        """
        h, w = composite.shape[:2]
        target_h = int(h * self._display_width / w)
        board = cv2.resize(composite, (self._display_width, target_h))
        sx = self._display_width / w
        sy = target_h / h
        if self.show_blocks:
            _draw_blocks(board, blocks, sx, sy)
        if self.show_tracker:
            _draw_notes(board, notes, sx, sy)
        cv2.circle(board, (board.shape[1] - 30, 30), 10,
                   (0, 165, 255) if layout_busy else (0, 255, 0), -1)
        self._board_cache = _bgr_to_surface(board, self._board_cache)
        return self._board_cache

