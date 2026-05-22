"""Visualization layer for the whiteboard pipeline.

Owns all OpenCV drawing logic and overlay toggle state. Pipeline code
in main.py stays free of display concerns.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from layout_service import Block
from registry import EntityState, SemanticEntity

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Color palettes (BGR)
# ---------------------------------------------------------------------------


_STATE_COLORS: dict[EntityState, tuple[int, int, int]] = {
    EntityState.STABILIZING: (0, 165, 255),
    EntityState.INFERRING: (0, 200, 255),
    EntityState.ACTIVE: (0, 230, 0),
    EntityState.ERASED: (0, 0, 220),
}

_CORNER_LABELS = ["TL", "TR", "BR", "BL"]

# ---------------------------------------------------------------------------
# Pure drawing functions (stateless)
# ---------------------------------------------------------------------------


def _apply_mask_overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = frame.copy()
    overlay[mask == 1] = (0, 0, 220)
    return cv2.addWeighted(frame, 0.65, overlay, 0.35, 0)


def _draw_corners(frame: np.ndarray, corners: np.ndarray | None) -> np.ndarray:
    if corners is None:
        return frame

    pts = corners.astype(np.int32)
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


def _draw_blocks(frame: np.ndarray, blocks: list[Block]) -> np.ndarray:
    overlay = frame.copy()
    for block in blocks:
        for line in block.lines:
            x1, y1, x2, y2 = line.bbox.astype(int)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), _ANCHOR_COLOR, -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), _ANCHOR_COLOR, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
    return frame


def _draw_entities(frame: np.ndarray, entities: list[SemanticEntity]) -> np.ndarray:
    overlay = frame.copy()
    for ent in entities:
        x1, y1, x2, y2 = ent.bbox
        color = _STATE_COLORS.get(ent.state, (255, 255, 255))

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        label = ent.state.value
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

    def __init__(self) -> None:
        self.show_corners = True
        self.show_mask = True
        self.show_blocks = True
        self.show_tracker = True

    def render_board(
        self,
        composite: np.ndarray,
        blocks: list[Block],
        entities: list[SemanticEntity],
        is_busy: bool,
    ) -> np.ndarray:
        """Draw block + entity overlays and Discovery busy dot. Returns the frame."""
        board = composite.copy()
        if self.show_blocks:
            board = _draw_blocks(board, blocks)
        if self.show_tracker:
            board = _draw_entities(
                board, [e for e in entities if e.state != EntityState.ERASED]
            )
        cv2.circle(
            board,
            (board.shape[1] - 30, 30),
            10,
            (0, 165, 255) if is_busy else (0, 255, 0),
            -1,
        )
        return board

    def render_raw(
        self,
        frame: np.ndarray,
        person_mask: np.ndarray,
        cached_corners: np.ndarray | None,
        is_busy: bool,
    ) -> np.ndarray:
        """Draw mask + corner overlays and SAM busy dot. Returns the frame."""
        raw = frame.copy()
        if self.show_mask:
            raw = _apply_mask_overlay(raw, person_mask)
        if self.show_corners:
            raw = _draw_corners(raw, cached_corners)
        cv2.circle(
            raw,
            (raw.shape[1] - 30, 30),
            10,
            (0, 165, 255) if is_busy else (0, 255, 0),
            -1,
        )
        return raw

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
