"""pytest configuration and shared fixtures.

Place reusable fixtures here so they are available to all test files
without explicit imports. Fixture images (synthetic or real whiteboard
photos) live in tests/fixtures/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# src/ modules use bare imports (e.g. `from layout import Region`) because
# debug_view.py is run with cwd=src/. Add src/ to sys.path so those bare
# imports resolve when pytest runs from the project root.
_src = str(Path(__file__).parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)


@pytest.fixture()
def blank_board() -> np.ndarray:
    """Return a plain white 1280×720 BGR image simulating an empty board."""
    return np.full((720, 1280, 3), fill_value=255, dtype=np.uint8)


@pytest.fixture()
def synthetic_board_frame() -> tuple[np.ndarray, np.ndarray]:
    """Return a camera frame containing a perspectively-skewed whiteboard.

    The whiteboard is a large off-white quadrilateral on a dark background,
    with two dark ink lines drawn inside it to simulate writing.

    Returns:
        Tuple of (frame, corners) where *frame* is a 720×1280 BGR uint8
        image and *corners* is the ground-truth quad in TL, TR, BR, BL
        order as a ``(4, 2)`` float32 array.
    """
    frame = np.full((720, 1280, 3), 80, dtype=np.uint8)  # dark gray wall

    corners = np.array(
        [[150, 80], [1100, 50], [1130, 640], [120, 660]],
        dtype=np.float32,
    )
    cv2.fillPoly(frame, [corners.astype(np.int32)], (245, 245, 245))

    # Simulated ink lines well inside the board
    cv2.line(frame, (300, 200), (900, 200), (20, 20, 20), 3)
    cv2.line(frame, (300, 380), (750, 380), (20, 20, 20), 2)

    return frame, corners
