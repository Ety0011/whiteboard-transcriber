"""FrameSource — abstract base class for all video/canvas frame sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

import numpy as np


class FrameSource(ABC):
    """Abstract base for video and canvas frame sources.

    Core contract: non-blocking frame delivery with start/stop lifecycle and
    optional pause/resume. Drawing interaction (mouse, clear) is not part of
    this interface — it belongs exclusively to CanvasCapture.

    Two concrete implementations:
    - :class:`~capture.Capture` — webcam, video file, or static image.
    - :class:`~capture.CanvasCapture` — mouse-drawable canvas (demo mode).
    """

    fps: float | None = None
    """Source frames-per-second from metadata. None for live cameras."""

    frame_size: tuple[int, int] | None = None
    """``(width, height)`` pixel dimensions. None if unknown before start()."""

    @abstractmethod
    def start(self) -> Self:
        """Open the source and begin producing frames.

        Returns:
            self, for fluent chaining (``cap = Capture(src).start()``).

        Raises:
            RuntimeError: If the source cannot be opened.
        """
        ...

    @abstractmethod
    def try_read(self) -> np.ndarray | None:
        """Non-blocking frame read.

        Returns:
            BGR uint8 frame, or None when no frame is available this tick
            (camera not yet buffered, paused, or end-of-stream). Never raises.
        """
        ...

    @abstractmethod
    def stop(self) -> None:
        """Release resources and unblock any pending reads."""
        ...

    @property
    def is_active(self) -> bool:
        """True while the source is open and has not reached end-of-stream.

        Default: True. Override in sources that can reach a natural end (e.g.
        video files) to allow callers to distinguish EOS from a transient empty
        try_read() tick.
        """
        return True

    def pause(self) -> None:
        """Freeze playback. Default: no-op."""

    def resume(self) -> None:
        """Unfreeze playback. Default: no-op."""

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.stop()
