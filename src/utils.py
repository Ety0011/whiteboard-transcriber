"""Shared utilities: logging configuration, constants, and hash table.

Provides:
    - ``configure_logging()``  — sets up the root logger with a consistent format.
    - ``HashTable``            — thin wrapper around a set for perceptual-hash
                                 deduplication across pipeline cycles.
    - ``BOARD_RESOLUTION``     — canonical (width, height) tuple used by Stage 1.
    - ``MIN_REGION_AREA``      — minimum changed-region area in pixels (Stage 4).
    - ``OCR_CONFIDENCE_GATE``  — EasyOCR confidence below which TrOCR is invoked.
"""

from __future__ import annotations


def process() -> None:
    """Placeholder — utils has no single process() entry point.

    Call the individual utilities (configure_logging, HashTable, etc.)
    directly from the modules that need them.
    """
    raise NotImplementedError
