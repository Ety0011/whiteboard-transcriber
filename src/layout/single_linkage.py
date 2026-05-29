"""Obstacle-vetoed agglomerative clustering for text layout grouping.

Merges cluster pairs in ascending order of nearest-point Euclidean
distance. A merge is vetoed when the union bbox would newly
enclose a third cluster that was not already touching either constituent,
indicating an obstacle sits between them. A distance cap (max_gap_px)
provides the stopping criterion — pairs beyond the threshold are never
considered regardless of obstacle state.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np

from .block import Block
from .text_detector import TextLine


@dataclass
class _Cluster:
    """Mutable working cluster used during agglomeration.

    Attributes:
        id: Unique integer identifier, monotonically increasing across merges.
        bbox: Tight axis-aligned bounding box in rectified 1920×1080 space,
            shape (4,) int32: x1, y1, x2, y2.
        lines: All TextLine objects absorbed into this cluster so far.
    """

    id: int
    bbox: np.ndarray
    lines: list[TextLine] = field(default_factory=list)


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    """Nearest-point Euclidean distance between two axis-aligned bboxes.

    Args:
        a: Bbox (x1, y1, x2, y2) of the first rectangle.
        b: Bbox (x1, y1, x2, y2) of the second rectangle.

    Returns:
        Euclidean distance between nearest points. Zero when boxes overlap or touch.
    """
    dx = float(max(0, max(a[0], b[0]) - min(a[2], b[2])))
    dy = float(max(0, max(a[1], b[1]) - min(a[3], b[3])))
    return math.sqrt(dx * dx + dy * dy)


def _union(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the smallest axis-aligned bbox that contains both *a* and *b*.

    Args:
        a: Bbox (x1, y1, x2, y2) of the first rectangle.
        b: Bbox (x1, y1, x2, y2) of the second rectangle.

    Returns:
        Shape (4,) int32 array: x1, y1, x2, y2.
    """
    return np.array(
        [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])],
        dtype=np.int32,
    )


def _intersects(a: np.ndarray, b: np.ndarray) -> bool:
    """Return True when two axis-aligned bboxes overlap (touching edges count).

    Args:
        a: Bbox (x1, y1, x2, y2) of the first rectangle.
        b: Bbox (x1, y1, x2, y2) of the second rectangle.

    Returns:
        True if the interiors overlap; False if they are disjoint or only share
        an edge.
    """
    return bool(a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1])


class SingleLinkageClusterer:
    """Agglomerative clustering with obstacle veto, distance cap, and hysteresis.

    Merges the closest cluster pair whose union bbox does not newly enclose
    any third cluster and whose nearest-point distance is within max_gap_px.
    Repeats until no valid merge remains.

    Hysteresis: pairs whose bboxes both overlapped a single block in the
    previous call are granted an extended merge threshold of
    ``max_gap_px + hysteresis_px``. This prevents multi-pixel detection
    noise from oscillating co-block lines between merged and split states
    across frames.
    """

    def __init__(
        self, max_gap_px: float = 20.0, hysteresis_px: float = 10.0
    ) -> None:
        """Configure the clusterer.

        Args:
            max_gap_px: Maximum nearest-point pixel distance between two
                clusters for a merge to be attempted. Pairs beyond this
                threshold are never pushed onto the heap, acting as a hard
                stopping criterion.
            hysteresis_px: Extra distance allowance granted to cluster pairs
                that were co-block in the previous call. Prevents multi-pixel
                detection noise from splitting stable groups.
        """
        self._max_gap_px = max_gap_px
        self._hysteresis_px = hysteresis_px
        self._prev_blocks: list[np.ndarray] = []

    def _merge_threshold(self, a: np.ndarray, b: np.ndarray) -> float:
        """Return the effective merge distance threshold for two bboxes.

        Grants hysteresis_px of extra distance when both bboxes overlap a
        common block from the previous call, indicating they were co-block
        and should resist splitting due to detection noise.

        Args:
            a: Bbox (x1, y1, x2, y2) of the first candidate.
            b: Bbox (x1, y1, x2, y2) of the second candidate.

        Returns:
            max_gap_px + hysteresis_px if co-block last frame, else max_gap_px.
        """
        if self._hysteresis_px > 0.0 and any(
            _intersects(prev, a) and _intersects(prev, b)
            for prev in self._prev_blocks
        ):
            return self._max_gap_px + self._hysteresis_px
        return self._max_gap_px

    @staticmethod
    def _compute_bbox(lines: list[TextLine]) -> np.ndarray:
        """Return the tight axis-aligned bbox enclosing all *lines*."""
        boxes = np.array([line.bbox for line in lines])
        return np.array(
            [
                np.min(boxes[:, 0]),
                np.min(boxes[:, 1]),
                np.max(boxes[:, 2]),
                np.max(boxes[:, 3]),
            ],
            dtype=np.int32,
        )

    def _make_block(self, lines: list[TextLine]) -> Block:
        """Construct a Block from *lines* — tight bbox and max confidence."""
        return Block(
            bbox=self._compute_bbox(lines),
            confidence=max(l.confidence for l in lines),
            lines=lines,
        )

    def cluster(self, lines: list[TextLine]) -> list[Block]:
        """Cluster text lines into blocks via obstacle-vetoed agglomeration.

        Args:
            lines: Text lines from Stage 6 detection.

        Returns:
            List of Blocks, each enclosing one or more spatially coherent lines.
        """
        if not lines:
            self._prev_blocks = []
            return []

        clusters: dict[int, _Cluster] = {
            i: _Cluster(id=i, bbox=line.bbox.copy(), lines=[line])
            for i, line in enumerate(lines)
        }
        next_id = len(lines)
        active: set[int] = set(clusters)

        # Min-heap of (distance, id_a, id_b). Stale entries (merged ids) are
        # skipped lazily when popped.
        heap: list[tuple[float, int, int]] = []
        ids = list(active)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a_bbox = clusters[ids[i]].bbox
                b_bbox = clusters[ids[j]].bbox
                d = _distance(a_bbox, b_bbox)
                if d <= self._merge_threshold(a_bbox, b_bbox):
                    heapq.heappush(heap, (d, ids[i], ids[j]))

        while heap:
            dist, id_a, id_b = heapq.heappop(heap)

            if id_a not in active or id_b not in active:
                continue  # stale entry

            a = clusters[id_a]
            b = clusters[id_b]
            m_bbox = _union(a.bbox, b.bbox)

            # Veto: C newly enclosed by M means C didn't touch A or B before.
            vetoed = any(
                _intersects(m_bbox, clusters[c].bbox)
                and not _intersects(a.bbox, clusters[c].bbox)
                and not _intersects(b.bbox, clusters[c].bbox)
                for c in active
                if c != id_a and c != id_b
            )
            if vetoed:
                continue

            merged = _Cluster(
                id=next_id,
                bbox=m_bbox,
                lines=a.lines + b.lines,
            )
            clusters[next_id] = merged
            active.discard(id_a)
            active.discard(id_b)
            active.add(next_id)

            for c in active:
                if c == next_id:
                    continue
                c_bbox = clusters[c].bbox
                d = _distance(m_bbox, c_bbox)
                if d <= self._merge_threshold(m_bbox, c_bbox):
                    heapq.heappush(heap, (d, next_id, c))

            next_id += 1

        self._prev_blocks = [clusters[cid].bbox for cid in active]
        return [self._make_block(clusters[cid].lines) for cid in active]
