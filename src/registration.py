"""Stage 1 — Spatial Registration.

Detects the whiteboard quadrilateral using SAM 3 (Segment Anything Model 3)
with a text prompt ("whiteboard"), extracts the board boundary from the
resulting binary mask, and warps every frame to a canonical 1280×720 output.

Between SAM 3 re-detections, a SAM 2.1 tiny model runs a lightweight
movement check every ``check_every`` frames: it segments the centre of the
frame with a point prompt and computes mask IoU against the stored SAM 3
reference. If IoU drops below ``iou_threshold`` the board has moved and
SAM 3 is re-triggered immediately. ``recompute_every`` provides an absolute
periodic fallback.

Models:
  ``sam3-l.pt``   — whiteboard detection (auto-downloaded by Ultralytics).
  ``sam2.1_t.pt`` — movement check proxy  (auto-downloaded by Ultralytics).

Typical usage::

    registrar = Registrar()
    warped = registrar.process(frame)
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import cv2
import numpy as np
from ultralytics import SAM
from ultralytics.models.sam import SAM3SemanticPredictor

logger = logging.getLogger(__name__)

OUTPUT_SIZE: tuple[int, int] = (1280, 720)


class Registrar:
    """Stateful perspective-correction stage backed by SAM 3 board detection.

    SAM 3 is re-triggered when the SAM 2.1 movement checker reports IoU below
    ``iou_threshold``, or unconditionally every ``recompute_every`` pipeline
    cycles.  Between runs the cached homography is reused, keeping per-frame
    cost minimal.
    """

    def __init__(
        self,
        output_size: tuple[int, int] = OUTPUT_SIZE,
        cache_threshold: float = 20.0,
        recompute_every: int = 300,
        check_every: int = 10,
        debug: bool = False,
        iou_threshold: float = 0.80,
        sam_model: str = "sam3.1_multiplex.pt",
        tracker_model: str = "sam2.1_t.pt",
    ) -> None:
        """
        Args:
            output_size: (width, height) of the warped output image.
            cache_threshold: Maximum corner displacement in pixels before the
                homography is recomputed even within a recompute cycle.
            recompute_every: Absolute SAM 3 re-detection interval in frames.
                Acts as a fallback; SAM 2.1 movement detection usually fires first.
            check_every: Run the SAM 2.1 movement check every this many frames.
            debug: When True, draw the detected quad and corners onto the
                raw input frame (shown in the debug window of debug_view.py).
            iou_threshold: Mask IoU below which the board is considered moved
                and SAM 3 re-detection is triggered.
            sam_model: Ultralytics model filename for SAM 3 detection.
            tracker_model: Ultralytics model filename for SAM 2.1 movement check.
        """
        self._output_size = output_size
        self._cache_threshold = cache_threshold
        self._recompute_every = recompute_every
        self._check_every = check_every
        self.debug = debug
        self._iou_threshold = iou_threshold

        _model = Path(__file__).parent.parent / "models" / sam_model
        self._sam: SAM3SemanticPredictor = SAM3SemanticPredictor(
            overrides=dict(
                model=str(_model),
                task="segment",
                mode="predict",
                imgsz=644,   # nearest multiple of SAM 3's stride-14 above default 640
                save=False,
                verbose=False,
            )
        )

        _tracker = Path(__file__).parent.parent / "models" / tracker_model
        self._sam_tracker = SAM(str(_tracker))

        # Start counters at their limits so the very first process() call
        # fires SAM 3 immediately.
        self._calls_since_detect: int = recompute_every
        self._calls_since_check: int = check_every

        self._homography: np.ndarray | None = None
        self._cached_corners: np.ndarray | None = None  # (4, 2) float32, TL/TR/BR/BL
        self._ref_mask: np.ndarray | None = None  # reference mask for IoU checks
        self._last_mask: np.ndarray | None = None  # most recent SAM 3 mask
        self._lock = threading.Lock()  # guards _homography and _cached_corners

        self._detecting: bool = False
        self._pending_frame: np.ndarray | None = None
        self._pending_mode: str = "detect"  # "detect" | "check"
        self._detect_event = threading.Event()
        self._detect_thread = threading.Thread(
            target=self._detection_loop,
            daemon=True,
            name="sam3-detect",
        )
        self._detect_thread.start()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Warp *frame* to remove perspective distortion.

        SAM 3 detection and SAM 2.1 movement checks are fired asynchronously
        on a background thread; this method never blocks on inference.

        Args:
            frame: BGR uint8 image captured from the camera.

        Returns:
            Perspective-corrected BGR uint8 image at ``output_size``.
            Falls back to a centred resize if no board has ever been detected.
        """
        self._calls_since_detect += 1
        self._calls_since_check += 1

        if not self._detecting:
            if self._calls_since_detect >= self._recompute_every:
                # Absolute SAM 3 periodic fallback
                self._calls_since_detect = 0
                self._calls_since_check = 0
                self._pending_mode = "detect"
                self._pending_frame = frame
                self._detecting = True
                self._detect_event.set()
            elif (
                self._ref_mask is not None
                and self._calls_since_check >= self._check_every
            ):
                # SAM 2.1 movement check (only after SAM 3 has run once)
                self._calls_since_check = 0
                self._pending_mode = "check"
                self._pending_frame = frame
                self._detecting = True
                self._detect_event.set()

        with self._lock:
            homography = self._homography
            cached_corners = self._cached_corners

        if self.debug and cached_corners is not None:
            frame = self._draw_debug(frame.copy(), cached_corners)

        if homography is None:
            logger.debug("No board detected yet — returning resized frame")
            return cv2.resize(frame, self._output_size)

        return cv2.warpPerspective(frame, homography, self._output_size)

    # ------------------------------------------------------------------
    # Background detection thread
    # ------------------------------------------------------------------

    def _detection_loop(self) -> None:
        """Daemon thread: handles SAM 2.1 movement checks and SAM 3 re-detections."""
        while True:
            self._detect_event.wait()
            self._detect_event.clear()

            frame = self._pending_frame
            mode = self._pending_mode
            if frame is None:
                self._detecting = False
                continue

            try:
                if mode == "check":
                    if self._has_board_moved(frame):
                        logger.debug("Board movement detected — running SAM 3")
                        mode = "detect"  # fall through to SAM 3

                if mode == "detect":
                    corners = self._detect_corners(frame)
                    if corners is not None:
                        sorted_corners = self._sort_corners(corners)
                        updated = False
                        with self._lock:
                            if self._cached_corners is None or self._corners_shifted(
                                sorted_corners
                            ):
                                self._homography = self._compute_homography(
                                    sorted_corners
                                )
                                self._cached_corners = sorted_corners
                                updated = True
                                logger.debug(
                                    "Homography updated — corners: %s", sorted_corners
                                )
                        if updated:
                            # Update IoU reference outside the lock (avoids holding
                            # it during an array copy).
                            self._ref_mask = self._last_mask
            except Exception:
                logger.exception("SAM detection failed (mode=%s)", mode)
            finally:
                self._detecting = False

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _detect_corners(self, frame: np.ndarray) -> np.ndarray | None:
        """Run SAM 3 with a text prompt and return the board quad.

        Uses SAM3SemanticPredictor with the concept prompt "whiteboard" so
        detection is location-independent — no point prompts needed.

        Stores the best segmentation mask in ``self._last_mask`` so the
        detection loop can promote it to ``_ref_mask`` after a successful
        homography update.

        Args:
            frame: BGR uint8 camera frame.

        Returns:
            Corner array of shape ``(4, 2)`` as float32, or ``None``.
        """
        results = self._sam(frame, text=["whiteboard"])

        if not results or results[0].masks is None:
            logger.debug("SAM 3 returned no masks for 'whiteboard'")
            return None

        masks = results[0].masks.data.cpu().numpy()
        if masks.shape[0] == 0:
            logger.debug("SAM 3 returned empty mask tensor")
            return None

        areas = masks.sum(axis=(1, 2))
        best = (masks[areas.argmax()] > 0.5).astype(np.uint8)
        self._last_mask = best
        return self._mask_to_corners(best)

    def _has_board_moved(self, frame: np.ndarray) -> bool:
        """Run SAM 2.1 with a centre-point prompt and compare mask IoU to reference.

        Args:
            frame: Current BGR uint8 camera frame.

        Returns:
            True if IoU < iou_threshold or SAM 2.1 returns no mask (conservative).
        """
        h, w = frame.shape[:2]
        results = self._sam_tracker(
            frame, points=[[w // 2, h // 2]], labels=[1], verbose=False
        )

        if not results or results[0].masks is None:
            return True  # can't confirm position — trigger re-detection

        masks = results[0].masks.data.cpu().numpy()
        if masks.shape[0] == 0:
            return True

        areas = masks.sum(axis=(1, 2))
        current_mask = (masks[areas.argmax()] > 0.5).astype(np.uint8)

        ref = self._ref_mask
        if current_mask.shape != ref.shape:
            current_mask = cv2.resize(
                current_mask,
                (ref.shape[1], ref.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.uint8)

        intersection = np.logical_and(current_mask, ref).sum()
        union = np.logical_or(current_mask, ref).sum()
        if union == 0:
            return True
        iou = float(intersection) / float(union)
        logger.debug("Board IoU: %.3f (threshold %.2f)", iou, self._iou_threshold)
        return iou < self._iou_threshold

    @staticmethod
    def _mask_to_corners(mask: np.ndarray) -> np.ndarray | None:
        """Extract a 4-corner quadrilateral from a binary segmentation mask.

        Takes the convex hull of the largest contour in *mask*, then sweeps
        the ``approxPolyDP`` epsilon until the polygon collapses to 4 vertices.

        Args:
            mask: Binary uint8 mask, shape ``(H, W)``, values 0 or 1.

        Returns:
            Corner array of shape ``(4, 2)`` as float32, or ``None`` if no
            suitable quadrilateral can be found.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        hull = cv2.convexHull(max(contours, key=cv2.contourArea))
        peri = cv2.arcLength(hull, True)

        for eps in (0.02, 0.04, 0.06, 0.08, 0.10):
            approx = cv2.approxPolyDP(hull, eps * peri, True)
            if len(approx) == 4:
                return approx.reshape(4, 2).astype(np.float32)

        return None

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sort_corners(pts: np.ndarray) -> np.ndarray:
        """Return corners ordered as TL, TR, BR, BL."""
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        d = np.diff(pts, axis=1).ravel()
        rect[1] = pts[np.argmin(d)]
        rect[3] = pts[np.argmax(d)]
        return rect

    def _compute_homography(self, corners: np.ndarray) -> np.ndarray:
        """Compute perspective transform from *corners* to the output rectangle."""
        w, h = self._output_size
        dst = np.array(
            [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
            dtype=np.float32,
        )
        return cv2.getPerspectiveTransform(corners, dst)

    def _corners_shifted(self, sorted_corners: np.ndarray) -> bool:
        """Return True if any corner moved more than ``cache_threshold`` pixels."""
        if self._cached_corners is None:
            return True
        dists = np.linalg.norm(sorted_corners - self._cached_corners, axis=1)
        return bool(np.any(dists > self._cache_threshold))

    # ------------------------------------------------------------------
    # Debug overlay
    # ------------------------------------------------------------------

    def _draw_debug(self, frame: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """Draw corner markers and quad outline on *frame*."""
        pts = corners.astype(np.int32)
        cv2.polylines(
            frame,
            [pts.reshape(-1, 1, 2)],
            isClosed=True,
            color=(0, 0, 220),
            thickness=2,
        )
        for i, (x, y) in enumerate(pts):
            cv2.circle(frame, (int(x), int(y)), 8, (0, 200, 0), -1)
            cv2.putText(
                frame,
                str(i),
                (int(x) + 10, int(y) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
        return frame


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_global_registrar: Registrar | None = None


def process(frame: np.ndarray) -> np.ndarray:
    """Warp *frame* using a module-global :class:`Registrar`."""
    global _global_registrar
    if _global_registrar is None:
        _global_registrar = Registrar()
    return _global_registrar.process(frame)
