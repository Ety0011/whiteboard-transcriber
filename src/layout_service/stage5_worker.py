import threading
import time

import numpy as np

from .base import BaseLayoutDetector


class Stage5LayoutDiscovery(threading.Thread):
    def __init__(
        self,
        detector: BaseLayoutDetector,
        target_w: int,
        target_h: int,
        on_regions_discovered,
    ):
        super().__init__(daemon=True)
        self.detector = detector
        self.target_w = target_w
        self.target_h = target_h
        self.on_regions_discovered = on_regions_discovered

        # Mailbox Synchronization
        self.mailbox_lock = threading.Lock()
        self.frame_mailbox: np.ndarray | None = None
        self.new_frame_event = threading.Event()
        self.is_running = True
        self.is_busy = False

    def submit_frame(self, frame: np.ndarray) -> bool:
        if self.is_busy:
            return False

        with self.mailbox_lock:
            self.frame_mailbox = frame.copy()
        self.new_frame_event.set()
        return True

    def run(self):
        self.detector.load()

        while self.is_running:
            self.new_frame_event.wait()
            self.new_frame_event.clear()

            with self.mailbox_lock:
                if self.frame_mailbox is None:
                    continue
                local_frame = self.frame_mailbox
                self.frame_mailbox = None

            self.is_busy = True
            start_time = time.time()

            try:
                discovered_regions = self.detector.detect(local_frame)
                latency = time.time() - start_time
                self.on_regions_discovered(discovered_regions, latency)

            except Exception as e:
                print(f"[Stage 5 Thread Error]: {e}")
            finally:
                self.is_busy = False

    def stop(self):
        self.is_running = False
        self.new_frame_event.set()
