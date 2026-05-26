"""Base classes for the two pipeline stage patterns.

Two execution patterns are used throughout the pipeline:

- InlineStage: runs synchronously in the main loop. Subclasses use _due() to
  self-throttle — returning a cached result when the interval has not elapsed.
  Suitable for fast transforms (<5ms) where subprocess IPC overhead would cost
  more than the operation itself.

- WorkerStage: runs in a dedicated subprocess. The main loop calls non-blocking
  each tick and receives the most recent cached result. Suitable for heavy model
  inference (>50ms) where true parallelism justifies the subprocess cost.
  Owns all multiprocessing boilerplate — subclasses only implement load() and
  _process_item().
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
from abc import ABC, abstractmethod
from typing import Any, Callable


def _run_worker(stage: WorkerStage) -> None:
    """Subprocess entry point — unpickled from parent, sets up logging, runs loop.

    This must be a module-level function to be picklable under the spawn start
    method used on macOS.
    """
    import os

    from logging_config import suppress_worker_noise

    _level = logging.DEBUG if os.getenv("LOG_LEVEL") == "DEBUG" else logging.INFO
    logging.basicConfig(level=_level, format="%(levelname)s %(name)s: %(message)s")
    suppress_worker_noise()
    stage.load()
    stage._run_loop()


class InlineStage(ABC):
    """Stage that runs synchronously in the main loop, self-throttled.

    Args:
        interval_s: Minimum seconds between runs. 0.0 means always run (no throttle).
    """

    def __init__(self, interval_s: float = 0.0) -> None:
        self._interval_s = interval_s
        self._last_t: float = 0.0

    def _due(self) -> bool:
        """Return True if interval_s seconds have elapsed since last run."""
        if self._interval_s <= 0.0:
            return True
        now = time.monotonic()
        if now - self._last_t >= self._interval_s:
            self._last_t = now
            return True
        return False

    @property
    def _log(self) -> logging.Logger:
        """Logger keyed to the concrete subclass name."""
        return logging.getLogger(type(self).__name__)

    def shutdown(self) -> None:
        """Optional cleanup hook called when the pipeline stops."""


class WorkerStage(ABC):
    """Stage running in a dedicated subprocess.

    All multiprocessing boilerplate lives here. Subclasses only need to:
      1. Set class-level config attributes (optional overrides below).
      2. Implement _process_item() to process one input item.
      3. Expose a public API method (e.g. detect(), segment()) that calls
         _submit() and _poll() as appropriate.

    Factory pattern (optional — covers most workers):
      Set self._factory to a zero-argument callable in __init__. The default
      load() will call it, load the model, and store the result in self._model.
      The default _on_shutdown() will call self._model.shutdown() if set.
      Override load() only when custom loading logic is required.

    Class-level config (override in subclass):
        _process_name (str):       subprocess name shown in ps
        _in_queue_size (int):      maxsize for the input queue
        _out_queue_size (int):     maxsize for the output queue
        _drop_old (bool):          drain output before put (latest-result pattern)
        _shutdown_timeout (float): join timeout in seconds before terminate()
    """

    _process_name: str = "worker"
    _in_queue_size: int = 1
    _out_queue_size: int = 1
    _drop_old: bool = True
    _shutdown_timeout: float = 5.0
    _daemon: bool = False

    def __init__(self) -> None:
        self._model: Any = None
        self._in_q: mp.Queue = mp.Queue(maxsize=self._in_queue_size)
        self._out_q: mp.Queue = mp.Queue(maxsize=self._out_queue_size)
        self._is_busy: bool = False
        self._last_submit: float = 0.0
        self._proc = mp.Process(
            target=_run_worker,
            args=(self,),
            daemon=self._daemon,
            name=self._process_name,
        )
        self._proc.start()
        self._log.info("worker started (pid=%d)", self._proc.pid)

    def _load_from_factory(self, factory: Callable[[], Any]) -> Any:
        """Instantiate and load a model from *factory*; log readiness. Runs in subprocess.

        Args:
            factory: Zero-argument callable returning an object with a load()
                method.

        Returns:
            The loaded model instance.
        """
        model = factory()
        model.load()
        self._log.info("%s ready", type(model).__name__)
        return model

    def load(self) -> None:
        """Load model inside the subprocess. Uses factory pattern if _factory is set."""
        factory = getattr(self, "_factory", None)
        if factory is not None:
            self._model = self._load_from_factory(factory)

    @abstractmethod
    def _process_item(self, item: Any) -> Any:
        """Process one input item. Runs exclusively inside the worker subprocess."""
        ...

    def _on_shutdown(self) -> None:
        """Shut down loaded model if present. Override for custom teardown."""
        if self._model is not None:
            self._model.shutdown()

    def _run_loop(self) -> None:
        """Main worker loop — blocks on input queue, processes items until sentinel."""
        while True:
            item = self._in_q.get()
            if item is None:
                self._on_shutdown()
                break
            try:
                result = self._process_item(item)
            except Exception:
                self._log.exception("process failed")
                continue
            if self._drop_old:
                try:
                    self._out_q.get_nowait()
                except Exception:
                    pass
            self._put_result(result)

    def _put_result(self, result: Any) -> None:
        """Put result on output queue, warns if full."""
        try:
            self._out_q.put_nowait(result)
        except Exception:
            self._log.warning("output queue full — result dropped")

    def _submit(self, item: Any) -> None:
        """Submit an item to the worker queue — non-blocking, sets is_busy."""
        try:
            self._in_q.put_nowait(item)
            self._is_busy = True
        except Exception:
            self._log.warning("input queue full — item dropped")

    def _submit_if_due(self, item: Any, interval_s: float) -> bool:
        """Submit *item* if *interval_s* has elapsed since last submission.

        Returns:
            True if submitted; False if the interval has not yet elapsed.
        """
        now = time.monotonic()
        if now - self._last_submit >= interval_s:
            self._submit(item)
            self._last_submit = now
            return True
        return False

    def _poll(self) -> Any | None:
        """Poll for the latest result — non-blocking, clears is_busy on success."""
        try:
            result = self._out_q.get_nowait()
            self._is_busy = False
            return result
        except Exception:
            return None

    @property
    def _log(self) -> logging.Logger:
        """Logger keyed to the concrete subclass name. Computed on demand so it
        works identically in the main process and inside the worker subprocess."""
        return logging.getLogger(type(self).__name__)

    @property
    def is_busy(self) -> bool:
        """True while the subprocess has unprocessed work in flight."""
        return self._is_busy

    def shutdown(self) -> None:
        """Send shutdown sentinel, join subprocess, terminate if timeout exceeded."""
        try:
            self._in_q.put_nowait(None)
        except Exception:
            pass
        self._proc.join(timeout=self._shutdown_timeout)
        if self._proc.is_alive():
            self._proc.terminate()
        self._log.info("worker stopped")

    def __getstate__(self) -> dict:
        """Exclude _proc from subprocess pickling — not meaningful inside the child."""
        state = self.__dict__.copy()
        state.pop("_proc", None)
        return state
