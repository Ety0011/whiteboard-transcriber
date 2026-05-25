"""Third-party logging noise suppression.

Call suppress_noise() once in main() before any model is loaded.
Call suppress_worker_noise() at the top of each subprocess worker.
Use devnull_fds() around specific init calls that bypass Python logging.
"""

from __future__ import annotations

import contextlib
import os
import warnings
from collections.abc import Iterator

_NOISY_LOGGERS = (
    "httpx",
    "ultralytics",
    "paddleocr",
    "ppocr",
    "paddle",
    "paddlex",
    "PIL",
    "transformers",
    "huggingface_hub",
    "mlx_vlm",
)


@contextlib.contextmanager
def devnull_fds(*fds: int) -> Iterator[None]:
    """Redirect file descriptors to /dev/null for the duration of the block.

    Use around C++ library init calls that write directly to fd 1 or fd 2
    regardless of Python logging configuration (e.g. glog, TFLite, ultralytics).
    """
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = [os.dup(fd) for fd in fds]
    for fd in fds:
        os.dup2(devnull, fd)
    os.close(devnull)
    try:
        yield
    finally:
        for fd, sv in zip(fds, saved):
            os.dup2(sv, fd)
            os.close(sv)


def suppress_noise() -> None:
    """Silence third-party noise. Call once in main() before any subprocess is spawned.

    Sets env vars (inherited by all worker subprocesses), filters Python warnings,
    and sets noisy logger levels to WARNING.
    """
    import logging

    # C++ glog / absl (MediaPipe): 0=INFO 1=WARN 2=ERROR 3=FATAL
    # Direct assignment — setdefault would be defeated by an inherited env var.
    # stderrthreshold=4: nothing reaches stderr even if minloglevel is ignored
    # (MediaPipe's clearcut uploader background thread fires ERROR retries that
    # survive minloglevel=3 alone on newer absl builds).
    os.environ["GLOG_minloglevel"] = "3"
    os.environ["GLOG_logtostderr"] = "0"
    os.environ["GLOG_stderrthreshold"] = "4"
    # TFLite delegate messages
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    # HuggingFace Hub tqdm download bars
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    # Ultralytics startup banner (read at import time)
    os.environ["YOLO_VERBOSE"] = "false"

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def suppress_worker_noise() -> None:
    """Silence noisy loggers in a subprocess worker.

    Env vars are inherited automatically; this only handles Python-level loggers
    and warning filters which are not inherited across process boundaries.
    """
    import logging

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
