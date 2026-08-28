"""
Releasing GPU memory between pipeline stages.

The pipeline runs a sequence of large models in one process -- a karaoke
RoFormer, one or two Demucs models, then the drum and guitar networks. None of
them need to be resident at the same time, but Python will happily keep every
one of them alive until the last reference goes, and PyTorch keeps the freed
blocks in its caching allocator rather than returning them to the driver.

On a 6 GB card that is the difference between working and a CUDA OOM: the
separation stages alone were sitting at 5.8 GB before anything else loaded.
Calling `free_gpu()` after each stage keeps the peak at roughly the largest
single model instead of their sum.
"""

from __future__ import annotations

import gc
import logging

logger = logging.getLogger(__name__)


def vram_used_gb() -> float | None:
    """VRAM currently in use on device 0, or None without CUDA."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()
        return (total - free) / 1e9
    except Exception:
        return None


def free_gpu(label: str = "") -> None:
    """Drop unreferenced tensors and return cached blocks to the driver.

    Call after a stage is finished with its model. `gc.collect()` matters here:
    the models tend to be held in reference cycles, so without it the memory is
    not released until a later collection happens to run.
    """
    before = vram_used_gb()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception as e:  # pragma: no cover - never fail a run over cleanup
        logger.debug(f"GPU cleanup skipped: {e}")
        return

    after = vram_used_gb()
    if before is not None and after is not None:
        freed = before - after
        where = f" after {label}" if label else ""
        logger.info(f"  VRAM{where}: {after:.2f} GB in use (released {freed:.2f} GB)")


def rss_gb() -> float | None:
    """Resident set size of this process, or None if unavailable."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024 / 1e9
    except OSError:
        pass
    return None


def free_tensorflow() -> None:
    """Drop TensorFlow's graph and its allocations.

    basic-pitch runs under TensorFlow, and its memory is not returned when the
    call finishes. Charting guitar and then bass therefore pays for the model
    twice, on top of everything PyTorch is already holding -- which is what gets
    the process killed on a 16 GB host.

    Imported lazily and guarded: TensorFlow is an optional dependency, and this
    must never be the thing that fails a run.
    """
    try:
        import sys

        if "tensorflow" not in sys.modules:
            return
        import tensorflow as tf

        tf.keras.backend.clear_session()
    except Exception as e:  # pragma: no cover
        logger.debug(f"TensorFlow cleanup skipped: {e}")


def free_memory(label: str = "") -> None:
    """Release host and device memory between pipeline stages."""
    before = rss_gb()
    free_tensorflow()
    free_gpu(label)
    after = rss_gb()
    if before is not None and after is not None:
        where = f" after {label}" if label else ""
        logger.info(f"  RAM{where}: {after:.2f} GB resident (released {before - after:.2f} GB)")
