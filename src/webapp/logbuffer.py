"""
In-memory ring buffer of the server's own log, exposed over HTTP.

The container's logs are the natural place to look when something misbehaves,
but they are awkward to reach on a remote deployment -- and the interesting
lines (weight downloads, job failures, separation fallbacks) are exactly the
ones a user cannot see from the browser. This keeps the recent tail in memory
so `/api/logs` can serve it.

A bounded deque, not a file: this is for looking at the last few thousand lines,
not for archival. Per-job pipeline output is written to disk separately, since
that is the log that actually matters for a failed chart.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Iterable

DEFAULT_CAPACITY = 5000

# These are per-request or per-chunk chatter: uvicorn's access log fires on
# every poll of /api/jobs, and httpx logs a line per HTTP call during weight
# downloads. Left in, they push everything interesting out of a bounded buffer
# within seconds. Warnings and errors from them are still kept.
NOISY_LOGGERS = ("uvicorn.access", "httpx", "httpcore", "urllib3", "filelock")


_NOISY_ROOTS = {name.split(".")[0] for name in NOISY_LOGGERS}


class RingBufferHandler(logging.Handler):
    """Logging handler that keeps the most recent records in memory."""

    def __init__(self, capacity: int = DEFAULT_CAPACITY):
        super().__init__()
        self.records: deque[tuple[int, str]] = deque(maxlen=capacity)

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno < logging.WARNING and record.name.split(".")[0] in _NOISY_ROOTS:
            return
        try:
            self.records.append((record.levelno, self.format(record)))
        except Exception:  # pragma: no cover - never break the app to log
            self.handleError(record)

    def tail(self, limit: int = 500, min_level: int = logging.NOTSET) -> list[str]:
        """Most recent `limit` lines at or above `min_level`, oldest first."""
        if min_level <= logging.NOTSET:
            lines: Iterable[str] = (text for _, text in self.records)
        else:
            lines = (text for level, text in self.records if level >= min_level)
        collected = list(lines)
        return collected[-limit:] if limit > 0 else collected


buffer = RingBufferHandler()


def install(level: int = logging.INFO) -> RingBufferHandler:
    """Attach the buffer to the root logger. Safe to call more than once."""
    buffer.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    buffer.setLevel(level)
    root = logging.getLogger()
    if buffer not in root.handlers:
        root.addHandler(buffer)
    # Without this the root logger's own level can filter records before any
    # handler sees them.
    if root.level > level or root.level == logging.NOTSET:
        root.setLevel(level)
    return buffer
