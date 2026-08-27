"""
Charting weights: check the volume at startup, download whatever is missing.

Weights are deliberately not baked into the image. They are 1.8 GB, licensed
separately, and worth keeping across rebuilds, so they live on a mounted volume
that starts empty. Filling it is therefore a first-run job, not a build step.

This runs inside the app rather than in a container entrypoint for three
reasons: it works the same when `strum-web` is run directly, the server binds
immediately so the UI is reachable while the download proceeds, and progress is
visible in the browser instead of only in container logs.

Jobs wait for this to finish. A job queued during the download sits in the queue
rather than failing against weights that are not there yet.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

CHECKPOINT_DIR = Path(os.environ.get("STRUM_CHECKPOINT_DIR") or (REPO_ROOT / "checkpoints"))


def _enabled() -> bool:
    raw = os.environ.get("STRUM_FETCH_CHECKPOINTS", "1").strip().lower()
    return raw not in ("0", "false", "no", "off", "")


class Weights:
    """Tracks whether the charting weights are on disk, and fetches them once."""

    def __init__(self, dest: Path = CHECKPOINT_DIR):
        self.dest = Path(dest)
        self.state = "unknown"      # unknown | downloading | ready | failed | disabled
        self.done = 0
        self.total = 0
        self.error = ""
        self._ready = asyncio.Event()
        self._lock = threading.Lock()

    # -- reporting ---------------------------------------------------------

    def public(self) -> dict:
        return {
            "state": self.state,
            "done": self.done,
            "total": self.total,
            "error": self.error,
            "path": str(self.dest),
        }

    @property
    def ready(self) -> bool:
        return self.state in ("ready", "disabled")

    async def wait_ready(self) -> bool:
        """Block until the fetch settles. False means weights are unusable."""
        await self._ready.wait()
        return self.state in ("ready", "disabled")

    def _finish(self, state: str, error: str = "") -> None:
        self.state, self.error = state, error
        self._ready.set()

    # -- the fetch ---------------------------------------------------------

    async def ensure(self) -> None:
        """Fill the checkpoint directory if anything is missing.

        Safe to call once at startup. Never raises: a failure is recorded in
        `state` so the UI can explain it, because a server that refuses to start
        is far harder to diagnose than one that says what is wrong.
        """
        try:
            from fetch_checkpoints import fetch_plan, missing, resolve_plan
        except ImportError as e:
            self._finish("failed", f"weight downloader unavailable: {e}")
            return

        plan = resolve_plan(self.dest)
        self.total = len(plan)
        absent = missing(plan)

        if not absent:
            logger.info(f"Weights present in {self.dest} ({self.total} files)")
            self.done = self.total
            self._finish("ready")
            return

        if not _enabled():
            self._finish(
                "disabled",
                f"{len(absent)} weight file(s) missing and STRUM_FETCH_CHECKPOINTS is off",
            )
            logger.warning(self.error)
            return

        self.state = "downloading"
        self.done = self.total - len(absent)
        logger.info(
            f"Fetching {len(absent)} of {self.total} weight file(s) into {self.dest}. "
            "The UI is usable now; charting waits for this to finish."
        )

        def progress(group: str, done: int, total: int) -> None:
            with self._lock:
                self.done = done
            logger.info(f"  weights [{done}/{total}] {group}")

        try:
            fetched, skipped, failed = await asyncio.to_thread(
                fetch_plan, plan, on_progress=progress
            )
        except Exception as e:
            logger.exception("Weight download failed")
            self._finish("failed", str(e))
            return

        self.done = self.total
        if failed:
            self._finish("failed", f"could not download: {', '.join(failed)}")
            logger.error(self.error)
            return

        logger.info(f"Weights ready: {fetched} downloaded, {skipped} already present")
        self._finish("ready")


weights = Weights()
