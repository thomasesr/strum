"""
Job model and the single-worker queue that drives the charting pipeline.

Charting is GPU-bound and one song already saturates the card, so jobs run
strictly one at a time; the queue exists to give waiting uploads a predictable
order rather than to get parallelism.

Each job shells out to scripts/batch_pipeline.py instead of importing it. That
buys process isolation (a CUDA OOM kills the worker's child, not the server),
frees GPU memory reliably at exit, and gives us a line-oriented log to stream
to the browser for free.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shutil
import signal
import sys
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from src.packaging.metadata import SongMeta, apply_metadata
from src.packaging.package import package_song, safe_name
from src.webapp.media import MediaError, extract_audio, inspect
from src.webapp.weights import weights

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_SCRIPT = REPO_ROOT / "scripts" / "batch_pipeline.py"


class Status(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in (Status.DONE, Status.FAILED, Status.CANCELLED)


# Substrings the pipeline logs, mapped to a label and a rough completion
# fraction. The pipeline has no progress reporting of its own, so the bar is
# derived from these milestones -- it is an indication of stage, not a timer.
STAGE_MARKERS: list[tuple[str, str, float]] = [
    ("Karaoke pre-separation", "Removing lead vocals", 0.08),
    ("Separating stems", "Separating stems", 0.15),
    ("Re-separating drums", "Refining drum stem", 0.35),
    ("Detecting tempo", "Detecting tempo", 0.45),
    ("Transcribing drums", "Charting drums", 0.52),
    ("Transcribing guitar", "Charting guitar", 0.65),
    ("Transcribing bass", "Charting bass", 0.74),
    ("Transcribing vocals", "Charting vocals", 0.82),
    ("Transcribing keys", "Charting keys", 0.88),
    ("Multi-track audio", "Writing audio tracks", 0.92),
    ("Created: song.ini", "Writing metadata", 0.95),
]


def _exit_reason(returncode: int) -> str:
    """Explain a non-zero pipeline exit.

    A negative code means the child was killed by a signal, which subprocess
    reports as the negated signal number. SIGKILL is worth calling out by name:
    on this pipeline it is almost always the kernel OOM killer, and "exited with
    code -9" gives the user nothing to act on.
    """
    if returncode >= 0:
        return f"Pipeline exited with code {returncode}"

    try:
        name = signal.Signals(-returncode).name
    except ValueError:
        return f"Pipeline killed by signal {-returncode}"

    if name == "SIGKILL":
        return (
            "Pipeline was killed (SIGKILL) — almost always the out-of-memory "
            "killer. Charting loads PyTorch and TensorFlow at once; give the "
            "host more RAM or swap, or disable an instrument to lower the peak."
        )
    return f"Pipeline killed by {name}"


@dataclass
class Options:
    """Per-job pipeline settings chosen in the web UI."""

    drums: bool = True
    guitar: bool = True
    bass: bool = True
    vocals: bool = True
    keys: bool = False
    stems: bool = True
    karaoke: bool = True
    backing_split: bool = False
    separator: str = "demucs"
    formats: tuple[str, ...] = ("zip", "sng")

    def cli_flags(self) -> list[str]:
        flags = []
        for name in ("drums", "guitar", "bass", "vocals"):
            if not getattr(self, name):
                flags.append(f"--no-{name}")
        if self.keys:
            flags.append("--keys")
        if self.stems:
            flags.append("--stems")
        return flags

    def env(self) -> dict[str, str]:
        return {
            "STRUM_KARAOKE": "1" if self.karaoke else "0",
            "STRUM_KARAOKE_BACKING_SPLIT": "1" if self.backing_split else "0",
            "STRUM_SEPARATOR": self.separator,
        }


@dataclass
class Job:
    id: str
    filename: str
    options: Options
    meta: SongMeta = field(default_factory=SongMeta)
    status: Status = Status.QUEUED
    stage: str = "Queued"
    progress: float = 0.0
    error: str = ""
    log: list[str] = field(default_factory=list)
    packages: dict[str, str] = field(default_factory=dict)
    duration_s: float = 0.0
    created_at: float = field(default_factory=time.time)
    finished_at: float = 0.0

    def public(self) -> dict:
        """The shape the browser sees. Paths never cross this boundary."""
        return {
            "id": self.id,
            "filename": self.filename,
            "title": self.meta.title,
            "artist": self.meta.artist,
            "status": self.status.value,
            "stage": self.stage,
            "progress": round(self.progress, 3),
            "error": self.error,
            "packages": sorted(self.packages),
            "duration_s": round(self.duration_s, 1),
            "created_at": self.created_at,
        }


class JobQueue:
    """Serial job runner with fan-out of state changes to SSE subscribers."""

    def __init__(self, data_dir: Path, max_log_lines: int = 500, retain_hours: float = 24.0):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.max_log_lines = max_log_lines
        self.retain_hours = retain_hours

        self.jobs: dict[str, Job] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._subscribers: set[asyncio.Queue] = set()
        self._worker: asyncio.Task | None = None
        self._current: asyncio.subprocess.Process | None = None
        self._current_id: str | None = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._run_forever())

    async def stop(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker
            self._worker = None

    def job_dir(self, job_id: str) -> Path:
        return self.data_dir / job_id

    def log_path(self, job_id: str) -> Path:
        """Full pipeline output. The in-memory log is capped; this is not."""
        return self.job_dir(job_id) / "pipeline.log"

    # -- pub/sub -----------------------------------------------------------

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    def _publish(self, job: Job) -> None:
        payload = job.public()
        for q in list(self._subscribers):
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                # A browser tab that stopped reading must not stall charting.
                self._subscribers.discard(q)

    def _set(self, job: Job, **fields) -> None:
        for key, value in fields.items():
            setattr(job, key, value)
        self._publish(job)

    # -- submission --------------------------------------------------------

    async def submit(
        self,
        filename: str,
        source: Path,
        options: Options,
        meta: SongMeta | None = None,
        cover: Path | None = None,
    ) -> Job:
        """Take ownership of a staged upload and queue it.

        The file is moved into the job directory *before* the id is queued: the
        worker may pick it up on the next tick, so it has to already be there.
        """
        job_id = uuid.uuid4().hex[:12]
        job = Job(id=job_id, filename=filename, options=options,
                  meta=meta or SongMeta())
        source = Path(source)

        # Validate up front so a bad upload fails immediately instead of after
        # sitting in the queue behind an hour of GPU work.
        try:
            await asyncio.to_thread(inspect, source)
        except MediaError as e:
            job.status, job.error, job.stage = Status.FAILED, str(e), "Rejected"
            job.finished_at = time.time()
            self.jobs[job_id] = job
            self._publish(job)
            return job

        in_dir = self.job_dir(job_id) / "input"
        in_dir.mkdir(parents=True, exist_ok=True)
        source.replace(in_dir / f"source{source.suffix.lower()}")
        if cover is not None and Path(cover).exists():
            # Kept beside the input rather than in the song folder, which does
            # not exist until the pipeline has run.
            Path(cover).replace(self.job_dir(job_id) / "cover.png")

        self.jobs[job_id] = job
        await self._queue.put(job_id)
        self._publish(job)
        return job

    def cancel(self, job_id: str) -> bool:
        """Cancel a job. Running jobs are killed; queued ones are skipped later."""
        job = self.jobs.get(job_id)
        if job is None or job.status.terminal:
            return False
        if job_id == self._current_id and self._current is not None:
            with contextlib.suppress(ProcessLookupError):
                self._current.kill()
        self._set(job, status=Status.CANCELLED, stage="Cancelled", finished_at=time.time())
        return True

    # -- worker ------------------------------------------------------------

    async def _run_forever(self) -> None:
        while True:
            job_id = await self._queue.get()
            job = self.jobs.get(job_id)
            if job is None or job.status is Status.CANCELLED:
                continue
            try:
                await self._run(job)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception(f"Job {job.id} crashed")
                self._set(job, status=Status.FAILED, stage="Failed", error=str(e),
                          finished_at=time.time())
            finally:
                self._cleanup_old()

    async def _run(self, job: Job) -> None:
        started = time.time()
        work = self.job_dir(job.id)
        in_dir, out_dir, pkg_dir = work / "input", work / "output", work / "packages"

        self._set(job, status=Status.RUNNING, stage="Preparing", progress=0.02)

        # On a fresh volume the weights are still arriving. Wait rather than
        # start a run that would fail on a missing checkpoint.
        if not weights.ready:
            self._set(job, stage="Waiting for model weights", progress=0.02)
        if not await weights.wait_ready():
            self._set(job, status=Status.FAILED, stage="Failed",
                      error=weights.error or "Model weights are unavailable",
                      duration_s=time.time() - started, finished_at=time.time())
            return

        # Video uploads get demuxed here; the pipeline only ever sees audio.
        source = next(in_dir.glob("source.*"))
        audio = await asyncio.to_thread(extract_audio, source, work / "extracted")

        # batch_pipeline takes a directory and reads artist/title off the
        # filename, so present it one file named the way it expects. Without
        # this it sees "source.mp3", logs "Unknown Artist - source", and wastes
        # its MusicBrainz and album-art lookups on that.
        audio_dir = work / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        for stray in audio_dir.iterdir():
            stray.unlink()
        stem = safe_name(
            f"{job.meta.artist} - {job.meta.title}".strip(" -"),
            fallback=Path(job.filename).stem,
        )
        staged = audio_dir / f"{stem}{audio.suffix}"
        try:
            staged.symlink_to(audio.resolve())
        except OSError:
            shutil.copy2(audio, staged)

        env = {**os.environ, **job.options.env(), "PYTHONUNBUFFERED": "1"}
        cmd = [sys.executable, str(PIPELINE_SCRIPT), str(audio_dir), str(out_dir),
               *job.options.cli_flags()]
        logger.info(f"Job {job.id}: {' '.join(cmd)}")
        self._set(job, stage="Starting pipeline", progress=0.05)

        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=str(REPO_ROOT), env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        self._current, self._current_id = proc, job.id
        try:
            await self._pump_logs(job, proc)
            returncode = await proc.wait()
        finally:
            self._current, self._current_id = None, None

        if job.status is Status.CANCELLED:
            shutil.rmtree(work, ignore_errors=True)
            return
        if returncode != 0:
            tail = "\n".join(job.log[-15:])
            self._set(job, status=Status.FAILED, stage="Failed",
                      error=_exit_reason(returncode),
                      duration_s=time.time() - started, finished_at=time.time())
            logger.error(f"Job {job.id} failed ({_exit_reason(returncode)}):\n{tail}")
            return

        self._set(job, stage="Packaging", progress=0.96)

        # batch_pipeline exits 0 even when every song fails, and it creates the
        # song folder before charting, so neither the return code nor the
        # folder's existence proves anything. The chart file does.
        song_folders = [p for p in out_dir.iterdir() if p.is_dir()] if out_dir.exists() else []
        charted = [p for p in song_folders if (p / "notes.mid").exists()]
        if not charted:
            reason = self._failure_reason(job)
            self._set(job, status=Status.FAILED, stage="Failed",
                      error=reason,
                      duration_s=time.time() - started, finished_at=time.time())
            logger.error(f"Job {job.id} produced no chart: {reason}")
            return

        song = charted[0]

        # Corrections are applied after charting: the pipeline guesses artist
        # and title from the filename, and that guess drives the folder name,
        # song.ini and therefore the package filename.
        cover = work / "cover.png"
        song = await asyncio.to_thread(
            apply_metadata, song, job.meta, cover if cover.exists() else None
        )

        packages = await asyncio.to_thread(
            package_song, song, pkg_dir, job.options.formats, safe_name(song.name)
        )
        # Working stems are the bulk of the job directory and are already
        # inside the packages; drop them once packaging succeeded.
        shutil.rmtree(song / "stems", ignore_errors=True)

        self._set(
            job, status=Status.DONE, stage="Done", progress=1.0,
            packages={fmt: str(path) for fmt, path in packages.items()},
            duration_s=time.time() - started, finished_at=time.time(),
        )

    @staticmethod
    def _failure_reason(job: Job) -> str:
        """Pull the pipeline's own error out of the log, for the UI to show.

        Its summary lines are far more useful than "produced no chart", and the
        user cannot see container logs.
        """
        for line in reversed(job.log):
            if "✗" in line or " - ERROR - " in line:
                return line.split(" - ERROR - ")[-1].split("✗")[-1].strip()
        return "Pipeline produced no chart"

    async def _pump_logs(self, job: Job, proc: asyncio.subprocess.Process) -> None:
        """Stream the child's output into the job log, updating the stage as it goes.

        Also written to disk unabridged: the in-memory copy is capped so a long
        run cannot grow without bound, but a diagnosis usually needs the lines
        before the failure, which are the first to be dropped.
        """
        assert proc.stdout is not None
        path = self.log_path(job.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", buffering=1) as fh:
            while True:
                raw = await proc.stdout.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").rstrip()
                if not line:
                    continue
                fh.write(line + "\n")
                job.log.append(line)
                if len(job.log) > self.max_log_lines:
                    del job.log[: len(job.log) - self.max_log_lines]
                for marker, label, fraction in STAGE_MARKERS:
                    if marker in line and fraction > job.progress:
                        self._set(job, stage=label, progress=fraction)
                        break

    def _cleanup_old(self) -> None:
        """Drop finished jobs and their files once they age out."""
        cutoff = time.time() - self.retain_hours * 3600
        for job_id, job in list(self.jobs.items()):
            if job.status.terminal and job.finished_at and job.finished_at < cutoff:
                shutil.rmtree(self.job_dir(job_id), ignore_errors=True)
                del self.jobs[job_id]
