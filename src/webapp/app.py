"""
FastAPI application for the STRUM web UI.

Upload audio or video, pick which instruments to chart, and get back Clone Hero
and YARG packages. One song is charted at a time; everything else waits in a
queue whose state is streamed to the browser over SSE.

Bind address and port come from the environment so the container can be pointed
wherever the operator wants:

    STRUM_WEB_HOST          default 0.0.0.0 (the container's own interface)
    STRUM_WEB_PORT          default 8000
    STRUM_WEB_DATA_DIR      default /data
    STRUM_WEB_MAX_UPLOAD_MB default 300
    STRUM_WEB_RETAIN_HOURS  default 24

There is no authentication. Uploads are validated and never executed, but the
endpoint does spend GPU time on request, so publish it only to a network you
trust -- keep the published port on 127.0.0.1 otherwise.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.packaging.metadata import SongMeta
from src.webapp.jobs import SEPARATORS, JobQueue, Options, Status
from src.webapp.logbuffer import buffer as log_buffer, install as install_log_buffer
from src.webapp.weights import separation, weights
from src.webapp.media import (
    ALLOWED_EXTS,
    IMAGE_EXTS,
    MAX_COVER_MB,
    MediaError,
    extract_cover,
    inspect,
    normalize_cover,
)

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

HOST = os.environ.get("STRUM_WEB_HOST", "0.0.0.0")
PORT = int(os.environ.get("STRUM_WEB_PORT", "8000"))
DATA_DIR = Path(os.environ.get("STRUM_WEB_DATA_DIR", "/data"))
MAX_UPLOAD_MB = int(os.environ.get("STRUM_WEB_MAX_UPLOAD_MB", "300"))
RETAIN_HOURS = float(os.environ.get("STRUM_WEB_RETAIN_HOURS", "24"))

UPLOAD_CHUNK = 1024 * 1024
# Staged uploads the user never submitted are swept after this long.
UPLOAD_TTL_S = 4 * 3600

_ID_RE = re.compile(r"[0-9a-f]{16,32}\Z")


def _staging(upload_id: str) -> Path:
    return DATA_DIR / "staging" / upload_id


def _checked_id(upload_id: str) -> str:
    """Reject anything that is not one of our own ids.

    Upload ids reach the filesystem as a path component, so this is what stops
    a crafted id from escaping the staging directory.
    """
    if not _ID_RE.match(upload_id):
        raise HTTPException(400, "Malformed upload id")
    return upload_id


def _cleanup_staging() -> None:
    root = DATA_DIR / "staging"
    if not root.is_dir():
        return
    cutoff = time.time() - UPLOAD_TTL_S
    for entry in root.iterdir():
        try:
            if entry.is_dir() and entry.stat().st_mtime < cutoff:
                shutil.rmtree(entry, ignore_errors=True)
        except OSError:
            pass

queue = JobQueue(DATA_DIR / "jobs", retain_hours=RETAIN_HOURS)


async def _fetch_weights() -> None:
    """Charting weights first, then separation warm-up.

    Sequential on purpose: both are large, and racing them just makes each
    slower on the sort of connection where this matters.
    """
    await weights.ensure()
    await separation.ensure()


@asynccontextmanager
async def lifespan(app: FastAPI):
    install_log_buffer()
    queue.start()
    # Checked against the mounted volume, not baked into the image. Started as a
    # task rather than awaited so the UI is reachable while 1.8 GB arrives;
    # the job queue waits on it separately.
    fetch = asyncio.create_task(_fetch_weights())
    logger.info(f"STRUM web UI ready on {HOST}:{PORT}, data dir {DATA_DIR}")
    yield
    fetch.cancel()
    await queue.stop()


app = FastAPI(title="STRUM Auto-Charter", lifespan=lifespan)


def _bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "on", "yes")


@app.get("/api/config")
async def get_config() -> dict:
    _d = Options.from_env()
    return {
        "allowed_extensions": sorted(ALLOWED_EXTS),
        "max_upload_mb": MAX_UPLOAD_MB,
        "retain_hours": RETAIN_HOURS,
        # What a job gets when the request does not say otherwise, so the UI can
        # show the deployment's settings instead of its own hardcoded ones.
        "defaults": {
            "karaoke": _d.karaoke,
            "backing_split": _d.backing_split,
            "separator": _d.separator,
        },
        "weights": weights.public(),
        "separation": separation.public(),
    }


@app.get("/api/weights")
async def get_weights() -> dict:
    """Weight-download state, polled by the UI until it settles."""
    return {**weights.public(), "separation": separation.public()}


@app.post("/api/uploads")
async def create_upload(file: UploadFile) -> dict:
    """Stage a file and report what we can read out of it.

    Charting is split into upload-then-submit so the browser can show the
    embedded tags and cover for editing first. The file is stored once and
    referenced by id; it is never sent twice.
    """
    filename = Path(file.filename or "upload").name
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTS:
        allowed = ", ".join(sorted(e.lstrip(".") for e in ALLOWED_EXTS))
        raise HTTPException(400, f"Unsupported file type. Accepted: {allowed}")

    _cleanup_staging()
    upload_id = os.urandom(12).hex()
    staging = _staging(upload_id)
    staging.mkdir(parents=True, exist_ok=True)
    source = staging / f"source{suffix}"

    limit = MAX_UPLOAD_MB * 1024 * 1024
    written = 0
    try:
        with open(source, "wb") as fh:
            while chunk := await file.read(UPLOAD_CHUNK):
                written += len(chunk)
                if written > limit:
                    raise HTTPException(413, f"File exceeds {MAX_UPLOAD_MB} MB")
                fh.write(chunk)
        if written == 0:
            raise HTTPException(400, "Empty upload")
        info = await asyncio.to_thread(inspect, source)
    except HTTPException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    except MediaError as e:
        shutil.rmtree(staging, ignore_errors=True)
        raise HTTPException(400, str(e)) from e

    # Pull the embedded cover now so the browser has something to preview.
    if info.get("has_cover"):
        await asyncio.to_thread(extract_cover, source, staging / "cover.png")

    metadata = dict(info["metadata"])
    if not metadata.get("title"):
        # Fall back to the filename, which is what the pipeline would guess.
        stem = Path(filename).stem
        artist, _, title = stem.partition(" - ")
        metadata["title"] = (title or stem).strip()
        if not metadata.get("artist") and title:
            metadata["artist"] = artist.strip()

    (staging / "upload.json").write_text(
        json.dumps({"filename": filename}), encoding="utf-8"
    )
    return {
        "upload_id": upload_id,
        "filename": filename,
        "duration_s": round(info["duration_s"], 1),
        "has_video": info["has_video"],
        "metadata": metadata,
        "has_cover": (staging / "cover.png").exists(),
    }


@app.get("/api/uploads/{upload_id}/cover")
async def get_cover(upload_id: str):
    """Serve the staged album art, embedded or uploaded."""
    cover = _staging(_checked_id(upload_id)) / "cover.png"
    if not cover.exists():
        raise HTTPException(404, "No album art for this upload")
    return FileResponse(cover, media_type="image/png")


@app.post("/api/uploads/{upload_id}/cover")
async def set_cover(upload_id: str, file: UploadFile) -> dict:
    """Replace the staged album art with an uploaded image."""
    staging = _staging(_checked_id(upload_id))
    if not staging.is_dir():
        raise HTTPException(404, "Upload expired or unknown")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in IMAGE_EXTS:
        allowed = ", ".join(sorted(e.lstrip(".") for e in IMAGE_EXTS))
        raise HTTPException(400, f"Album art must be one of: {allowed}")

    limit = MAX_COVER_MB * 1024 * 1024
    raw = staging / f"cover_upload{suffix}"
    written = 0
    try:
        with open(raw, "wb") as fh:
            while chunk := await file.read(UPLOAD_CHUNK):
                written += len(chunk)
                if written > limit:
                    raise HTTPException(413, f"Album art exceeds {MAX_COVER_MB} MB")
                fh.write(chunk)
        await asyncio.to_thread(normalize_cover, raw, staging / "cover.png")
    except MediaError as e:
        raise HTTPException(400, str(e)) from e
    finally:
        raw.unlink(missing_ok=True)
    return {"has_cover": True}


@app.post("/api/jobs")
async def create_job(
    upload_id: str = Form(...),
    title: str = Form(""),
    artist: str = Form(""),
    album: str = Form(""),
    year: str = Form(""),
    genre: str = Form(""),
    charter: str = Form(""),
    drums: str = Form("true"),
    guitar: str = Form("true"),
    bass: str = Form("true"),
    vocals: str = Form("true"),
    keys: str = Form("false"),
    stems: str = Form("true"),
    karaoke: str | None = Form(None),
    backing_split: str | None = Form(None),
    separator: str | None = Form(None),
    formats: str = Form("zip,sng"),
) -> dict:
    """Queue a staged upload for charting."""
    # Validate the cheap inputs before looking for the upload, so bad arguments
    # report as bad arguments rather than as a missing upload.
    #
    # Anything the request leaves out falls back to the deployment's own
    # settings rather than to a hardcoded value, so STRUM_* in compose actually
    # decides what a bare API call does.
    defaults = Options.from_env()

    if separator is None:
        resolved_separator = defaults.separator
    elif separator.strip().lower() in SEPARATORS:
        resolved_separator = SEPARATORS[separator.strip().lower()]
    else:
        raise HTTPException(400, f"Unknown separator: {separator}")

    wanted = tuple(f for f in formats.split(",") if f in ("zip", "sng"))
    if not wanted:
        raise HTTPException(400, "Pick at least one package format")

    staging = _staging(_checked_id(upload_id))
    sources = [p for p in staging.glob("source.*")] if staging.is_dir() else []
    if not sources:
        raise HTTPException(404, "Upload expired or unknown; upload the file again")

    options = Options(
        drums=_bool(drums, True), guitar=_bool(guitar, True),
        bass=_bool(bass, True), vocals=_bool(vocals, True),
        keys=_bool(keys), stems=_bool(stems, True),
        karaoke=_bool(karaoke, defaults.karaoke),
        backing_split=_bool(backing_split, defaults.backing_split),
        separator=resolved_separator, formats=wanted,
    )
    meta = SongMeta(title=title, artist=artist, album=album,
                    year=year, genre=genre, charter=charter)

    try:
        original = json.loads((staging / "upload.json").read_text())["filename"]
    except (OSError, ValueError, KeyError):
        original = sources[0].name

    cover = staging / "cover.png"
    try:
        job = await queue.submit(
            original, sources[0], options, meta, cover if cover.exists() else None
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    return job.public()


@app.get("/api/jobs")
async def list_jobs() -> list[dict]:
    return [
        job.public()
        for job in sorted(queue.jobs.values(), key=lambda j: j.created_at, reverse=True)
    ]


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str, log: int = 50) -> dict:
    job = queue.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "No such job")
    return {**job.public(), "log": job.log[-max(0, min(log, 500)):]}


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str) -> dict:
    if job_id not in queue.jobs:
        raise HTTPException(404, "No such job")
    if not queue.cancel(job_id):
        raise HTTPException(409, "Job already finished")
    return queue.jobs[job_id].public()


@app.get("/api/jobs/{job_id}/download/{fmt}")
async def download(job_id: str, fmt: str):
    """Serve a finished package.

    The path is looked up from the job's own record rather than built from the
    request, so nothing the client sends can select a file we did not produce.
    """
    job = queue.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "No such job")
    if job.status is not Status.DONE:
        raise HTTPException(409, f"Job is {job.status.value}")
    path_str = job.packages.get(fmt)
    if path_str is None:
        raise HTTPException(404, f"No {fmt} package for this job")
    path = Path(path_str)
    if not path.exists():
        raise HTTPException(410, "Package has been cleaned up")
    return FileResponse(path, filename=path.name, media_type="application/octet-stream")


@app.get("/api/jobs/{job_id}/log")
async def job_log(job_id: str, tail: int = 0) -> PlainTextResponse:
    """Full pipeline output for one job, as plain text.

    Served from disk, so it is the complete log rather than the capped copy the
    JSON endpoint carries. `tail` limits it to the last N lines; 0 means all.
    """
    job = queue.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "No such job")

    path = queue.log_path(job_id)
    if path.exists():
        text = path.read_text(encoding="utf-8", errors="replace")
    elif job.log:
        # The pipeline may not have started yet, or the file aged out.
        text = "\n".join(job.log) + "\n"
    else:
        text = f"No pipeline output yet (job is {job.status.value}).\n"

    if tail > 0:
        text = "\n".join(text.splitlines()[-tail:]) + "\n"
    return PlainTextResponse(text)


@app.get("/api/diagnostics")
async def diagnostics() -> dict:
    """Versions and environment, for diagnosing a deployment you cannot shell into.

    Dependency version skew -- torch against torchaudio, setuptools against
    packages that still want pkg_resources -- surfaces as unrelated failures
    deep in the pipeline, so it is worth being able to read the versions back.

    Only STRUM_* environment variables are reported; the rest of the environment
    is none of the browser's business.
    """
    import importlib.metadata as md
    import platform

    packages = [
        "torch", "torchaudio", "torchvision", "demucs", "audio-separator",
        "basic-pitch", "tensorflow", "librosa", "numpy", "setuptools",
        "onnxruntime", "huggingface-hub", "soundfile", "mido",
    ]
    versions: dict[str, str] = {}
    for name in packages:
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            versions[name] = "not installed"

    cuda: dict[str, object] = {}
    try:
        import torch

        cuda = {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
            "torch_cuda": torch.version.cuda or "cpu build",
        }
        if torch.cuda.is_available():
            # Total VRAM is the real ceiling on which separation models fit;
            # free is what is left with the server idle.
            free, total = torch.cuda.mem_get_info()
            cuda["vram_total_gb"] = round(total / 1e9, 1)
            cuda["vram_free_gb"] = round(free / 1e9, 1)
    except Exception as e:
        cuda = {"error": str(e)}

    try:
        import importlib
        importlib.import_module("pkg_resources")
        pkg_resources = "present"
    except Exception as e:
        pkg_resources = f"missing: {e}"

    memory: dict[str, object] = {}
    try:
        # /proc/meminfo rather than psutil, which is not a dependency. Inside a
        # container this reports the host's memory unless a cgroup limit is set,
        # which is exactly the number that matters for an OOM kill.
        info = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            info[key] = int(rest.strip().split()[0]) * 1024
        memory = {
            "total_gb": round(info.get("MemTotal", 0) / 1e9, 1),
            "available_gb": round(info.get("MemAvailable", 0) / 1e9, 1),
            "swap_total_gb": round(info.get("SwapTotal", 0) / 1e9, 1),
        }

        # SwapTotal counts zram, which is compressed memory living in RAM
        # rather than on storage. It does buy capacity, but only by whatever
        # ratio the pages compress to, and it can never exceed physical RAM.
        # Model weights and audio tensors are float data that compresses close
        # to 1:1, so on this workload a large SwapTotal made up of zram is close
        # to no headroom at all -- which is worth reporting separately.
        devices = []
        real_swap = 0
        swaps = Path("/proc/swaps")
        if swaps.exists():
            for line in swaps.read_text().splitlines()[1:]:
                fields = line.split()
                if len(fields) < 4:
                    continue
                name, size_kb = fields[0], int(fields[2])
                is_zram = "zram" in name
                devices.append({
                    "name": name,
                    "size_gb": round(size_kb * 1024 / 1e9, 1),
                    "used_gb": round(int(fields[3]) * 1024 / 1e9, 1),
                    "zram": is_zram,
                })
                if not is_zram:
                    real_swap += size_kb * 1024
        memory["swap_devices"] = devices
        memory["real_swap_gb"] = round(real_swap / 1e9, 1)
        if devices and not real_swap:
            memory["note"] = (
                "All swap is zram: compressed pages held in RAM, capped by "
                "physical memory. Float tensors compress close to 1:1, so this "
                "adds little usable headroom for charting. A swapfile on disk "
                "would give real spill space."
            )
        # ZFS ARC is not counted as used memory by MemAvailable's reckoning,
        # but it is real RAM the pipeline cannot have until ARC gives it back,
        # and ARC reclaims slowly relative to a sudden multi-GB allocation.
        # Default zfs_arc_max is half of RAM, which on a 16 GB host is enough to
        # decide whether a run fits.
        arcstats = Path("/proc/spl/kstat/zfs/arcstats")
        if arcstats.exists():
            arc = {}
            for line in arcstats.read_text().splitlines():
                fields = line.split()
                if len(fields) == 3 and fields[0] in ("size", "c_max", "c_min"):
                    arc[fields[0]] = int(fields[2])
            memory["zfs_arc"] = {
                "current_gb": round(arc.get("size", 0) / 1e9, 1),
                "max_gb": round(arc.get("c_max", 0) / 1e9, 1),
                "hint": (
                    "ARC holds real RAM and reclaims slowly under a sudden "
                    "allocation. Cap it with zfs_arc_max if charting is being "
                    "OOM-killed."
                ),
            }

        limit = Path("/sys/fs/cgroup/memory.max")
        if limit.exists():
            raw = limit.read_text().strip()
            memory["cgroup_limit_gb"] = (
                "unlimited" if raw == "max" else round(int(raw) / 1e9, 1)
            )
    except Exception as e:
        memory = {"error": str(e)}

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "memory": memory,
        "packages": versions,
        "pkg_resources": pkg_resources,
        "cuda": cuda,
        "env": {k: v for k, v in sorted(os.environ.items()) if k.startswith("STRUM_")},
        "checkpoint_dir": str(weights.dest),
    }


@app.get("/api/logs")
async def server_log(tail: int = 500, level: str = "INFO") -> PlainTextResponse:
    """Recent server log, as plain text.

    This is the app's own logging -- weight downloads, job lifecycle, separation
    fallbacks -- which otherwise only exists in the container's stdout. `level`
    filters to that severity and above.
    """
    threshold = logging.getLevelName(level.strip().upper())
    if not isinstance(threshold, int):
        raise HTTPException(400, f"Unknown log level: {level}")
    lines = log_buffer.tail(limit=max(0, min(tail, 5000)), min_level=threshold)
    return PlainTextResponse(("\n".join(lines) + "\n") if lines else "(no log yet)\n")


@app.get("/api/events")
async def events(request: Request) -> StreamingResponse:
    """Server-sent events carrying every job state change.

    The initial burst replays current state so a tab opened mid-run does not sit
    blank until the next transition.
    """
    async def stream():
        q = queue.subscribe()
        try:
            for job in sorted(queue.jobs.values(), key=lambda j: j.created_at):
                yield f"data: {json.dumps(job.public())}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    payload = await asyncio.wait_for(q.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    # Comment frame: keeps proxies from timing the stream out.
                    yield ": keepalive\n\n"
                    continue
                yield f"data: {json.dumps(payload)}\n\n"
        finally:
            queue.unsubscribe(q)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# Mounted last so the API routes above win on any path collision.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


def main() -> None:
    """Console entry point: `strum-web`."""
    import uvicorn

    logging.basicConfig(
        level=os.environ.get("STRUM_WEB_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
