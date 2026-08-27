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
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.packaging.metadata import SongMeta
from src.webapp.jobs import JobQueue, Options, Status
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    queue.start()
    logger.info(f"STRUM web UI ready on {HOST}:{PORT}, data dir {DATA_DIR}")
    yield
    await queue.stop()


app = FastAPI(title="STRUM Auto-Charter", lifespan=lifespan)


def _bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "on", "yes")


@app.get("/api/config")
async def get_config() -> dict:
    return {
        "allowed_extensions": sorted(ALLOWED_EXTS),
        "max_upload_mb": MAX_UPLOAD_MB,
        "retain_hours": RETAIN_HOURS,
    }


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
    karaoke: str = Form("true"),
    backing_split: str = Form("false"),
    separator: str = Form("demucs"),
    formats: str = Form("zip,sng"),
) -> dict:
    """Queue a staged upload for charting."""
    staging = _staging(_checked_id(upload_id))
    sources = [p for p in staging.glob("source.*")] if staging.is_dir() else []
    if not sources:
        raise HTTPException(404, "Upload expired or unknown; upload the file again")

    if separator not in ("demucs", "bs_roformer_sw"):
        raise HTTPException(400, "Unknown separator")
    wanted = tuple(f for f in formats.split(",") if f in ("zip", "sng"))
    if not wanted:
        raise HTTPException(400, "Pick at least one package format")

    options = Options(
        drums=_bool(drums, True), guitar=_bool(guitar, True),
        bass=_bool(bass, True), vocals=_bool(vocals, True),
        keys=_bool(keys), stems=_bool(stems, True),
        karaoke=_bool(karaoke, True), backing_split=_bool(backing_split),
        separator=separator, formats=wanted,
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
