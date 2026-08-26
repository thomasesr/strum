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
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.webapp.jobs import JobQueue, Options, Status
from src.webapp.media import ALLOWED_EXTS

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

HOST = os.environ.get("STRUM_WEB_HOST", "0.0.0.0")
PORT = int(os.environ.get("STRUM_WEB_PORT", "8000"))
DATA_DIR = Path(os.environ.get("STRUM_WEB_DATA_DIR", "/data"))
MAX_UPLOAD_MB = int(os.environ.get("STRUM_WEB_MAX_UPLOAD_MB", "300"))
RETAIN_HOURS = float(os.environ.get("STRUM_WEB_RETAIN_HOURS", "24"))

UPLOAD_CHUNK = 1024 * 1024

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


@app.post("/api/jobs")
async def create_job(
    file: UploadFile,
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
    """Accept an upload and queue it for charting."""
    filename = Path(file.filename or "upload").name
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTS:
        allowed = ", ".join(sorted(e.lstrip(".") for e in ALLOWED_EXTS))
        raise HTTPException(400, f"Unsupported file type. Accepted: {allowed}")

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

    # Stage the upload before creating the job so a half-written file never
    # reaches the queue. The name is ours, not the client's, so a crafted
    # filename cannot escape the job directory.
    staging = DATA_DIR / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    tmp = staging / f"{os.urandom(8).hex()}{suffix}"
    limit = MAX_UPLOAD_MB * 1024 * 1024
    written = 0
    try:
        with open(tmp, "wb") as fh:
            while chunk := await file.read(UPLOAD_CHUNK):
                written += len(chunk)
                if written > limit:
                    raise HTTPException(413, f"File exceeds {MAX_UPLOAD_MB} MB")
                fh.write(chunk)
    except HTTPException:
        tmp.unlink(missing_ok=True)
        raise
    if written == 0:
        tmp.unlink(missing_ok=True)
        raise HTTPException(400, "Empty upload")

    try:
        job = await queue.submit(filename, tmp, options)
    finally:
        # submit() moves the file on success; this only fires on rejection.
        tmp.unlink(missing_ok=True)

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
