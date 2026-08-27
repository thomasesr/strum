# Web UI

Upload a song, pick what to chart, get Clone Hero and YARG packages back. The
browser starts the downloads on its own the moment a job finishes.

```
browser ──upload──► FastAPI ──► queue (one job at a time)
                       │              │
                       │              ├─ ffmpeg: demux audio if the upload was video
                       │              ├─ scripts/batch_pipeline.py (subprocess)
                       │              └─ package: .zip + .sng
                       └──SSE progress──► browser ──auto-download──► .zip / .sng
```

## Running it

```bash
docker compose -f docker/docker-compose.yml up --build
# then open http://127.0.0.1:8000
```

Without Docker:

```bash
pip install -e '.[full,web]'
strum-web
```

**Python 3.11 only.** `basic-pitch` requires `tensorflow<2.15.1`, and
TensorFlow shipped no cp312 wheels before 2.16, so on Python 3.12 the two
constraints have no solution and the install dies with `ResolutionImpossible`.
The image pins 3.11 for this reason; if you install by hand, use 3.11 too.
Installing just `.[web]` has no such constraint.

A C compiler is also needed at install time: `diffq`, a hard `audio-separator`
dependency on Linux, publishes no cp311 wheel at any version and always builds
from source. The image installs `build-essential` and removes it again in the
same layer.

The image is built on `python:3.11-slim` rather than a CUDA base: the torch
cu126 wheels bundle the CUDA runtime and cuDNN they use, so the host
requirement is the same either way -- NVIDIA driver plus
nvidia-container-toolkit -- and the image is several GB smaller.

## Accepted input

| Kind | Extensions | Handling |
|---|---|---|
| Audio | `mp3` `wav` `ogg` `flac` `m4a` | Passed to the pipeline as-is. |
| Video | `mp4` `mkv` | ffmpeg demuxes the first audio stream to FLAC first. |

Every upload is checked with `ffprobe`, not by its extension or the
browser-supplied content type: a file with no audio stream, or one that is not
media at all, is rejected before it can reach the queue. Uploads shorter than
5 s or longer than 30 min are refused, and anything over
`STRUM_WEB_MAX_UPLOAD_MB` is cut off mid-stream.

## Options

| Option | Effect |
|---|---|
| Drums / Guitar / Bass / Vocals / Keys | Which instrument parts to chart. Keys is off by default; it is the weakest part. |
| Karaoke pre-pass | Strip the lead vocal before separation. See [Separation](SEPARATION.md). |
| Split backing vocals | Second pass isolating backing vocals. Costs one more separation run. |
| Stem model | `Demucs htdemucs_6s` or `BS-RoFormer SW`. |
| Multi-track audio | One `.ogg` per instrument, which is what enables mute-on-miss in game. Off means a single mixed `song.ogg`. |
| `.zip` / `.sng` | `.zip` extracts to a folder both games read. `.sng` is YARG's single-file container. |

## Output

With multi-track audio on, a package contains:

```
Artist - Title/
├── song.ini        metadata (becomes the .sng metadata section)
├── notes.mid       the chart, all instruments, four difficulties each
├── drums.ogg
├── bass.ogg
├── guitar.ogg
├── keys.ogg
├── vocals.ogg
├── song.ogg        backing: everything not charted to its own track
└── album.png
```

`song.ogg` is deliberately *not* the full mix here — it is the leftover audio.
Shipping the full mix alongside per-instrument tracks would play the song twice.

Intermediate separation output (`stems/`) never enters a package.

## Configuration

All via environment variables; the container reads the same ones.

| Variable | Default | Meaning |
|---|---|---|
| `STRUM_WEB_HOST` | `0.0.0.0` | Bind address. Inside a container this should stay `0.0.0.0`; limit reach with the published port instead. |
| `STRUM_WEB_PORT` | `8000` | Bind port. |
| `STRUM_WEB_DATA_DIR` | `/data` | Uploads, job output and packages. |
| `STRUM_WEB_MAX_UPLOAD_MB` | `300` | Per-upload size cap. |
| `STRUM_WEB_RETAIN_HOURS` | `24` | How long finished jobs and their files survive. |
| `STRUM_WEB_LOG_LEVEL` | `INFO` | Server log level. |

Compose adds two more for the host side: `STRUM_BIND` (default `127.0.0.1`) and
`STRUM_PORT` (default `8000`), which is where the published port lands.

Separation defaults (`STRUM_KARAOKE`, `STRUM_SEPARATOR`, …) are documented in
[Separation](SEPARATION.md); the web UI overrides them per job.

## Exposure

There is no authentication. Uploads are validated and never executed, and
download paths are looked up from the server's own job records rather than
built from the request — but any client that can reach the service can queue
work on your GPU and read back what it produced.

The compose file therefore publishes to `127.0.0.1` by default. Set
`STRUM_BIND=0.0.0.0` only on a network you trust, and put a reverse proxy with
authentication in front of it for anything beyond that.

## Job behaviour

Jobs run strictly one at a time — one song already saturates a GPU, so a queue
gives predictable ordering rather than parallelism. The pipeline runs as a
child process, so a CUDA OOM kills that job and not the server, and its GPU
memory is reclaimed on exit.

Progress is derived from milestones in the pipeline's log, so the bar indicates
which stage is running rather than a true time estimate.

Cancelling kills the running child and deletes that job's working files.
Finished jobs are cleaned up after `STRUM_WEB_RETAIN_HOURS`.

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/config` | Accepted extensions, upload cap, retention. |
| `POST` | `/api/jobs` | Multipart upload plus options. Returns the job. |
| `GET` | `/api/jobs` | All known jobs, newest first. |
| `GET` | `/api/jobs/{id}?log=N` | One job, with the last N log lines. |
| `POST` | `/api/jobs/{id}/cancel` | Kill or dequeue a job. |
| `GET` | `/api/jobs/{id}/download/{zip\|sng}` | Download a finished package. |
| `GET` | `/api/events` | SSE stream of job state; replays current state on connect. |
