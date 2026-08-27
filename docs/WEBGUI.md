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
docker/build.sh
# then open http://127.0.0.1:8000
```

The wrapper exists for one reason: the Dockerfile uses BuildKit cache mounts to
keep the ~4 GB of wheels out of the image layers, and some Docker installs still
default to the legacy builder, which fails with:

```
the --mount option requires BuildKit
```

`docker/build.sh` forces the right builder and passes any compose subcommand
through (`docker/build.sh logs -f`, `docker/build.sh down`, and so on).

To use plain compose instead, either export the variables per command:

```bash
DOCKER_BUILDKIT=1 COMPOSE_DOCKER_CLI_BUILD=1 \
  docker compose -f docker/docker-compose.yml up -d --build
```

or turn BuildKit on permanently in `/etc/docker/daemon.json`:

```json
{ "features": { "buildkit": true } }
```

followed by `sudo systemctl restart docker`. Docker Engine 23 and newer use
BuildKit by default, so this only comes up on older installs or where it has
been explicitly disabled.

Without Docker:

```bash
pip install -e '.[full,web]'
strum-web
```

### Rebuilds

Downloads live in BuildKit cache mounts, not in image layers. Those mounts are
not part of the image and are **not** cleared by `--no-cache`, so a rebuild
re-runs the install steps but does not re-fetch the ~4 GB of torch and CUDA
wheels. Only `docker builder prune` throws them away.

Layers are ordered so the expensive step survives ordinary work:

| You changed | What rebuilds |
|---|---|
| Anything under `src/`, `scripts/`, `configs/` | Just the `COPY`. Seconds. |
| `pyproject.toml` | Dependency install re-runs, but wheels come from the cache mount. |
| `docker/Dockerfile` above the install | Everything after the edit; downloads still cached. |

Dependencies are resolved against a stub `src/` package, so editing sources
cannot invalidate the install layer. The real sources are copied afterwards and
the editable install picks them up from the same path.

Prefer plain `build` and let the layer cache work:

```bash
docker/build.sh build
```

Reach for `--no-cache` only to force a genuinely clean dependency resolve --
it is not needed just because a previous build failed.

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

## Model weights

**Nothing is downloaded at build time.** The image carries no weights at all.
They live on a writable volume (`strum-checkpoints`), and the app checks that
volume on startup: anything missing is downloaded from
[`opria123/strum`](https://huggingface.co/opria123/strum), anything already
there is left alone. So the first run fetches 1.8 GB and later runs cost one
directory listing.

The download does not block startup. The server binds immediately, the UI shows
a progress banner, and jobs queued during the download **wait** for it rather
than failing against weights that have not arrived. `docker compose logs -f`
shows the same progress.

If the download fails the server keeps running, the banner says why, and jobs
fail with that reason instead of an obscure missing-checkpoint error. Restart to
retry; already-downloaded files are kept.

Set `STRUM_FETCH_CHECKPOINTS=0` to skip the check, if you populate the volume
yourself.

This runs inside the app rather than in a container entrypoint so that it also
works when `strum-web` is run directly. To fetch them by hand:

```bash
python scripts/fetch_checkpoints.py           # only what is missing
python scripts/fetch_checkpoints.py --list    # show the plan, download nothing
```

Note that `huggingface-cli download opria123/strum --local-dir checkpoints/`
is *not* equivalent: the Hub nests files under `drums/`, `guitar/` and so on,
while the loaders expect flat paths. `fetch_checkpoints.py` maps between the
two, taking the mapping from `push_to_hf.py` so the two directions cannot drift.

Separation weights are separate again -- Demucs and audio-separator download
their own on first use, into `/models/cache` inside the container.

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

## Song details

Artist and title are what the game shows in its song list, and the pipeline can
only guess them from the filename. So the upload is a two-step flow: the file is
sent once, the server reads its tags, and the browser shows them for editing
before anything is queued.

Prefilled from the file's own metadata (ID3v2, Vorbis comments, MP4 atoms --
whatever ffprobe can read), falling back to splitting the filename on `" - "`.
`year` is normalised to four digits, so a `date` of `2004-11-16` becomes `2004`.

Album art follows the same idea. Embedded cover art is extracted and shown; if
there is none, or you want different art, upload a PNG/JPEG/WebP. Either way it
is converted to a 512x512 PNG -- fitted, not stretched, padded to square.

Title and artist are required, since they name the song folder and therefore
the package file. Everything else is optional, and blank fields never erase
what the pipeline worked out on its own.

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
| `GET` | `/api/config` | Accepted extensions, upload cap, retention, weight state. |
| `GET` | `/api/weights` | Weight-download progress. Polled by the UI until ready. |
| `POST` | `/api/uploads` | Multipart upload. Stages the file, returns an id plus its tags. |
| `GET` | `/api/uploads/{id}/cover` | Staged album art as PNG. |
| `POST` | `/api/uploads/{id}/cover` | Replace the staged album art. |
| `POST` | `/api/jobs` | Queue a staged upload: `upload_id`, metadata, options. |
| `GET` | `/api/jobs` | All known jobs, newest first. |
| `GET` | `/api/jobs/{id}?log=N` | One job, with the last N log lines. |
| `POST` | `/api/jobs/{id}/cancel` | Kill or dequeue a job. |
| `GET` | `/api/jobs/{id}/download/{zip\|sng}` | Download a finished package. |
| `GET` | `/api/events` | SSE stream of job state; replays current state on connect. |
