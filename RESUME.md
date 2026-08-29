# STRUM fork — session handoff

Fork of [opria123/strum](https://github.com/opria123/strum), branched at `9f420cb`.
Goal: turn an MP3 (or video) into a Clone Hero / YARG chart package through a
browser, with a karaoke pre-separation pass in front of the stem separator.

- Repo: `/home/dev/git/charter-ai`
- Remote: `origin` = `thomasesr/strum`, `upstream` = `opria123/strum`
- Branch: `feat/karaoke-preseparation` (22 commits, `6e8a52f`..`784a9e5`)
- Deployment: `http://strum:8000`
- Test audio: `songs/` (2 Led Zeppelin / Black Sabbath MP3s)

## Running it

```bash
git pull
docker/build.sh          # build only
docker/build.sh up -d    # start
docker/build.sh logs -f
```

`docker/build.sh` exists because the Dockerfile uses BuildKit cache mounts and
this host defaults to the legacy builder. It forces `DOCKER_BUILDKIT=1` and
passes any compose subcommand through.

## Where it got to

Working end to end. Last verified run — *Going to California*, 10.3 min:

```
PART DRUMS    1000 notes
PART GUITAR   2390 notes
PART BASS     2327 notes
PART VOCALS    267 notes
HARM1          267 notes
diff_vocals = 1
```

Packages: `.zip` + `.sng`, multi-stem audio (`drums/bass/guitar/keys/vocals/song.ogg`),
four difficulties per instrument, metadata and album art applied from the UI.

An earlier Black Sabbath run produced 2042 Expert drum hits and charted
guitar + bass. Neither chart has been **played in a game yet** — that is the
biggest untested thing in the whole project.

## What was built

**Separation**
- Mel-Band RoFormer karaoke pass before the 6-stem split; lead vocal removed
  first so sung melody stops bleeding into `other`/`guitar`, and its isolate
  replaces the Demucs vocals stem. Optional backing-vocal split.
- BS-RoFormer SW as an alternative 6-stem model, via **ONNX Runtime**.
- GPU memory released between every stage (`src/preprocessing/gpu.py`).

**Web app** (`src/webapp/`)
- Upload audio or video (ffmpeg demuxes video), two-phase so ID3 tags and
  embedded cover art prefill an editable metadata form before queueing.
- Serial job queue, SSE progress, cancel/delete, auto-download on finish.
- Weights fetched on first run into volumes; jobs wait for them.

**Packaging** (`src/packaging/`)
- `.zip` (both games) and `.sng` (YARG single-file container, written from spec).
- User metadata applied *after* charting, since `parse_filename` runs a
  MusicBrainz check that can swap artist and title.

**Debug endpoints** — added because this deployment is remote:
- `GET /api/jobs/{id}/log` — full pipeline output, plain text
- `GET /api/logs?tail=&level=` — server log ring buffer
- `GET /api/diagnostics` — versions, CUDA, VRAM, RAM, swap, ZFS ARC

## Host environment (BEAST)

Matters more than usual here — several bugs traced back to it.

- RTX 3050, **6 GB VRAM**. 16.7 GB RAM. Root filesystem is **ZFS**, pool `zpool`.
- **zram swap is masked off** (`systemd-zram-setup@zram0.service`). It was
  compressed RAM, so it added little usable headroom for float tensors.
- Swap is a **zvol**: `/dev/zvol/zpool/swap`, 32 GB, priority 10.
  Not yet in `/etc/fstab` — add:
  ```
  /dev/zvol/zpool/swap none swap defaults,pri=10,nofail,x-systemd.requires=zfs-volume-wait.service 0 0
  ```
- **The machine froze once** with zvol swap under load. ZFS swap-on-zvol has a
  known deadlock. Capping ARC is the safer lever and is **still not applied**:
  ```bash
  grep -E "^(size|c_max)" /proc/spl/kstat/zfs/arcstats
  echo $((4 * 1024**3)) | sudo tee /sys/module/zfs/parameters/zfs_arc_max
  # persist: options zfs zfs_arc_max=4294967296  in /etc/modprobe.d/zfs.conf
  ```

## Next steps

1. **Play a chart in Clone Hero or YARG.** Everything so far is structural
   verification. Only playing tells you whether the charts are any good.
2. **Cap ZFS ARC**, then re-run and see whether the zvol swap is needed at all.
3. **Rebuild and try BS-RoFormer for real.** The ONNX backend is validated
   offline but has never run in the container. Watch for
   `BS-RoFormer ONNX: N chunks, providers=['CUDAExecutionProvider']` — if it
   says CPU, expect ~17 min/song and `onnxruntime-gpu` did not load its provider.
4. **Compare Demucs vs BS-RoFormer** on the same song. RoFormer should give
   cleaner `guitar`/`other`, which is where basic-pitch invents notes.
5. **Lyric sync** — see `docs/ROADMAP.md`. Forced alignment against the lyrics
   `src/lyrics/fetcher.py` already retrieves, not Whisper's transcription.

## Known issues

- **Two drum classifiers never run.** `V4: missing checkpoint or config` and
  `V17: ...`. The checkpoints download fine; `configs/onset_classifier_v4.yaml`
  and `_v17.yaml` **do not exist in the repo** — an upstream gap
  (`scripts/batch_infer_hybrid.py:198,208`). Running a 4-model ensemble, not 6.
- **6-stem drumsep skipped** — looks at hardcoded `/home/opria123/drumsep/`
  paths. That is the tom-vs-cymbal arbiter, so those calls are weaker.
- **Genre string mangles** multi-value ID3 frames:
  `Arena RockBluesBlues RockClassic Rock...`. Take the first value.
- Keys are off by default and untested.
- Whisper `medium` (~1.5 GB) is the default; `STRUM_WHISPER_MODEL=small` if
  memory is tight.

## Traps for the next session

- **Many repo files are CRLF** (`pyproject.toml`, `README.md`,
  `src/preprocessing/*.py`, `scripts/vocals_charter.py`, `configs/*.yaml`).
  Python text-mode reads normalise them, so a naive `open(p,'w').write(s)`
  silently rewrites the whole file. Read and write with `newline=''`.
  `scripts/` and `docs/` are mixed — check per file.
- **`numpy<2` is load-bearing.** TensorFlow 2.15 (via basic-pitch) requires it,
  which pins `audio-separator` to 0.30.2 and rules out MSST (`peft`) and any
  audio-separator ≥0.42. This is why BS-RoFormer runs through ONNX.
- **Python is pinned to 3.11** — TensorFlow <2.15.1 has no cp312 wheels.
- **`setuptools<81`** — 81 removed `pkg_resources`, which dependencies still
  import; its absence surfaced as an unrelated failure deep in the pipeline.
- **`batch_pipeline.py` exits 0 even when every song fails.** Success is judged
  on `notes.mid` existing, not the return code.
- **Importing TensorFlow resets the root log level**, silencing everything the
  pipeline logs afterwards. Guarded in `scripts/guitar_basicpitch.py`; if logs
  go quiet mid-run again, suspect this first.
- Docker layers are ordered cheapest-invalidation-last: torch → heavy deps
  (`docker/requirements-heavy.txt`) → project → sources. Editing `src/` rebuilds
  seconds, not gigabytes. Do not use `--no-cache` out of habit.
