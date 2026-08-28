# syntax=docker/dockerfile:1.7
#
# STRUM auto-charter web UI.
#
# Python is pinned to 3.11 and must stay there: basic-pitch requires
# tensorflow<2.15.1, and TensorFlow published no cp312 wheels until 2.16. On
# 3.12 the two constraints have no solution and the install fails outright.
#
# There is no CUDA base image here on purpose. The torch cu126 wheels bundle
# the CUDA runtime and cuDNN they need, so the host requirement is unchanged --
# NVIDIA driver plus nvidia-container-toolkit -- while the image loses several
# GB of duplicated CUDA layers.
#
# Separation and transcription are GPU-bound; the image still runs without
# --gpus, but a 4-minute song takes hours on CPU.
#
# Downloads are held in BuildKit cache mounts rather than in the layers
# themselves. Those mounts are not part of the image and survive `--no-cache`,
# so a rebuild re-runs the install steps but does not re-fetch the ~4 GB of
# wheels. Only `docker builder prune` clears them.
FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# The base image ships a config that deletes downloaded .deb files after every
# install. That defeats the apt cache mounts below.
RUN rm -f /etc/apt/apt.conf.d/docker-clean \
    && echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' \
        > /etc/apt/apt.conf.d/keep-cache

# ffmpeg does the video demux and every audio encode; libsndfile backs soundfile.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 git curl ca-certificates

WORKDIR /app

# Two separate fixes in one line.
#
# pip: the base image ships 24.0, which skips wheels whose metadata it cannot
# parse and backtracks instead. On scikit-learn that lands on a pre-wheel
# release that builds through the removed numpy.distutils.
#
# setuptools: pinned below 81, where pkg_resources was removed. Several
# dependencies still import it, and its absence surfaces as unrelated failures
# deep in the pipeline rather than as an install error.
#
# This layer is upstream of every install below, so changing it rebuilds all of
# them. The cache mounts still spare the downloads.
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install --upgrade pip wheel "setuptools<81"

# Torch first, from the CUDA index, so the dependency resolver below cannot
# quietly pull the CPU-only build. TensorFlow arrives later as a basic-pitch
# dependency but keeps its own CUDA libraries behind an `and-cuda` extra we
# never request, so it cannot displace these.
# torchvision is included even though nothing here uses it directly: it arrives
# transitively via audio-separator -> onnx2torch, and letting pip pull it from
# PyPI afterwards would drag the CPU-only torch build in with it.
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install --index-url https://download.pytorch.org/whl/cu126 \
        torch torchaudio torchvision

# The heavy dependencies go in first, keyed only on their own file. Installing
# them as part of the project below would mean every pyproject.toml edit
# reinstalled several GB. See the file for what belongs in it.
#
# diffq is a hard audio-separator dependency on Linux and publishes no cp311
# wheel at any version, so it always compiles from source. The toolchain is
# installed and removed inside one layer so it never reaches the final image.
COPY docker/requirements-heavy.txt ./docker/
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && pip install -r docker/requirements-heavy.txt \
    && apt-get purge -y --auto-remove build-essential

# The project itself, resolved against a stub package so that editing anything
# under src/ cannot invalidate this layer -- the real sources are copied in
# afterwards and the editable install points at the same path either way.
# Everything expensive is already satisfied by the layer above, so this is fast.
COPY pyproject.toml README.md ./
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    mkdir -p src && touch src/__init__.py \
    && pip install -e '.[full,web]'

COPY src ./src
COPY scripts ./scripts
COPY configs ./configs

# Weights and job data are mounted, not baked: the checkpoints are large and
# licensed separately, and job output should outlive the container. The
# checkpoint volume is writable because the app fills it on first run -- nothing
# is downloaded at build time.
VOLUME ["/data", "/app/checkpoints", "/models"]

ENV STRUM_WEB_HOST=0.0.0.0 \
    STRUM_WEB_PORT=8000 \
    STRUM_WEB_DATA_DIR=/data \
    STRUM_WEB_MAX_UPLOAD_MB=300 \
    STRUM_WEB_RETAIN_HOURS=24 \
    STRUM_CHECKPOINT_DIR=/app/checkpoints \
    STRUM_FETCH_CHECKPOINTS=1 \
    STRUM_MSST_DIR=/models/msst \
    STRUM_SEPARATION_DIR=/models \
    STRUM_KARAOKE_MODEL_DIR=/models/audio-separator \
    XDG_CACHE_HOME=/models/cache

EXPOSE 8000

# The server binds immediately and downloads weights in the background, so it is
# healthy well before the 1.8 GB first-run fetch completes.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${STRUM_WEB_PORT}/api/config" || exit 1

# Runs as root by default so mounted volumes stay writable regardless of host
# ownership; override with `user:` in compose if you would rather not.
CMD ["strum-web"]
