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
FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ffmpeg does the video demux and every audio encode; libsndfile backs soundfile.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# The image ships pip 24.0, which skips wheels whose metadata it cannot parse
# and backtracks instead. On scikit-learn that lands on a pre-wheel release that
# builds via the removed numpy.distutils. Upgrading first keeps the resolver on
# the same versions the dependency set was verified against.
RUN pip install --upgrade pip setuptools wheel

# Torch first, from the CUDA index, so the dependency resolver below cannot
# quietly pull the CPU-only build. TensorFlow arrives later as a basic-pitch
# dependency but keeps its own CUDA libraries behind an `and-cuda` extra we
# never request, so it cannot displace these.
# torchvision is included even though nothing here uses it directly: it arrives
# transitively via audio-separator -> onnx2torch, and letting pip pull it from
# PyPI afterwards would drag the CPU-only torch build in with it.
RUN pip install --index-url https://download.pytorch.org/whl/cu126 \
        torch torchaudio torchvision

COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs

# diffq is a hard audio-separator dependency on Linux and publishes no cp311
# wheel at any version, so it always compiles from source. The toolchain is
# installed and removed inside one layer so it never reaches the final image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && pip install -e '.[full,web]' \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

# Weights and job data are mounted, not baked: the checkpoints are large and
# licensed separately, and job output should outlive the container.
VOLUME ["/data", "/app/checkpoints", "/models"]

ENV STRUM_WEB_HOST=0.0.0.0 \
    STRUM_WEB_PORT=8000 \
    STRUM_WEB_DATA_DIR=/data \
    STRUM_WEB_MAX_UPLOAD_MB=300 \
    STRUM_WEB_RETAIN_HOURS=24 \
    STRUM_MSST_DIR=/models/msst \
    XDG_CACHE_HOME=/models/cache

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${STRUM_WEB_PORT}/api/config" || exit 1

# Runs as root by default so mounted volumes stay writable regardless of host
# ownership; override with `user:` in compose if you would rather not.
CMD ["strum-web"]
