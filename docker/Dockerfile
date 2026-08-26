# STRUM auto-charter web UI.
#
# CUDA runtime base: separation and transcription are GPU-bound, and the image
# is unusable in practice on CPU (a 4-minute song takes hours). It still runs
# without a GPU if you leave --gpus off; every model falls back to CPU.
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

# ffmpeg does the video demux and every audio encode; libsndfile backs soundfile.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv \
        ffmpeg libsndfile1 git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Torch first, from the CUDA index, so the dependency resolver below cannot
# quietly pull the CPU-only build.
RUN pip install --index-url https://download.pytorch.org/whl/cu126 \
        torch torchaudio

COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs

RUN pip install -e '.[full,web]'

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
