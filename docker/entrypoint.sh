#!/usr/bin/env bash
# Container entrypoint: make sure the charting weights are present, then start
# whatever command the image was given.
#
# The checkpoints live on a writable volume rather than in the image: they are
# 1.8 GB, licensed separately, and worth keeping across rebuilds. The trade-off
# is that a fresh volume starts empty, so the first boot has to fill it.
set -euo pipefail

CHECKPOINT_DIR="${STRUM_CHECKPOINT_DIR:-/app/checkpoints}"
FETCH="${STRUM_FETCH_CHECKPOINTS:-1}"

if [ "$FETCH" = "1" ] || [ "$FETCH" = "true" ]; then
    if [ ! -w "$CHECKPOINT_DIR" ]; then
        echo "entrypoint: $CHECKPOINT_DIR is not writable; skipping weight download." >&2
        echo "entrypoint: mount it read-write, or pre-populate it on the host." >&2
    else
        echo "entrypoint: checking model weights in $CHECKPOINT_DIR"
        # The script skips files that already exist, so the steady-state cost of
        # this is one directory listing. Only a fresh volume actually downloads.
        if ! python /app/scripts/fetch_checkpoints.py; then
            echo "entrypoint: weight download failed. The server will still start," >&2
            echo "entrypoint: but charting jobs will fail until it succeeds." >&2
            echo "entrypoint: retry with: docker compose restart strum-web" >&2
        fi
    fi
else
    echo "entrypoint: STRUM_FETCH_CHECKPOINTS=$FETCH, skipping weight download."
fi

exec "$@"
