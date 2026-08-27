#!/usr/bin/env bash
# Build and start the STRUM web UI with BuildKit enabled.
#
# The Dockerfile uses cache mounts to hold ~4 GB of torch and CUDA wheels
# outside the image layers. Those need BuildKit. Some Docker installs still
# default to the legacy builder, which rejects the syntax with:
#
#     the --mount option requires BuildKit
#
# This wrapper forces the right builder rather than relying on the daemon's
# default. See docs/WEBGUI.md for enabling BuildKit permanently instead.
set -euo pipefail

export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

COMPOSE_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/docker-compose.yml"

if docker compose version >/dev/null 2>&1; then
    compose=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
    compose=(docker-compose)
else
    echo "Neither 'docker compose' nor 'docker-compose' is available." >&2
    exit 1
fi

# Default to build+start; pass any other compose subcommand through, e.g.
#   docker/build.sh build
#   docker/build.sh logs -f
if [ "$#" -eq 0 ]; then
    set -- up -d --build
fi

echo "==> ${compose[*]} -f $COMPOSE_FILE $*"
exec "${compose[@]}" -f "$COMPOSE_FILE" "$@"
