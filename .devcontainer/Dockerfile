FROM --platform=linux/amd64 ghcr.io/prefix-dev/pixi AS pixi

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git
