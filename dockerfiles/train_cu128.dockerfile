FROM nvidia/cuda:12.9.1-runtime-ubuntu24.04@sha256:1287141d283b8f06f45681b56a48a85791398c615888b1f96bfb9fc981392d98

# Install UV from the official image
COPY --from=ghcr.io/astral-sh/uv:latest@sha256:94a23af2d50e97b87b522d3cea24aaf8a1faedec1344c952767434f69585cbf9 /uv /uvx /bin/

ENV UV_PYTHON=python3.12

# 1. Standardize the environment
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

ARG DEVICE="cu128"
WORKDIR /app

# 2. Cache Dependencies (The "Structured" way)
COPY uv.lock pyproject.toml README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --extra $DEVICE

# 3. Copy source code last
COPY src/ ./src/
COPY configs/ ./configs/
COPY data.dvc ./data.dvc
COPY .dvc ./.dvc/

# 4. Copy entrypoint script (dvc pull happens at runtime when GCS credentials are available)
COPY dockerfiles/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# 5. Final Sync/Install
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra $DEVICE

ENTRYPOINT ["/entrypoint.sh"]
