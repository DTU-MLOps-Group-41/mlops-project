FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

# Install UV from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

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
