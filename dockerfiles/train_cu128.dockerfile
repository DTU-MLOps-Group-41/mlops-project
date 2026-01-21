FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

# Install UV from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 1. Standardize the environment
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

ARG DEVICE="gpu"
WORKDIR /app

# 2. Cache Dependencies (The "Structured" way)
COPY uv.lock pyproject.toml README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --extra $DEVICE

# 3. Copy source code last
COPY src/ ./src/
# Data should mounted during runtime

# 4. Final Sync/Install
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra $DEVICE

# No 'uv run' needed because .venv/bin is in PATH
ENTRYPOINT ["python", "src/customer_support/train.py"]
