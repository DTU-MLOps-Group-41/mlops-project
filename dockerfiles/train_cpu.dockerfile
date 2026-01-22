FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# 1. Standardize the environment
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

ARG DEVICE="cpu"
WORKDIR /app

# 2. Cache Dependencies (The "Structured" way)
COPY uv.lock pyproject.toml README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --extra $DEVICE

# 3. Copy source code last
COPY src/ ./src/
COPY configs/ ./configs/
# Data should mounted during runtime

# 4. Final Sync/Install
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra $DEVICE

# No 'uv run' needed because .venv/bin is in PATH
ENTRYPOINT ["python", "src/customer_support/train.py"]
