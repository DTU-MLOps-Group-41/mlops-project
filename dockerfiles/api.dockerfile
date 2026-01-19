FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# 1. Environment variables
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

ARG DEVICE="cpu"
WORKDIR /app

# 2. Install dependencies ONLY (Best for Caching)
# We only copy the files needed to resolve dependencies.
COPY uv.lock pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --extra $DEVICE

# 3. Copy the rest of the project
# Now, changes to src/ won't trigger the heavy dependency sync above.
COPY README.md LICENSE ./
COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/

# 4. Install the project itself
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra $DEVICE

# 5. Security: Run as non-root user
RUN groupadd -r appuser && useradd -r -m -g appuser appuser
USER appuser

EXPOSE 8080

# 6. Entrypoint (Simplified since .venv/bin is in PATH)
ENTRYPOINT ["uvicorn", "customer_support.api:app", "--host", "0.0.0.0", "--port", "8080"]
