FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ARG DEVICE="cpu"
WORKDIR /app

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY LICENSE LICENSE
COPY src/ src/
COPY data/ data/

# Set UV link mode to copy to avoid symlink issues with cache mounts
ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv uv sync --no-dev --extra $DEVICE --locked --no-install-project

# Install project (2-step for better cacheability)
RUN uv pip install --no-deps -e .

EXPOSE 8080

ENTRYPOINT ["uv", "run", "--no-sync", "uvicorn", "src.customer_support.api:app", "--host", "0.0.0.0", "--port", "8080"]
