FROM ghcr.io/astral-sh/uv:python3.12-bookworm

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

RUN uv sync --frozen --no-install-project

COPY README.md README.md
COPY LICENSE LICENSE
COPY src src/

COPY data/ data/

RUN uv sync --frozen

EXPOSE 8080

ENTRYPOINT ["uv", "run", "uvicorn", "src.customer_support.api:app", "--host", "0.0.0.0", "--port", "8080"]
