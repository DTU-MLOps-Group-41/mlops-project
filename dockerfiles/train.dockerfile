FROM ghcr.io/astral-sh/uv:python3.12-alpine@sha256:68a899a90e077a6c0da311f268a9f05ee0f4984927761b576574c9548f469ead AS base

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

RUN uv sync --frozen --no-install-project

COPY src src/

RUN uv sync --frozen

ENTRYPOINT ["uv", "run", "src/customer_support/train.py"]
