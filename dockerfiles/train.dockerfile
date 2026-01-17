FROM ghcr.io/astral-sh/uv:python3.12-alpine@sha256:ebdbfba8b5668276ab6d3a0e98e915c1b0c0a0007ab7f803c62df6cf93362c5d AS base

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

RUN uv sync --frozen --no-install-project

COPY src src/

RUN uv sync --frozen

ENTRYPOINT ["uv", "run", "src/customer_support/train.py"]
