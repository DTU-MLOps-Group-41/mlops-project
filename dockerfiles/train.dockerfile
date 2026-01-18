FROM ghcr.io/astral-sh/uv:python3.12-bookworm@sha256:2260761e85882bf515f79c99ac967dcff56dd4ef0905476ef77920d7a42ec636

WORKDIR /app

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

RUN uv sync --frozen --no-install-project

COPY README.md README.md
COPY LICENSE LICENSE
COPY src src/
COPY data/ data/

RUN uv sync --frozen

ENTRYPOINT ["uv", "run","python", "src/customer_support/train.py"]
