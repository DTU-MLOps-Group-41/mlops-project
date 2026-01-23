FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# 1. Environment variables
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/app/.cache/huggingface

ARG DEVICE="cpu"
WORKDIR /app

# 2. Install dependencies ONLY (Best for Caching)
# We only copy the files needed to resolve dependencies.
COPY uv.lock pyproject.toml README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --extra $DEVICE

# 3. Copy the rest of the project
COPY src/ ./src/
RUN mkdir -p /app/artifacts && chmod 777 /app/artifacts

# 4. Install the project itself
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra $DEVICE

# 5. Pre-download HuggingFace model files to avoid rate limiting at runtime
RUN python -c "from transformers import DistilBertTokenizer, DistilBertForSequenceClassification; \
    DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased'); \
    DistilBertForSequenceClassification.from_pretrained('distilbert-base-multilingual-cased')"

# 6. Security: Run as non-root user
RUN groupadd -r appuser && useradd -r -m -g appuser appuser
USER appuser

EXPOSE 8080

# 7. Entrypoint (Simplified since .venv/bin is in PATH)
ENTRYPOINT ["uvicorn", "customer_support.api:app", "--host", "0.0.0.0", "--port", "8080"]
