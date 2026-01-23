# Multi-stage build for Streamlit frontend
# Optimized for Google Cloud Run deployment

FROM python:3.12-slim as builder

WORKDIR /app

# Install uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:$PATH"

# Copy project files
COPY pyproject.toml uv.lock ./

# Build dependencies
RUN uv sync --frozen --no-dev

# Final stage
FROM python:3.12-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 streamlit

# Copy virtual environment from builder
COPY --from=builder --chown=streamlit:streamlit /app/.venv /app/.venv

# Copy application code
COPY --chown=streamlit:streamlit src /app/src
COPY --chown=streamlit:streamlit configs /app/configs

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_LOGGER_LEVEL=warning

# Create cache directory
RUN mkdir -p /app/.streamlit && chown -R streamlit:streamlit /app

# Switch to non-root user
USER streamlit

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD streamlit config show 2>/dev/null || exit 1

# Run Streamlit app
EXPOSE 8080
CMD ["streamlit", "run", "src/customer_support/frontend.py"]
