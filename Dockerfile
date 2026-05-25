FROM python:3.12-slim

RUN apt-get update -qq \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock* ./
COPY core/        ./core/
COPY extractors/  ./extractors/
COPY pipeline/    ./pipeline/
COPY retrieval/   ./retrieval/
COPY mcp_server/  ./mcp_server/
COPY ingest_sources.py docker_entrypoint.sh ./

RUN uv sync --frozen --no-dev && \
    rm -rf /root/.cache/uv /root/.cache/pip /tmp/* && \
    chmod +x /app/docker_entrypoint.sh

ENV HOST=0.0.0.0
ENV PORT=8000
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

HEALTHCHECK --interval=120s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1

ENTRYPOINT ["/bin/bash", "/app/docker_entrypoint.sh"]
