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
    rm -rf /app/.venv/lib/python*/site-packages/triton* && \
    rm -rf /root/.cache/uv /root/.cache/pip /tmp/* && \
    chmod +x /app/docker_entrypoint.sh

ENV HOST=0.0.0.0
ENV PORT=8000
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:$PATH"
ENV HF_HOME=/hf_cache
ENV SENTENCE_TRANSFORMERS_HOME=/hf_cache

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -sf -X POST http://localhost:8000/hybrid-mcp-server/mcp \
        -H "Content-Type: application/json" \
        -H "Accept: application/json" \
        -d '{"jsonrpc":"2.0","method":"tools/list","id":1}' \
        || exit 1

ENTRYPOINT ["/bin/bash", "/app/docker_entrypoint.sh"]
