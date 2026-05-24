#!/bin/bash
set -e

echo "=== Hybrid RAG Startup ==="

echo "Waiting for Qdrant..."
python -c "
import time, os
from qdrant_client import QdrantClient
host = os.getenv('QDRANT_HOST', 'qdrant')
port = int(os.getenv('QDRANT_PORT', '6333'))
for i in range(30):
    try:
        c = QdrantClient(host=host, port=port)
        c.get_collections()
        print('Qdrant ready.')
        break
    except Exception:
        print(f'  waiting... ({i+1}/30)')
        time.sleep(2)
else:
    print('Qdrant not available after 60s')
    exit(1)
"

echo "Setting up Qdrant collection..."
uv run python core/setup_db.py

if [ -f "$RETAIL_ARTICLES_CSV" ]; then
    # Check if collection already has data
    POINTS_COUNT=$(python -c "
import os
from qdrant_client import QdrantClient
host = os.getenv('QDRANT_HOST', 'qdrant')
port = int(os.getenv('QDRANT_PORT', '6333'))
c = QdrantClient(host=host, port=port)
try:
    info = c.get_collection('agnostic_rag_collection')
    print(info.points_count)
except Exception:
    print('0')
")
    if [ "$POINTS_COUNT" -gt "0" ]; then
        echo "Collection already has $POINTS_COUNT points — skipping ingestion."
    else
        echo "Ingesting retail data from $RETAIL_ARTICLES_CSV ..."
        uv run python ingest_sources.py || echo "Ingestion finished (may have warnings)."
    fi
else
    echo "No CSV found at $RETAIL_ARTICLES_CSV — skipping ingestion."
fi

echo "Starting MCP server..."
exec uv run python -m mcp_server.server
