# Hybrid RAG Engine

A production-grade, domain-agnostic Hybrid Retrieval-Augmented Generation (RAG) system built on **Hexagonal Architecture (Ports & Adapters)**. The core engine is completely blind to the nature of the data it processes — swap data domains (retail catalogs, Linux kernel source, mailing lists, medical text) by changing only the peripheral extractors.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         MCP Server Layer                                 │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  FastMCP (streamable-http)  ──  query_hybrid_engine()            │    │
│  │                                   refresh_data()                 │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                              ▲                                           │
└──────────────────────────────┼───────────────────────────────────────────┘
                               │
┌──────────────────────────────┼───────────────────────────────────────────┐
│                         Phase 3: Retrieval Engine                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐  │
│  │ Local Search │  │ External     │  │ RRF Fusion   │  │ Cross-Enc.  │  │
│  │ (dense+sparse│  │ Search Port  │  │ (rank merge) │  │ Reranker    │  │
│  │  parallel)   │  │ (optional)   │  │              │  │             │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └─────────────┘  │
│         ▲                                                        │      │
│         │                                                        │      │
│  ┌──────┴─────────────────────────────────────────────────────────┘      │
│  │                    HybridSearchOrchestrator                            │
│  └───────────────────────────────────────────────────────────────────────┘
└─────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Phase 2: Ingestion Pipeline                      │
│  ┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐   │
│  │ DataExtractor   │───▶│ IngestionWorker  │───▶│   Qdrant         │   │
│  │ (domain-specific│    │ (batch embed +   │    │ (dual-vector     │   │
│  │  plug-in)       │    │  bulk upsert)    │    │  storage)        │   │
│  └─────────────────┘    └──────────────────┘    └──────────────────┘   │
│         ▲                                                                │
│         │                                                                │
│  RetailExtractor  (current default)                                      │
│  LinuxCodeExtractor  (Tree-sitter, available)                           │
│  MailingListExtractor (.mbox parser, available)                         │
└─────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Phase 1: Storage Infrastructure                  │
│                          Qdrant (Docker)                                 │
│         Dense vectors (cosine, HNSW on-disk) + int8 quantization         │
│         Sparse vectors (inverted index, keyword tokens)                  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Core Invariant: The `UnifiedChunk`

Every layer communicates via this immutable schema. **Do not modify it** for domain-specific data — use `metadata.custom_attributes` instead.

```json
{
  "id": "uuid-v4",
  "text_content": "Raw text fed to embeddings and LLM context",
  "source_type": "retail_product | linux_kernel | lkml_email | ...",
  "source_id": "Unique upstream identifier",
  "sparse_tokens": {
    "tokens": ["keyword", "list", "for", "sparse", "retrieval"]
  },
  "metadata": {
    "hierarchical_tags": ["retail", "food", "lacteos", "leche"],
    "parent_structure": "Optional structural namespace",
    "file_path_or_url": "csv://retail_catalog/12345",
    "custom_attributes": {
      "codart": "12345",
      "desart": "Leche entera 1L",
      "category_chain": "SUPERMERCADO > Frescos > Lácteos > Leche > Entera"
    }
  }
}
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Vector DB** | Qdrant (dense + sparse dual storage, int8 scalar quantization, on-disk HNSW) |
| **Embeddings** | OpenRouter (OpenAI-compatible) or FastEmbed (local, free) |
| **Reranking** | Local cross-encoder (`BAAI/bge-reranker-base` via sentence-transformers) |
| **Code Parsing** | Tree-sitter (AST-aware chunking for C code) |
| **Protocol** | Model Context Protocol (MCP) via `FastMCP` over streamable HTTP |
| **Runtime** | Python 3.12, `uv` for dependency management |
| **Deployment** | Docker Compose |

---

## Project Structure

```
.
├── core/
│   ├── models.py           # UnifiedChunk + UnifiedChunkMetadata (invariant contract)
│   └── setup_db.py         # Idempotent Qdrant collection bootstrap
│
├── extractors/
│   ├── base.py             # DataExtractor ABC (the "port")
│   ├── retail_extractor.py # Concrete: retail CSV (articles + category tree)
│   ├── code_extractor.py   # Concrete: Linux kernel C source (Tree-sitter)
│   └── mail_extractor.py   # Concrete: .mbox mailing list threads
│
├── pipeline/
│   └── ingestion_worker.py # Batch embed + dedup + upsert with checkpoint/resume
│
├── retrieval/
│   ├── local_search.py     # Parallel dense + sparse Qdrant queries
│   ├── external_search.py  # Optional live data fetch port
│   ├── fusion.py           # Reciprocal Rank Fusion (pure math)
│   ├── reranker.py         # Cross-encoder contextual scoring
│   ├── orchestrator.py     # Phase 3 controller chaining all stages
│   └── models.py           # SearchResult DTO
│
├── mcp_server/
│   └── server.py           # FastMCP server exposing hybrid_search + refresh tools
│
├── infra/
│   └── qdrant_config.yaml  # Production-optimized Qdrant settings (memmap, int8)
│
├── docker-compose.yml      # Qdrant + MCP server services
├── Dockerfile              # Multi-stage build for the MCP server
├── run_pipeline.py         # Standalone ingestion smoke test
├── ingest_sources.py       # One-shot ingestion script
├── test_retrieval_engine.py# Retrieval pipeline test harness
├── .env.example            # Environment variable template
└── pyproject.toml          # uv-based Python dependencies
```

---

## Quick Start

### 1. Prerequisites

- [Docker](https://docs.docker.com/get-docker/) & Docker Compose
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python package manager)
- An [OpenRouter](https://openrouter.ai/keys) API key (or use FastEmbed for local embeddings)

### 2. Environment Setup

```bash
cp .env.example .env
# Edit .env with your OpenRouter key and data paths
```

Key variables:

```bash
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_EMBED_MODEL=openai/text-embedding-3-small

# Or use local embeddings (no API key needed):
DENSE_EMBED_PROVIDER=fastembed
FASTEMBED_DENSE_MODEL=BAAI/bge-small-en-v1.5

QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=agnostic_rag_collection

# Retail data paths
RETAIL_ARTICLES_CSV=/home/ubuntu/raw_data/d_articulos.csv
RETAIL_CATEGORIES_CSV=/home/ubuntu/raw_data/d_categorizacion_tbl.csv
RETAIL_ACTIVITY_FILTER=FOOD
```

### 3. Start Qdrant

```bash
docker compose up -d qdrant
```

Qdrant will be available at `http://localhost:6333`.

### 4. Initialize the Collection

```bash
uv sync
uv run python core/setup_db.py
```

This creates the `agnostic_rag_collection` with:
- Dense vectors (1536-dim, cosine, HNSW on-disk)
- Sparse vectors (inverted index)
- `int8` scalar quantization for memory efficiency

### 5. Ingest Data

**Option A: Run the standalone pipeline**

```bash
uv run python run_pipeline.py
```

**Option B: Ingest via the MCP server (after starting it)**

Call the `refresh_data` tool or use the one-shot script:

```bash
uv run python ingest_sources.py
```

### 6. Test Retrieval

```bash
uv run python test_retrieval_engine.py
```

### 7. Start the MCP Server

```bash
uv run python mcp_server/server.py
```

The server exposes:
- **Tool**: `query_hybrid_engine(query, filter_tags, limit)` — hybrid search with dense + sparse + RRF + reranking
- **Tool**: `refresh_data(articles_csv, categories_csv)` — background re-ingestion

Endpoint: `http://localhost:8000/mcp`

---

## Docker Deployment

Build and run everything:

```bash
docker compose up -d --build
```

Services:
| Service | Port | Description |
|---------|------|-------------|
| `hybrid-rag-qdrant` | `6333` (HTTP), `6334` (gRPC) | Vector database |
| `hybrid-rag-mcp` | `8000` | MCP server with search + ingestion tools |

Volumes:
- `qdrant_data` — persistent vector storage
- `model_cache` — Hugging Face model cache (cross-encoder)
- `ingest_cache` — ingestion checkpoints and vocab files

---

## How It Works

### Phase 1: Dual-Vector Storage

Every chunk is stored with **two vectors**:

1. **Dense vector** — semantic embedding from an LLM (OpenRouter or FastEmbed). Captures meaning and conceptual similarity.
2. **Sparse vector** — keyword token indices from `sparse_tokens`. Captures exact term matches and rare technical vocabulary.

Qdrant indexes both simultaneously. Dense uses HNSW (on-disk to save RAM). Sparse uses an inverted index. Both are quantized to `int8`.

### Phase 2: Pluggable Extraction

Extractors implement `DataExtractor.extract_chunks(source_path) -> List[UnifiedChunk]`:

- **RetailExtractor**: Reads `d_articulos.csv` + `d_categorizacion_tbl.csv`, builds category chains, deduplicates by version, tokenizes descriptions.
- **LinuxCodeExtractor**: Uses Tree-sitter to chunk C code at function/struct boundaries.
- **MailingListExtractor**: Parses `.mbox` threads into conversational segments.

The `IngestionWorker` handles the rest: batching, embedding (with retry + exponential backoff), deduplication, checkpoint/resume, and bulk Qdrant upsert.

### Phase 3: Hybrid Retrieval Pipeline

For each query, the `HybridSearchOrchestrator` executes:

1. **Parallel Local Search**
   - Dense query: embed the query, search HNSW index
   - Sparse query: tokenize the query, search inverted index
   - Both respect `hierarchical_tags` filters

2. **External Fetch** (optional)
   - Live data from an external API/feed mapped into transient `UnifiedChunk` objects

3. **Reciprocal Rank Fusion (RRF)**
   - Merges all ranked lists with the formula: `score = Σ 1/(k + rank)` where `k=60`
   - Deduplicates by `chunk.id`

4. **Cross-Encoder Reranking**
   - Scores query→chunk pairs with a local transformer model
   - Returns the top-N most contextually relevant results

### Phase 4: MCP Protocol Delivery

The `FastMCP` server exposes generic tools. Any MCP client (Claude, Cursor, etc.) can call:

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "id": 1,
  "params": {
    "name": "query_hybrid_engine",
    "arguments": {
      "query": "leche entera sin lactosa",
      "filter_tags": ["retail", "food"],
      "limit": 5
    }
  }
}
```

---

## Adding a New Data Domain

The architecture enforces **zero domain awareness in the core**. To add a new domain:

1. **Create an extractor** in `extractors/my_domain_extractor.py`:

```python
from extractors.base import DataExtractor
from core.models import UnifiedChunk, UnifiedChunkMetadata

class MyDomainExtractor(DataExtractor):
    def extract_chunks(self, source_path: str) -> list[UnifiedChunk]:
        # Parse your raw data
        # Map each logical unit to UnifiedChunk
        return chunks
```

2. **Register it** in `ingest_sources.py` (or call it from your own script).

3. **Re-ingest**: `uv run python ingest_sources.py`

No changes needed in `core/`, `pipeline/`, `retrieval/`, or `mcp_server/`.

---

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | — | API key for OpenRouter embedding endpoint |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter base URL |
| `OPENROUTER_EMBED_MODEL` | `openai/text-embedding-3-small` | Embedding model name |
| `DENSE_EMBED_PROVIDER` | `openrouter` | `openrouter` or `fastembed` |
| `FASTEMBED_DENSE_MODEL` | `BAAI/bge-small-en-v1.5` | Local embedding model |
| `DENSE_VECTOR_DIM` | `1536` | Must match the embedding model output size |
| `QDRANT_HOST` | `localhost` | Qdrant hostname |
| `QDRANT_PORT` | `6333` | Qdrant HTTP port |
| `QDRANT_COLLECTION` | `agnostic_rag_collection` | Collection name |
| `INGEST_BATCH_SIZE` | `64` | Chunks per embedding batch |
| `INTER_BATCH_DELAY` | `0.2` | Seconds between batches (rate limiting) |
| `RERANK_MODEL` | `BAAI/bge-reranker-base` | Cross-encoder model |
| `LOG_LEVEL` | `INFO` | Server logging level |

---

## Memory & Performance Notes

- **Qdrant**: Configured for VPS deployment with `memmap_threshold_kb: 20480` and `on_disk: true` for HNSW graphs. Cold segments live on disk; hot working sets stay in page cache.
- **Embeddings**: FastEmbed runs entirely locally with no API calls. OpenRouter requires network but offloads compute.
- **Reranker**: Loaded lazily on first query, then cached. Runs on CPU by default. Set `device=cuda` in `reranker.py` if GPU is available.
- **Ingestion**: Streaming JSONL cache support means ingestion uses O(batch_size) memory regardless of total dataset size.

---

## Development

```bash
# Install dependencies
uv sync

# Run linting/type-checking (add your tools to pyproject.toml)
# uv run ruff check .
# uv run mypy .

# Run tests
# uv run pytest

# Format code
# uv run ruff format .
```

---

## License

MIT
