from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()

from mcp.server.fastmcp import FastMCP  # noqa: E402

from retrieval.orchestrator import HybridSearchOrchestrator  # noqa: E402

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("mcp_server")

_orchestrator: HybridSearchOrchestrator | None = None


def _get_orchestrator() -> HybridSearchOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = HybridSearchOrchestrator()
    return _orchestrator


async def _warmup_models() -> None:
    logger.info("Pre-loading ML models (reranker + dense embedder)...")
    orch = _get_orchestrator()
    await orch.warmup()


HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
MCP_PATH = os.getenv("MCP_PATH", "/mcp")


@asynccontextmanager
async def lifespan(server: FastMCP):
    await _warmup_models()
    yield


mcp = FastMCP(
    "hybrid-rag-engine",
    host=HOST,
    port=PORT,
    streamable_http_path=MCP_PATH,
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    stateless_http=True,
    json_response=True,
    lifespan=lifespan,
)


def _build_json_response(
    search_results: List[Any],
    query: str,
    filter_tags: List[str] | None,
) -> Dict[str, Any]:
    results_list: List[Dict[str, Any]] = []
    for i, result in enumerate(search_results, start=1):
        chunk = getattr(result, "chunk", result)
        relevance_score = getattr(result, "relevance_score", None)
        rrqf_score = getattr(result, "rrqf_score", None)

        entry: Dict[str, Any] = {
            "rank": i,
            "source_type": getattr(chunk, "source_type", ""),
            "source_id": getattr(chunk, "source_id", ""),
        }
        if relevance_score is not None:
            entry["relevance_score"] = round(relevance_score, 4)
        if rrqf_score is not None:
            entry["rrqf_score"] = round(rrqf_score, 4)
        results_list.append(entry)

    return {
        "query": query,
        "filter_tags": filter_tags,
        "result_count": len(results_list),
        "results": results_list,
    }


@mcp.tool(
    description=(
        "Execute a dual-vector hybrid search across the domain-agnostic RAG "
        "knowledge base. This tool triggers both dense (semantic) and sparse "
        "(keyword) retrieval paths in parallel, then mathematically fuses the "
        "results and applies a cross-encoder reranker for contextual precision.\n\n"
        "**Query construction guidance:**\n"
        "- For broad conceptual topics, use natural-language phrases that capture "
        "the semantic meaning (e.g. 'error handling patterns in kernel modules').\n"
        "- For precise term lookups, include exact technical keywords alongside "
        "the description (e.g. 'memory allocation kmalloc GFP_KERNEL slab').\n"
        "- Combining conceptual framing with specific tokens yields the best "
        "dual-vector results because both index types contribute to the fused ranking.\n\n"
        "**Filtering:** Provide *filter_tags* to restrict the search to specific "
        "hierarchical metadata scopes (e.g. ['retail', 'food'] or "
        "['linux_kernel', 'drivers']). When omitted the search spans all indexed "
        "domains.\n\n"
        "**Result limit:** Defaults to 5. Raise for broader exploration, lower "
        "for quick fact lookups."
    ),
)
async def query_hybrid_engine(
    query: str,
    filter_tags: List[str] | None = None,
    limit: int = 5,
) -> Dict[str, Any]:
    logger.info(
        "Tool call: query='%s' filter_tags=%s limit=%d",
        query,
        filter_tags,
        limit,
    )

    try:
        orch = _get_orchestrator()
        results = await orch.query(
            query=query,
            filter_tags=filter_tags,
            limit=limit,
        )
        return _build_json_response(results, query, filter_tags)

    except TimeoutError:
        logger.error("Query timed out for '%s'", query)
        return {
            "query": query,
            "filter_tags": filter_tags,
            "result_count": 0,
            "error": "The search timed out. The knowledge base may be under heavy load "
                     "or the query may be too broad. Try narrowing the scope with "
                     "filter_tags or simplifying the query text.",
            "results": [],
        }

    except ValueError as exc:
        logger.warning("Invalid arguments for query '%s': %s", query, exc)
        return {
            "query": query,
            "filter_tags": filter_tags,
            "result_count": 0,
            "error": f"Invalid request: {exc}. Ensure query is a non-empty string "
                     f"and limit is a positive integer.",
            "results": [],
        }

    except RuntimeError as exc:
        logger.exception("Infrastructure failure for query '%s'", query)
        return {
            "query": query,
            "filter_tags": filter_tags,
            "result_count": 0,
            "error": f"An internal pipeline error occurred: {exc}",
            "results": [],
        }

    except Exception:
        logger.exception("Unexpected error for query '%s'", query)
        return {
            "query": query,
            "filter_tags": filter_tags,
            "result_count": 0,
            "error": "An unexpected error occurred. Check server logs for details.",
            "results": [],
        }


@mcp.tool(
    description=(
        "Re-execute the full data extraction and ingestion pipeline. Useful when "
        "source CSV files have been updated and you want to refresh the knowledge "
        "base without restarting the server.\n\n"
        "This runs asynchronously in the background. The tool returns immediately "
        "with a confirmation; monitor server logs for progress.\n\n"
        "Optional parameters let you override default CSV paths from the environment."
    ),
)
async def refresh_data(
    articles_csv: str | None = None,
    categories_csv: str | None = None,
) -> str:
    logger.info("Tool call: refresh_data articles_csv=%s categories_csv=%s",
                 articles_csv, categories_csv)
    asyncio.create_task(_run_background_ingestion(articles_csv, categories_csv))
    return "Ingestion started in background. Check server logs for progress."


async def _run_background_ingestion(
    articles_csv: str | None,
    categories_csv: str | None,
) -> None:
    import json
    import time
    from pathlib import Path
    from typing import List

    from core.models import UnifiedChunk
    from extractors.retail_extractor import RetailDataExtractor
    from pipeline.ingestion_worker import IngestionWorker

    RETAIL_ARTICLES_CSV = articles_csv or os.getenv(
        "RETAIL_ARTICLES_CSV",
        str(Path.home() / "raw_data" / "d_articulos.csv"),
    )
    RETAIL_CATEGORIES_CSV = categories_csv or os.getenv(
        "RETAIL_CATEGORIES_CSV",
        str(Path.home() / "raw_data" / "d_categorizacion_tbl.csv"),
    )
    RETAIL_ACTIVITY_FILTER = os.getenv("RETAIL_ACTIVITY_FILTER", "FOOD")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    OPENROUTER_EMBED_MODEL = os.getenv("OPENROUTER_EMBED_MODEL", "openai/text-embedding-3-small")
    DENSE_EMBED_PROVIDER = os.getenv("DENSE_EMBED_PROVIDER", "openrouter")
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "agnostic_rag_collection")
    BATCH_SIZE = int(os.getenv("INGEST_BATCH_SIZE", "64"))
    INTER_BATCH_DELAY = float(os.getenv("INTER_BATCH_DELAY", "0.2"))
    CHECKPOINT_FILE = os.getenv("INGEST_CHECKPOINT_FILE", ".ingest_checkpoint.json")
    CHUNKS_CACHE_FILE = os.getenv("CHUNKS_CACHE_FILE", ".ingest_chunks_cache.jsonl")
    DENSE_VECTOR_DIM = int(os.getenv("DENSE_VECTOR_DIM", "1536"))

    t0 = time.monotonic()
    try:
        logger.info("Background ingestion: extracting from %s ...", RETAIL_ARTICLES_CSV)
        extractor = RetailDataExtractor(
            categories_csv=RETAIL_CATEGORIES_CSV,
            activity_filter=RETAIL_ACTIVITY_FILTER,
        )
        chunks: List[UnifiedChunk] = extractor.extract_chunks(RETAIL_ARTICLES_CSV)
        logger.info("Background ingestion: extracted %d chunks in %.1fs",
                     len(chunks), time.monotonic() - t0)

        cache_path = Path(CHUNKS_CACHE_FILE)
        with cache_path.open("w", encoding="utf-8") as fh:
            for chunk in chunks:
                obj = chunk.model_dump(mode="json")
                obj["_cache_fingerprint"] = chunk.metadata.file_path_or_url
                fh.write(json.dumps(obj) + "\n")
        logger.info("Background ingestion: saved %d chunks to cache", len(chunks))

        worker = IngestionWorker(
            qdrant_host=QDRANT_HOST,
            qdrant_port=QDRANT_PORT,
            collection_name=QDRANT_COLLECTION,
            batch_size=BATCH_SIZE,
            checkpoint_file=CHECKPOINT_FILE,
            inter_batch_delay=INTER_BATCH_DELAY,
            dense_dim=DENSE_VECTOR_DIM,
            dense_provider=DENSE_EMBED_PROVIDER,
            lite_llm_base_url=OPENROUTER_BASE_URL,
            lite_llm_api_key=OPENROUTER_API_KEY,
            embedding_model=OPENROUTER_EMBED_MODEL,
        )
        worker.clear_checkpoint()
        await worker.ingest_from_cache(CHUNKS_CACHE_FILE)
        await worker.close()

        logger.info("Background ingestion completed in %.1fs", time.monotonic() - t0)
    except Exception as exc:
        logger.exception("Background ingestion failed after %.1fs: %s",
                          time.monotonic() - t0, exc)


if __name__ == "__main__":
    logger.info(
        "Starting hybrid-rag MCP server on %s:%d%s (transport=streamable-http)",
        HOST,
        PORT,
        MCP_PATH,
    )
    mcp.run(transport="streamable-http")
