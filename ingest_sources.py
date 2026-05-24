#!/usr/bin/env python3
"""
Production Ingestion Script — Retail Edition
=============================================
Reads the retail CSV dump (articles + category hierarchy), extracts
``UnifiedChunk`` objects via ``RetailDataExtractor``, and bulk-ingests
them into Qdrant using OpenRouter embeddings.

Features
--------
* Single-pass CSV extraction (no per-file concurrency needed)
* Version deduplication (latest version per codart kept)
* Checkpoint-resume ingestion with text deduplication
* Detailed extraction stats

Usage:  uv run python ingest_sources.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from core.models import UnifiedChunk
from extractors.retail_extractor import RetailDataExtractor
from pipeline.ingestion_worker import IngestionWorker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------
RETAIL_ARTICLES_CSV = os.getenv(
    "RETAIL_ARTICLES_CSV",
    str(Path.home() / "raw_data" / "d_articulos.csv"),
)
RETAIL_CATEGORIES_CSV = os.getenv(
    "RETAIL_CATEGORIES_CSV",
    str(Path.home() / "raw_data" / "d_categorizacion_tbl.csv"),
)
RETAIL_ACTIVITY_FILTER = os.getenv("RETAIL_ACTIVITY_FILTER", "FOOD")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
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
FRESH_START = os.getenv("FRESH_START", "false").lower() in ("1", "true", "yes")
DENSE_VECTOR_DIM = int(os.getenv("DENSE_VECTOR_DIM", "1536"))


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_chunks() -> List[UnifiedChunk]:
    logger.info(
        "Extracting chunks from %s (categories: %s, activity: %s) ...",
        RETAIL_ARTICLES_CSV,
        RETAIL_CATEGORIES_CSV,
        RETAIL_ACTIVITY_FILTER,
    )
    t0 = time.monotonic()

    extractor = RetailDataExtractor(
        categories_csv=RETAIL_CATEGORIES_CSV,
        activity_filter=RETAIL_ACTIVITY_FILTER,
    )
    chunks = extractor.extract_chunks(RETAIL_ARTICLES_CSV)

    elapsed = time.monotonic() - t0
    logger.info(
        "Extraction complete in %.1f s — %d chunk(s) produced (%.0f chunks/s).",
        elapsed,
        len(chunks),
        len(chunks) / elapsed if elapsed > 0 else 0,
    )
    return chunks


def save_chunks_cache(chunks: List[UnifiedChunk], cache_path: str) -> None:
    path = Path(cache_path)
    with path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            obj = chunk.model_dump(mode="json")
            obj["_cache_fingerprint"] = chunk.metadata.file_path_or_url
            fh.write(json.dumps(obj) + "\n")
    logger.info("Saved %d chunk(s) to %s.", len(chunks), cache_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> int:
    articles_path = Path(RETAIL_ARTICLES_CSV)
    if not articles_path.exists():
        logger.error("Retail articles CSV not found: %s", articles_path)
        return 1

    categories_path = Path(RETAIL_CATEGORIES_CSV)
    if not categories_path.exists():
        logger.error("Retail categories CSV not found: %s", categories_path)
        return 1

    if not OPENROUTER_API_KEY:
        logger.error(
            "OPENROUTER_API_KEY is not set. Please add your key to .env."
        )
        return 1

    chunks = extract_chunks()
    if not chunks:
        logger.warning("No chunks produced — nothing to ingest.")
        return 0

    save_chunks_cache(chunks, CHUNKS_CACHE_FILE)

    logger.info(
        "Using %s embeddings (model: %s).",
        DENSE_EMBED_PROVIDER,
        OPENROUTER_EMBED_MODEL,
    )
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

    if FRESH_START:
        logger.info("FRESH_START=true — clearing checkpoint.")
        worker.clear_checkpoint()

    try:
        await worker.ingest_from_cache(CHUNKS_CACHE_FILE)
    except Exception as exc:
        logger.error("Ingestion failed: %s", exc, exc_info=True)
        return 1
    finally:
        await worker.close()

    logger.info("Ingestion finished successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
