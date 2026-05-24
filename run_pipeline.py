"""
Validation Runner — Phase 2 (Retail Edition)
============================================
Lightweight end-to-end smoke test that exercises:

1. ``RetailDataExtractor`` on the real retail CSV dump.
2. ``IngestionWorker`` batching, mock embeddings, and Qdrant upsert.

Run with:
    uv run python run_pipeline.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from core.models import UnifiedChunk
from extractors.retail_extractor import RetailDataExtractor
from pipeline.ingestion_worker import IngestionWorker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

RAW_DATA = Path.home() / "raw_data"


async def main() -> int:
    logger.info("Phase 2 validation (retail) started.")

    articles_csv = RAW_DATA / "d_articulos.csv"
    categories_csv = RAW_DATA / "d_categorizacion_tbl.csv"

    if not articles_csv.exists():
        logger.error("Articles CSV not found: %s", articles_csv)
        return 1
    if not categories_csv.exists():
        logger.error("Categories CSV not found: %s", categories_csv)
        return 1

    extractor = RetailDataExtractor(
        categories_csv=str(categories_csv),
        activity_filter="FOOD",
    )
    chunks: list[UnifiedChunk] = extractor.extract_chunks(str(articles_csv))

    logger.info("Extracted %d chunk(s) from retail CSV.", len(chunks))

    for i, c in enumerate(chunks[:5]):
        logger.info(
            "  → %s | %s | tags=%s",
            c.source_type,
            c.source_id,
            c.metadata.hierarchical_tags,
        )
    if len(chunks) > 5:
        logger.info("  ... and %d more.", len(chunks) - 5)

    worker = IngestionWorker(
        batch_size=64,
        dense_dim=1536,
        dense_embed_fn=IngestionWorker.mock_dense_fn(dim=1536),
    )

    try:
        await worker.ingest(chunks)

        count_result = await worker._qdrant.count(
            collection_name=worker.collection_name
        )
        logger.info(
            "Qdrant collection '%s' now holds ~%d point(s).",
            worker.collection_name,
            count_result.count,
        )
        if count_result.count <= 0:
            logger.error("Collection appears empty after ingestion.")
            return 1
    except Exception as exc:
        logger.error("Ingestion pipeline failed: %s", exc, exc_info=True)
        return 1
    finally:
        await worker.close()

    logger.info("Phase 2 validation PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
