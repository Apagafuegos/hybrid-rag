"""
Collection Initialization Script
================================
Idempotent bootstrapper for the `agnostic_rag_collection` in Qdrant.

Run this *after* the Qdrant container is healthy:
    uv run python core/setup_db.py
"""

from __future__ import annotations

import logging
import os
import sys

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    SparseVectorParams,
    VectorParams,
)

COLLECTION_NAME = "agnostic_rag_collection"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
DENSE_DIM = int(os.getenv("DENSE_VECTOR_DIM", "1536"))
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def get_client() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def setup_collection(client: QdrantClient) -> None:
    if client.collection_exists(COLLECTION_NAME):
        logger.info("Collection '%s' already exists — skipping creation.", COLLECTION_NAME)
        return

    logger.info("Creating collection '%s' ...", COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            DENSE_VECTOR_NAME: VectorParams(
                size=DENSE_DIM,
                distance=Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: SparseVectorParams(),
        },
        quantization_config=ScalarQuantization(
            scalar=ScalarQuantizationConfig(
                type=ScalarType.INT8,
                always_ram=True,
            )
        ),
    )

    logger.info("Collection '%s' created successfully.", COLLECTION_NAME)


def main() -> int:
    try:
        client = get_client()
        setup_collection(client)
        logger.info("Database setup complete.")
        return 0
    except Exception as exc:
        logger.error("Database setup failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
