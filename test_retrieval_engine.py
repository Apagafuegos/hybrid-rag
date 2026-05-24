"""
End-to-End Retrieval Engine Validation
========================================
Orchestrated test of the full Phase-3 pipeline:

    Local dual-vector search  →  External live fetch  →  RRF fusion  →  Cross-encoder rerank

Run this script after Qdrant is healthy:
    uv run python test_retrieval_engine.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
import uuid
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, ScalarQuantization, ScalarQuantizationConfig, ScalarType, SparseVectorParams, VectorParams

from core.models import UnifiedChunk, UnifiedChunkMetadata
from pipeline.ingestion_worker import IngestionWorker
from retrieval.external_search import MockExternalSearchProvider
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.local_search import LocalSearcher
from retrieval.models import SearchResult
from retrieval.reranker import CrossEncoderReranker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("test_retrieval_engine")

TEST_COLLECTION = "test_retrieval_engine_collection"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
DENSE_DIM = 1536


def build_seed_chunks() -> List[UnifiedChunk]:
    """Generate synthetic retail-product chunks that exercise hierarchical tags."""
    chunks: List[UnifiedChunk] = []

    products = [
        (
            "Leche entera fresca brick 1 litro. Familia: Lácteos. Sección: Frescos. "
            "Categoría: SUPERMERCADO > Frescos > Lácteos > Leche > Entera. Marca: Pascual.",
            ["retail", "food", "supermercado", "frescos", "lacteos", "leche", "entera"],
            "leche-entera-001",
        ),
        (
            "Nata para montar 200ml. Familia: Lácteos. Sección: Frescos. "
            "Categoría: SUPERMERCADO > Frescos > Lácteos > Natas. Marca: IFA.",
            ["retail", "food", "supermercado", "frescos", "lacteos", "natas"],
            "nata-montar-002",
        ),
        (
            "Leche desnatada sin lactosa brick 1 litro. Familia: Lácteos. Sección: Frescos. "
            "Categoría: SUPERMERCADO > Frescos > Lácteos > Leche > Desnatada. Marca: Kaiku.",
            ["retail", "food", "supermercado", "frescos", "lacteos", "leche", "desnatada"],
            "leche-desnatada-003",
        ),
        (
            "Aceite de oliva virgen extra botella 1 litro. Familia: Aceites. Sección: Alimentación. "
            "Categoría: SUPERMERCADO > Alimentación > Aceites y Condimentos > Aceites. Marca: Carbonell.",
            ["retail", "food", "supermercado", "alimentacion", "aceites"],
            "aceite-oliva-004",
        ),
        (
            "Refresco cola lata 33cl pack 6. Familia: Bebidas. Sección: Bebidas. "
            "Categoría: SUPERMERCADO > Bebidas > Refrescos > Cola. Marca: Coca-Cola.",
            ["retail", "food", "supermercado", "bebidas", "refrescos"],
            "refresco-cola-005",
        ),
        (
            "Pan de molde integral 500g. Familia: Panadería. Sección: Frescos. "
            "Categoría: SUPERMERCADO > Frescos > Panadería > Molde. Marca: Bimbo.",
            ["retail", "food", "supermercado", "frescos", "panaderia"],
            "pan-molde-006",
        ),
        (
            "Yogur natural pack 4 unidades. Familia: Lácteos. Sección: Frescos. "
            "Categoría: SUPERMERCADO > Frescos > Lácteos > Yogures > Natural. Marca: Danone.",
            ["retail", "food", "supermercado", "frescos", "lacteos", "yogures"],
            "yogur-natural-007",
        ),
        (
            "Café molido mezcla paquete 250g. Familia: Café. Sección: Alimentación. "
            "Categoría: SUPERMERCADO > Alimentación > Café e Infusiones > Café. Marca: Marcilla.",
            ["retail", "food", "supermercado", "alimentacion", "cafe"],
            "cafe-molido-008",
        ),
        (
            "Detergente lavadora líquido 3 litros. Familia: Limpieza. Sección: Hogar. "
            "Categoría: SUPERMERCADO > Hogar > Limpieza > Lavado. Marca: Ariel.",
            ["retail", "food", "supermercado", "hogar", "limpieza"],
            "detergente-lavadora-009",
        ),
        (
            "Champú anticaspa frasco 400ml. Familia: Higiene. Sección: Cosmética. "
            "Categoría: SUPERMERCADO > Cosmética > Higiene > Capilar. Marca: H&S.",
            ["retail", "food", "supermercado", "cosmetica", "higiene"],
            "champu-anticaspa-010",
        ),
        (
            "Arroz redondo paquete 1kg. Familia: Pastas y Arroces. Sección: Alimentación. "
            "Categoría: SUPERMERCADO > Alimentación > Pastas y Arroces > Arroz. Marca: SOS.",
            ["retail", "food", "supermercado", "alimentacion", "arroz"],
            "arroz-redondo-011",
        ),
        (
            "Cerveza rubia lata 33cl pack 6. Familia: Bebidas. Sección: Bebidas. "
            "Categoría: SUPERMERCADO > Bebidas > Cervezas > Rubia. Marca: Mahou.",
            ["retail", "food", "supermercado", "bebidas", "cervezas"],
            "cerveza-rubia-012",
        ),
    ]

    for desc, tags, source_id in products:
        tokens = sorted(set(desc.lower().replace(".", " ").replace(":", " ").replace(",", " ").split()))
        tokens = [t for t in tokens if len(t) > 2]

        chunk = UnifiedChunk(
            id=str(uuid.uuid4()),
            text_content=desc,
            source_type="retail_product",
            source_id=source_id,
            sparse_tokens={"tokens": tokens},
            metadata=UnifiedChunkMetadata(
                hierarchical_tags=tags,
                parent_structure=tags[-1] if tags else None,
                file_path_or_url=f"test://retail_catalog/{source_id}",
                custom_attributes={"seed": True, "source_id": source_id},
            ),
        )
        chunks.append(chunk)

    return chunks


def setup_test_collection() -> None:
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    try:
        if client.collection_exists(TEST_COLLECTION):
            client.delete_collection(TEST_COLLECTION)
            logger.info("Deleted existing test collection.")

        client.create_collection(
            collection_name=TEST_COLLECTION,
            vectors_config={
                "dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(),
            },
            quantization_config=ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8,
                    always_ram=True,
                )
            ),
        )
        logger.info("Created test collection '%s'.", TEST_COLLECTION)
    finally:
        client.close()


def teardown_test_collection() -> None:
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    try:
        if client.collection_exists(TEST_COLLECTION):
            client.delete_collection(TEST_COLLECTION)
            logger.info("Deleted test collection '%s'.", TEST_COLLECTION)
    finally:
        client.close()


async def run_hybrid_retrieval(query: str, top_n: int = 10, top_k: int = 5) -> List[SearchResult]:
    mock_dense = IngestionWorker.mock_dense_fn(dim=DENSE_DIM)
    mock_sparse = IngestionWorker.mock_sparse_fn()

    local_searcher = LocalSearcher(
        collection_name=TEST_COLLECTION,
        top_n=top_n,
        dense_embed_fn=mock_dense,
        sparse_embed_fn=mock_sparse,
    )
    local_results = await local_searcher.search(query, top_n=top_n)
    await local_searcher.close()

    dense_results = local_results["dense"]
    sparse_results = local_results["sparse"]
    logger.info("[Pipeline] Dense results: %d | Sparse results: %d", len(dense_results), len(sparse_results))

    external_provider = MockExternalSearchProvider(max_hits=3)
    live_results = await external_provider.fetch_live_context(query)
    logger.info("[Pipeline] Live external results: %d", len(live_results))

    fused = reciprocal_rank_fusion([dense_results, sparse_results, live_results])
    logger.info("[Pipeline] Fused unique chunks: %d", len(fused))

    reranker = CrossEncoderReranker()
    final = await reranker.rerank(query, fused, top_k=top_k)
    logger.info("[Pipeline] Final reranked chunks: %d", len(final))

    return final


async def main() -> int:
    logger.info("=" * 60)
    logger.info("Phase 3 Retrieval Engine — End-to-End Validation")
    logger.info("=" * 60)

    query = "leche fresca entera sin lactosa"

    setup_test_collection()

    seed = build_seed_chunks()
    logger.info("Seeding %d synthetic retail chunks into test collection ...", len(seed))
    worker = IngestionWorker(
        collection_name=TEST_COLLECTION,
        dense_dim=DENSE_DIM,
        dense_embed_fn=IngestionWorker.mock_dense_fn(dim=DENSE_DIM),
        sparse_embed_fn=IngestionWorker.mock_sparse_fn(),
    )
    await worker.ingest(seed)
    await worker.close()
    logger.info("Seed ingestion complete.")

    logger.info("Running hybrid retrieval for query: '%s'", query)
    try:
        final_results = await run_hybrid_retrieval(query, top_n=10, top_k=5)
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc, exc_info=True)
        teardown_test_collection()
        return 1

    logger.info("-" * 60)
    logger.info("FINAL RERANKED RESULTS")
    logger.info("-" * 60)
    for rank, result in enumerate(final_results, start=1):
        chunk = result.chunk
        logger.info(
            "%2d. [%s] %s ... (source_id=%s, score=%.4f, tags=%s)",
            rank,
            chunk.source_type,
            chunk.text_content[:70],
            chunk.source_id,
            result.relevance_score,
            chunk.metadata.hierarchical_tags,
        )
    logger.info("-" * 60)

    teardown_test_collection()
    logger.info("Validation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
