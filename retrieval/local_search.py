"""
Dual-Vector Local Retriever
============================
Asynchronous retriever that executes parallel dense + sparse queries against
Qdrant and returns ordered arrays of ``UnifiedChunk`` models.

Zero domain awareness: this module knows nothing about C code, email formats,
or retail catalogs.  It only speaks ``UnifiedChunk``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import SparseVector

from core.models import UnifiedChunk

logger = logging.getLogger(__name__)

DEFAULT_QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
DEFAULT_QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
DEFAULT_COLLECTION = os.getenv("QDRANT_COLLECTION", "agnostic_rag_collection")
DEFAULT_LITE_LLM_BASE = os.getenv("LITE_LLM_BASE_URL") or os.getenv("OPENROUTER_BASE_URL", "http://localhost:4000")
DEFAULT_LITE_LLM_KEY = os.getenv("LITE_LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY", "dummy-key")
DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL") or os.getenv("OPENROUTER_EMBED_MODEL", "text-embedding-3-small")
DEFAULT_DENSE_PROVIDER = os.getenv("DENSE_EMBED_PROVIDER", "fastembed")
DEFAULT_FASTEMBED_DENSE_MODEL = os.getenv("FASTEMBED_DENSE_MODEL", "BAAI/bge-small-en-v1.5")
DEFAULT_VOCAB_FILE = os.getenv("VOCAB_FILE", ".ingest_vocab.json")


# Simple tokenizer for query text (same style as extractors)
_STOPWORDS = {
    "the", "and", "for", "are", "but", "not", "you", "all", "can",
    "had", "her", "was", "one", "our", "out", "day", "get", "has",
    "him", "his", "how", "its", "may", "new", "now", "old", "see",
    "two", "who", "boy", "did", "she", "use", "way", "many",
}


def _tokenize_query(text: str) -> List[str]:
    """Extract keyword tokens from a raw query string."""
    tokens = re.findall(r"[A-Za-z_]\w*", text)
    return sorted({
        t.lower() for t in tokens
        if len(t) > 2 and t.lower() not in _STOPWORDS
    })


class LocalSearcher:
    """
    Async dual-vector retriever for Qdrant.

    Parameters
    ----------
    collection_name:
        Target Qdrant collection.
    top_n:
        Default number of results to fetch from each vector index.
    dense_embed_fn:
        Async callable that turns a list of query strings into dense vectors.
    sparse_embed_fn:
        Async callable that turns a list of query strings into SparseVector objects.
        If None, uses the vocab-based token approach (matching the ingestion default).
    """

    def __init__(
        self,
        *,
        qdrant_host: str = DEFAULT_QDRANT_HOST,
        qdrant_port: int = DEFAULT_QDRANT_PORT,
        collection_name: str = DEFAULT_COLLECTION,
        top_n: int = 20,
        dense_embed_fn: Optional[Callable[[List[str]], Any]] = None,
        sparse_embed_fn: Optional[Callable[[List[str]], Any]] = None,
        vocab_path: str = DEFAULT_VOCAB_FILE,
    ) -> None:
        self.collection_name = collection_name
        self.top_n = top_n
        self._qdrant = AsyncQdrantClient(host=qdrant_host, port=qdrant_port)

        # Dense embedding
        if dense_embed_fn is not None:
            self._dense_fn = dense_embed_fn
        elif DEFAULT_DENSE_PROVIDER == "openrouter":
            self._dense_fn = self._make_openai_dense_fn(
                DEFAULT_LITE_LLM_BASE, DEFAULT_LITE_LLM_KEY, DEFAULT_EMBED_MODEL
            )
        else:
            self._dense_fn = self._make_fastembed_dense_fn(DEFAULT_FASTEMBED_DENSE_MODEL)

        # Sparse embedding
        if sparse_embed_fn is not None:
            self._sparse_fn = sparse_embed_fn
            self._token_vocab = None
        else:
            self._sparse_fn = None  # use vocab-based
            self._token_vocab = self._load_vocab(vocab_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        *,
        hierarchical_tags: Optional[List[str]] = None,
        top_n: Optional[int] = None,
    ) -> Dict[str, List[UnifiedChunk]]:
        limit = top_n or self.top_n
        logger.debug("LocalSearcher querying '%s' (limit=%d)", query, limit)

        # Dense vector
        dense_vector = (await self._embed_dense([query]))[0]

        # Sparse vector
        if self._sparse_fn is not None:
            sparse_vector = (await self._embed_sparse([query]))[0]
        else:
            sparse_vector = self._query_sparse_vector(query)

        qdrant_filter = self._build_filter(hierarchical_tags)

        dense_task = asyncio.create_task(
            self._qdrant.query_points(
                collection_name=self.collection_name,
                query=dense_vector,
                using="dense",
                query_filter=qdrant_filter,
                limit=limit,
                with_payload=True,
            )
        )
        sparse_task = asyncio.create_task(
            self._qdrant.query_points(
                collection_name=self.collection_name,
                query=sparse_vector,
                using="sparse",
                query_filter=qdrant_filter,
                limit=limit,
                with_payload=True,
            )
        )

        dense_response = await dense_task
        sparse_response = await sparse_task

        dense_chunks = [self._point_to_chunk(r) for r in dense_response.points]
        sparse_chunks = [self._point_to_chunk(r) for r in sparse_response.points]

        logger.info(
            "LocalSearcher returned dense=%d sparse=%d for query '%s'",
            len(dense_chunks),
            len(sparse_chunks),
            query,
        )
        return {"dense": dense_chunks, "sparse": sparse_chunks}

    async def close(self) -> None:
        await self._qdrant.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _embed_dense(self, texts: List[str]) -> List[List[float]]:
        return await self._dense_fn(texts)

    async def _embed_sparse(self, texts: List[str]) -> List[SparseVector]:
        return await self._sparse_fn(texts)

    def _query_sparse_vector(self, query: str) -> Optional[SparseVector]:
        """Build a sparse vector from query tokens using the ingestion vocab."""
        if not self._token_vocab:
            return SparseVector(indices=[], values=[])
        tokens = _tokenize_query(query)
        indices: List[int] = []
        seen: set[int] = set()
        for t in tokens:
            idx = self._token_vocab.get(t)
            if idx is not None and idx not in seen:
                indices.append(idx)
                seen.add(idx)
        if not indices:
            return SparseVector(indices=[], values=[])
        return SparseVector(indices=sorted(indices), values=[1.0] * len(indices))

    @staticmethod
    def _point_to_chunk(point: Any) -> UnifiedChunk:
        payload: Dict[str, Any] = point.payload or {}
        return UnifiedChunk.model_validate(payload)

    @staticmethod
    def _build_filter(hierarchical_tags: Optional[List[str]]) -> Optional[Any]:
        if not hierarchical_tags:
            return None
        from qdrant_client.models import FieldCondition, Filter, MatchAny
        return Filter(
            must=[
                FieldCondition(
                    key="metadata.hierarchical_tags",
                    match=MatchAny(any=hierarchical_tags),
                )
            ]
        )

    @staticmethod
    def _load_vocab(vocab_path: str) -> Dict[str, int]:
        path = Path(vocab_path)
        if not path.exists():
            logger.warning("Vocab file %s not found — sparse search disabled.", vocab_path)
            return {}
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    # ------------------------------------------------------------------
    # Dense embedding helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_openai_dense_fn(base_url: str, api_key: str, model: str):
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError("openai package required for default dense embeddings") from exc

        client = AsyncOpenAI(base_url=base_url, api_key=api_key)

        async def _embed(texts: List[str]) -> List[List[float]]:
            response = await client.embeddings.create(model=model, input=texts)
            return [item.embedding for item in response.data]

        return _embed

    @staticmethod
    def _make_fastembed_dense_fn(model_name: str):
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise RuntimeError("fastembed package required for local dense embeddings") from exc

        model = TextEmbedding(model_name=model_name)

        async def _embed(texts: List[str]) -> List[List[float]]:
            loop = asyncio.get_running_loop()
            embeddings = await loop.run_in_executor(
                None, lambda: list(model.embed(texts))
            )
            return [emb.tolist() for emb in embeddings]

        return _embed
