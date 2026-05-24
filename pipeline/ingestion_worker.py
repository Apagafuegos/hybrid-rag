"""
Agnostic Batch Ingestion Engine
=================================
Asynchronous ingestion manager that accepts uniform ``UnifiedChunk``
arrays and automates batched embedding + bulk upsert into Qdrant.

Zero domain awareness: this module knows nothing about C code, email
formats, or retail catalogs.  It only speaks ``UnifiedChunk``.

Sparse vectors are built **instantly** from each chunk's pre-computed
``sparse_tokens`` field using a global vocabulary — no ML model overhead.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct, SparseVector

from core.models import UnifiedChunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults (override via env vars or constructor arguments)
# ---------------------------------------------------------------------------
DEFAULT_QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
DEFAULT_QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
DEFAULT_COLLECTION = os.getenv("QDRANT_COLLECTION", "agnostic_rag_collection")
DEFAULT_BATCH_SIZE = int(os.getenv("INGEST_BATCH_SIZE", "32"))
DEFAULT_LITE_LLM_BASE = os.getenv("LITE_LLM_BASE_URL") or os.getenv("OPENROUTER_BASE_URL", "http://localhost:4000")
DEFAULT_LITE_LLM_KEY = os.getenv("LITE_LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY", "dummy-key")
DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL") or os.getenv("OPENROUTER_EMBED_MODEL", "text-embedding-3-small")
DEFAULT_CHECKPOINT_FILE = os.getenv("INGEST_CHECKPOINT_FILE", ".ingest_checkpoint.json")
DEFAULT_MAX_EMBED_CHARS = int(os.getenv("MAX_EMBEDDING_CHARS", "32000"))
DEFAULT_DENSE_PROVIDER = os.getenv("DENSE_EMBED_PROVIDER", "fastembed")
DEFAULT_FASTEMBED_DENSE_MODEL = os.getenv("FASTEMBED_DENSE_MODEL", "BAAI/bge-small-en-v1.5")
DEFAULT_DENSE_DIM = int(os.getenv("DENSE_VECTOR_DIM", "1536"))
DEFAULT_VOCAB_FILE = os.getenv("VOCAB_FILE", ".ingest_vocab.json")


class IngestionWorker:
    """
    Async batch ingestion orchestrator.

    Supports checkpoint/resume so long-running ingestions survive
    transient API failures without re-processing already-uploaded
    batches.
    """

    def __init__(
        self,
        *,
        qdrant_host: str = DEFAULT_QDRANT_HOST,
        qdrant_port: int = DEFAULT_QDRANT_PORT,
        collection_name: str = DEFAULT_COLLECTION,
        batch_size: int = DEFAULT_BATCH_SIZE,
        lite_llm_base_url: str = DEFAULT_LITE_LLM_BASE,
        lite_llm_api_key: str = DEFAULT_LITE_LLM_KEY,
        embedding_model: str = DEFAULT_EMBED_MODEL,
        dense_dim: int = DEFAULT_DENSE_DIM,
        dense_embed_fn: Optional[Callable[[List[str]], Any]] = None,
        sparse_embed_fn: Optional[Callable[[List[str]], Any]] = None,
        checkpoint_file: str = DEFAULT_CHECKPOINT_FILE,
        inter_batch_delay: float = 0.0,
        max_embedding_chars: int = DEFAULT_MAX_EMBED_CHARS,
        dense_provider: str = DEFAULT_DENSE_PROVIDER,
        fastembed_dense_model: str = DEFAULT_FASTEMBED_DENSE_MODEL,
    ) -> None:
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.dense_dim = dense_dim
        self.max_embedding_chars = max_embedding_chars
        self._qdrant = AsyncQdrantClient(host=qdrant_host, port=qdrant_port)
        self._checkpoint_path = Path(checkpoint_file)
        self._inter_batch_delay = inter_batch_delay

        # Dense embedding strategy
        if dense_embed_fn is not None:
            self._dense_fn = dense_embed_fn
        elif dense_provider == "openrouter":
            self._dense_fn = self._make_openai_dense_fn(
                lite_llm_base_url, lite_llm_api_key, embedding_model
            )
        else:
            self._dense_fn = self._make_fastembed_dense_fn(fastembed_dense_model)

        # Sparse embedding strategy — None means use vocab-based token vectors
        self._sparse_fn: Optional[Callable] = sparse_embed_fn
        self._token_vocab: Dict[str, int] = {}

    # -----------------------------------------------------------------------
    # Mock helpers (for testing without real embedding APIs)
    # -----------------------------------------------------------------------

    @staticmethod
    def mock_dense_fn(*, dim: int = 1536):
        """Return a deterministic mock dense embedding function."""
        import numpy as np
        rng = np.random.RandomState(42)

        async def _mock_dense(texts: List[str]) -> List[List[float]]:
            vectors = rng.randn(len(texts), dim).astype(np.float32)
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / norms
            return vectors.tolist()

        return _mock_dense

    @staticmethod
    def mock_sparse_fn():
        """Return a deterministic mock sparse embedding function."""
        from qdrant_client.models import SparseVector

        async def _mock_sparse(texts: List[str]) -> List[SparseVector]:
            results: List[SparseVector] = []
            for text in texts:
                tokens = sorted(set(text.lower().split()))
                indices = [hash(t) % 1000 for t in tokens]
                results.append(SparseVector(
                    indices=sorted(set(indices)),
                    values=[1.0] * len(set(indices)),
                ))
            return results

        return _mock_sparse

    # -----------------------------------------------------------------------
    # Checkpoint helpers
    # -----------------------------------------------------------------------

    def _load_checkpoint(self) -> int:
        if not self._checkpoint_path.exists():
            return 0
        try:
            data = json.loads(self._checkpoint_path.read_text())
            offset = data.get("last_offset", 0)
            logger.info("Resuming from checkpoint offset %d.", offset)
            return offset
        except Exception:
            return 0

    def _save_checkpoint(self, offset: int) -> None:
        self._checkpoint_path.write_text(json.dumps({"last_offset": offset}))

    def clear_checkpoint(self) -> None:
        if self._checkpoint_path.exists():
            self._checkpoint_path.unlink()

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    async def ingest(self, chunks: List[UnifiedChunk]) -> None:
        """Ingest a pre-loaded list of chunks (may use significant memory)."""
        if not chunks:
            logger.info("No chunks provided for ingestion.")
            return

        total_before = len(chunks)

        # Build vocab from all chunks (needs full scan)
        self._build_and_save_vocab(chunks)

        # Global dedup scan
        seen_hashes: set[str] = set()
        deduped_indices: List[int] = []
        for i, ch in enumerate(chunks):
            h = hashlib.sha256(ch.text_content.encode()).hexdigest()
            if h not in seen_hashes:
                seen_hashes.add(h)
                deduped_indices.append(i)

        skipped = total_before - len(deduped_indices)
        if skipped:
            logger.info("Removed %d duplicate chunks before ingestion.", skipped)
        if not deduped_indices:
            logger.info("No unique chunks remaining after deduplication.")
            return

        total = len(deduped_indices)
        start_offset = self._load_checkpoint()

        logger.info("Starting ingestion of %d chunk(s) from offset %d ...", total, start_offset)

        t0 = time.monotonic()
        for batch_start in range(start_offset, total, self.batch_size):
            batch_idx = deduped_indices[batch_start : batch_start + self.batch_size]
            batch = [chunks[i] for i in batch_idx]
            try:
                await self._ingest_batch(batch, batch_offset=batch_start)
                self._save_checkpoint(batch_start + len(batch))
            except Exception as exc:
                logger.error("Fatal failure at offset %d (size %d): %s", batch_start, len(batch), exc, exc_info=True)
                raise

            if self._inter_batch_delay > 0:
                await asyncio.sleep(self._inter_batch_delay)

            if (batch_start - start_offset) % (self.batch_size * 20) == 0 and batch_start > start_offset:
                elapsed = time.monotonic() - t0
                done = batch_start - start_offset
                rate = done / elapsed if elapsed > 0 else 0
                remaining = total - batch_start
                eta = remaining / rate if rate > 0 else 0
                logger.info("Progress: %d/%d (%.1f%%) — %.0f chunks/s, ETA %.0f min",
                             batch_start, total, batch_start / total * 100, rate, eta / 60)

        self.clear_checkpoint()
        elapsed = time.monotonic() - t0
        logger.info("Ingestion complete. %d chunk(s) in %.0f s (%.0f chunks/s).", total, elapsed, total / elapsed if elapsed > 0 else 0)

    async def ingest_from_cache(self, cache_path: str) -> None:
        """Ingest directly from a JSONL cache file — constant memory usage.

        Two passes over the file:
        1. Build vocab from sparse tokens (no full chunk load)
        2. Stream chunks one at a time, dedup, embed, upsert
        """
        from pathlib import Path

        cp = Path(cache_path)
        if not cp.exists():
            logger.error("Cache file not found: %s", cache_path)
            return

        # ---- Pass 1: build vocab -----
        logger.info("Pass 1/2: building vocab from %s ...", cache_path)
        all_tokens: set[str] = set()
        line_count = 0

        with cp.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    tokens = obj.get("sparse_tokens", {}).get("tokens", [])
                    all_tokens.update(tokens)
                except Exception:
                    pass
                line_count += 1

        self._token_vocab = {t: i for i, t in enumerate(sorted(all_tokens))}
        self._save_vocab()
        logger.info("Vocab: %d tokens from %d cache lines.", len(self._token_vocab), line_count)

        # ---- Pass 2: stream, dedup, embed, upsert -----
        start_offset = self._load_checkpoint()
        logger.info("Pass 2/2: ingesting from offset %d ...", start_offset)

        processed_hashes: set[str] = set()
        batch: List[UnifiedChunk] = []
        chunks_processed = 0      # total unique chunks *processed*
        chunks_skipped_start = 0  # how many we've passed to reach offset
        t0 = time.monotonic()

        with cp.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    obj.pop("_cache_fingerprint", None)
                    h = hashlib.sha256(obj["text_content"].encode()).hexdigest()
                except Exception:
                    continue

                # Global dedup: skip if we've already seen this exact text
                if h in processed_hashes:
                    continue
                processed_hashes.add(h)

                # Skip chunks before the checkpoint offset
                if chunks_skipped_start < start_offset:
                    chunks_skipped_start += 1
                    continue

                try:
                    chunk = UnifiedChunk.model_validate(obj)
                except Exception:
                    continue

                batch.append(chunk)

                if len(batch) >= self.batch_size:
                    await self._ingest_batch(batch, batch_offset=chunks_processed)
                    chunks_processed += len(batch)
                    self._save_checkpoint(chunks_processed)
                    batch.clear()

                    if self._inter_batch_delay > 0:
                        await asyncio.sleep(self._inter_batch_delay)

                    if chunks_processed % (self.batch_size * 20) == 0:
                        elapsed = time.monotonic() - t0
                        rate = chunks_processed / elapsed if elapsed > 0 else 0
                        logger.info("Progress: %d chunks — %.0f chunks/s, ETA unbounded",
                                     chunks_processed, rate)

            # Flush remaining
            if batch:
                await self._ingest_batch(batch, batch_offset=chunks_processed)
                chunks_processed += len(batch)
                self._save_checkpoint(chunks_processed)

        self.clear_checkpoint()
        elapsed = time.monotonic() - t0
        logger.info("Ingestion complete. %d chunks in %.0f s (%.0f chunks/s).",
                     chunks_processed, elapsed,
                     chunks_processed / elapsed if elapsed > 0 else 0)

    async def close(self) -> None:
        await self._qdrant.close()

    # -----------------------------------------------------------------------
    # Vocabulary builder (token-based sparse vectors)
    # -----------------------------------------------------------------------

    @staticmethod
    def _build_vocab(chunks: List[UnifiedChunk]) -> Dict[str, int]:
        """Build a global token→index mapping from all chunks' sparse_tokens."""
        all_tokens: set[str] = set()
        for ch in chunks:
            tokens = ch.sparse_tokens.get("tokens", [])
            all_tokens.update(tokens)
        return {t: i for i, t in enumerate(sorted(all_tokens))}

    def _build_and_save_vocab(self, chunks: List[UnifiedChunk]) -> None:
        self._token_vocab = self._build_vocab(chunks)
        logger.info("Built sparse vocab with %d unique tokens.", len(self._token_vocab))
        self._save_vocab()

    def _save_vocab(self) -> None:
        """Persist the token vocabulary to disk so the retriever can use it."""
        path = Path(DEFAULT_VOCAB_FILE)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self._token_vocab, fh)
        logger.info("Saved vocab (%d tokens) to %s.", len(self._token_vocab), path)

    # -----------------------------------------------------------------------
    # Batch internals
    # -----------------------------------------------------------------------

    async def _ingest_batch(
        self, batch: List[UnifiedChunk], batch_offset: int
    ) -> None:
        # Truncate texts for embedding only; keep full text in payload.
        texts: List[str] = []
        for idx, chunk in enumerate(batch):
            text = chunk.text_content
            if len(text) > self.max_embedding_chars:
                logger.warning(
                    "Chunk %d (%s) text length %d exceeds max_embedding_chars (%d); "
                    "truncating for embedding.",
                    batch_offset + idx,
                    chunk.source_id,
                    len(text),
                    self.max_embedding_chars,
                )
                text = text[: self.max_embedding_chars]
            texts.append(text)

        dense_vectors: List[Optional[List[float]]] = [None] * len(batch)
        sparse_vectors: List[Optional[SparseVector]] = [None] * len(batch)

        # Compute dense embeddings (API call)
        try:
            dense_results = await self._dense_fn(texts)
            for i in range(len(batch)):
                dense_vectors[i] = dense_results[i]
        except Exception as batch_exc:
            logger.warning(
                "Batch dense embedding failed at offset %d (size %d): %s. "
                "Falling back to per-chunk ...",
                batch_offset,
                len(batch),
                batch_exc,
            )
            for i, text in enumerate(texts):
                try:
                    dense_v = await self._dense_fn([text])
                    dense_vectors[i] = dense_v[0]
                except Exception as single_exc:
                    logger.error(
                        "Skipping chunk %d (%s): dense embedding failed: %s",
                        batch_offset + i,
                        batch[i].source_id,
                        single_exc,
                    )

        # Build sparse vectors from tokens (instant, no API/ML call)
        for i, chunk in enumerate(batch):
            if self._sparse_fn is not None:
                # Custom sparse embedder (legacy)
                try:
                    sparse_v = await self._sparse_fn([chunk.text_content])
                    sparse_vectors[i] = sparse_v[0]
                except Exception:
                    pass
            else:
                sparse_vectors[i] = self._chunk_sparse_vector(chunk)

        # Build Qdrant points
        points: List[PointStruct] = []
        for idx, chunk in enumerate(batch):
            if dense_vectors[idx] is None or sparse_vectors[idx] is None:
                continue
            points.append(
                PointStruct(
                    id=chunk.id,
                    vector={
                        "dense": dense_vectors[idx],
                        "sparse": sparse_vectors[idx],
                    },
                    payload=chunk.model_dump(mode="json"),
                )
            )

        if not points:
            logger.warning(
                "All chunks in batch %d-%d were skipped; nothing to upsert.",
                batch_offset + 1,
                batch_offset + len(batch),
            )
            return

        try:
            await self._qdrant.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            logger.info(
                "Upserted batch %d-%d (%d/%d chunks) into '%s'.",
                batch_offset + 1,
                batch_offset + len(batch),
                len(points),
                len(batch),
                self.collection_name,
            )
        except Exception as exc:
            logger.error(
                "Qdrant upsert failure at offset %d: %s",
                batch_offset,
                exc,
                exc_info=True,
            )
            raise

    def _chunk_sparse_vector(self, chunk: UnifiedChunk) -> Optional[SparseVector]:
        """Build a SparseVector from the chunk's pre-computed sparse tokens."""
        tokens = chunk.sparse_tokens.get("tokens", [])
        if not tokens:
            return None
        indices: List[int] = []
        seen_idx: set[int] = set()
        for t in tokens:
            idx = self._token_vocab.get(t)
            if idx is not None and idx not in seen_idx:
                indices.append(idx)
                seen_idx.add(idx)
        if not indices:
            return None
        return SparseVector(
            indices=sorted(indices),
            values=[1.0] * len(indices),
        )

    # -----------------------------------------------------------------------
    # Dense embedding (OpenAI-compatible API)
    # -----------------------------------------------------------------------

    @staticmethod
    def _make_openai_dense_fn(base_url: str, api_key: str, model: str):
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is required for default dense embeddings") from exc

        import httpx

        client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=httpx.AsyncClient(
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
                timeout=httpx.Timeout(60.0, connect=10.0),
            ),
        )

        async def _embed(texts: List[str]) -> List[List[float]]:
            max_retries = 3
            base_wait = 1.0

            for attempt in range(1, max_retries + 1):
                try:
                    response = await client.embeddings.create(
                        model=model,
                        input=texts,
                    )
                    if not response.data:
                        raise ValueError(
                            f"Empty embedding data in response. "
                            f"Model={model}, texts={len(texts)}"
                        )
                    return [item.embedding for item in response.data]
                except Exception as exc:
                    logger.warning(
                        "Embedding API attempt %d/%d failed: %s",
                        attempt,
                        max_retries,
                        exc,
                    )
                    if attempt == max_retries:
                        raise RuntimeError(
                            f"Embedding API failed after {max_retries} attempts: {exc}"
                        ) from exc

                    wait = base_wait * (2 ** (attempt - 1)) + random.uniform(0, 1)
                    logger.info("Retrying in %.1f seconds ...", wait)
                    await asyncio.sleep(wait)

            raise RuntimeError("Embedding loop exited unexpectedly")

        return _embed

    # -----------------------------------------------------------------------
    # Dense embedding (FastEmbed - local, fast, free)
    # -----------------------------------------------------------------------

    @staticmethod
    def _make_fastembed_dense_fn(model_name: str):
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise RuntimeError(
                "fastembed package is required for local dense embeddings"
            ) from exc

        model = TextEmbedding(model_name=model_name)

        async def _embed(texts: List[str]) -> List[List[float]]:
            loop = asyncio.get_running_loop()
            embeddings = await loop.run_in_executor(
                None, lambda: list(model.embed(texts))
            )
            return [emb.tolist() for emb in embeddings]

        return _embed

