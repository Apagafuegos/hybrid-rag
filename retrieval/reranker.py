"""
Cross-Encoder Reranker
=======================
Local contextual scoring layer that re-orders chunks by absolute semantic
relevance to the query using a lightweight cross-encoder model.

Designed for VPS deployment via ``sentence-transformers``.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import List, Tuple

from core.models import UnifiedChunk
from retrieval.models import SearchResult

logger = logging.getLogger(__name__)

DEFAULT_RERANK_MODEL = os.getenv(
    "RERANK_MODEL", "BAAI/bge-reranker-base"
)


class CrossEncoderReranker:
    """
    Async cross-encoder reranker.

    Parameters
    ----------
    model_name:
        Hugging-Face model identifier for the cross-encoder.
    device:
        Torch device (``cpu``, ``cuda``, etc.).  Defaults to CPU for VPS safety.
    max_length:
        Maximum token length per query+text pair.
    """

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_RERANK_MODEL,
        device: str = "cpu",
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self._model = None  # Lazy-load

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def rerank(
        self,
        query: str,
        chunks_with_scores: List[Tuple[float, UnifiedChunk]],
        *,
        top_k: int = 5,
    ) -> List[SearchResult]:
        """
        Score each chunk against *query* and return the top *top_k*.

        Parameters
        ----------
        query:
            Raw user query string.
        chunks_with_scores:
            Candidate chunks with their RRF scores (score, chunk).
        top_k:
            Number of top-scoring chunks to return.

        Returns
        -------
        List[SearchResult]
            Re-sorted results pruned to *top_k* items.
        """
        if not chunks_with_scores:
            logger.debug("Reranker received empty chunk list — returning immediately.")
            return []

        model = self._load_model()

        chunks = [chunk for _, chunk in chunks_with_scores]
        pairs = [(query, chunk.text_content) for chunk in chunks]

        loop = asyncio.get_running_loop()
        scores = await loop.run_in_executor(
            None,
            lambda: model.predict(
                pairs,
                batch_size=8,
                show_progress_bar=False,
                convert_to_numpy=True,
            ),
        )

        scored = list(zip(chunks_with_scores, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        results = [
            SearchResult(
                chunk=chunk,
                rrqf_score=rrqf,
                relevance_score=float(score),
            )
            for (rrqf, chunk), score in scored[:top_k]
        ]
        logger.info(
            "Reranker scored %d chunks, returning top %d for query '%s'",
            len(chunks),
            len(results),
            query,
        )
        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_model(self):
        """Lazy-load the cross-encoder model."""
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for cross-encoder reranking. "
                "Install it with: uv add sentence-transformers"
            ) from exc

        logger.info("Loading cross-encoder model '%s' on device '%s' ...", self.model_name, self.device)
        self._model = CrossEncoder(
            self.model_name,
            device=self.device,
            max_length=self.max_length,
        )
        return self._model
