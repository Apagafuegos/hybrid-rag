"""
Domain-Blind Orchestration Controller
=======================================
Wraps the complete Phase 3 retrieval pipeline (LocalSearch → ExternalFetch
→ RRF → Rerank) behind a single async entry point for the MCP server layer.

Zero domain awareness: this module knows nothing about retail, kernel code,
or email threads.  All inputs and outputs are generic strings and
``UnifiedChunk`` instances.
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional

from retrieval.external_search import ExternalSearchPort
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.local_search import LocalSearcher
from retrieval.models import SearchResult
from retrieval.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)

LOCAL_FETCH_LIMIT = 20
RRF_K = 60


class HybridSearchOrchestrator:
    """
    Stateless async controller that chains the full retrieval pipeline.

    Lifecycle
    ---------
    This class is designed to be created once and reused across many calls.
    ``searcher``, ``external_provider``, and ``reranker`` are all injected
    (or lazily defaulted) and must remain valid across calls.
    """

    def __init__(
        self,
        *,
        external_provider: Optional[ExternalSearchPort] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        default_limit: int = 5,
        query_timeout: float = 30.0,
    ) -> None:
        self._external = external_provider
        self._reranker = reranker or CrossEncoderReranker()
        self._default_limit = default_limit
        self._query_timeout = query_timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def query(
        self,
        query: str,
        *,
        filter_tags: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Execute the full hybrid retrieval pipeline for *query*.

        Pipeline stages:
            1. Parallel dense + sparse Qdrant lookups
            2. Live external context fetch
            3. Reciprocal Rank Fusion (RRF) across all result lists
            4. Cross-encoder semantic reranking

        Parameters
        ----------
        query:
            Raw free-text search string.
        filter_tags:
            Optional hierarchical tag values for Qdrant scope restrictions.
        limit:
            Maximum chunks to return after reranking (default: 5).

        Returns
        -------
        List[SearchResult]
            Ordered (best-first) relevant chunks with relevance scores.

        Raises
        ------
        TimeoutError
            If any pipeline stage exceeds *query_timeout*.
        RuntimeError
            On unrecoverable infrastructure failures.
        """
        if not query.strip():
            raise ValueError("query must be a non-empty string")

        searcher = LocalSearcher()
        top_n = limit if limit else self._default_limit

        try:
            async with asyncio.timeout(self._query_timeout):
                local = await searcher.search(
                    query,
                    hierarchical_tags=filter_tags,
                    top_n=LOCAL_FETCH_LIMIT,
                )
                logger.debug(
                    "Local search returned dense=%d sparse=%d",
                    len(local["dense"]),
                    len(local["sparse"]),
                )

                lists = [local["dense"], local["sparse"]]
                if self._external is not None:
                    live = await self._external.fetch_live_context(query)
                    logger.debug("External provider returned %d hits", len(live))
                    lists.append(live)

                fused = reciprocal_rank_fusion(lists, k=RRF_K)
                logger.info("RRF fused into %d unique chunks", len(fused))

                final = await self._reranker.rerank(query, fused, top_k=top_n)
                logger.info(
                    "Pipeline complete for query '%s' — returning %d results",
                    query,
                    len(final),
                )
                return final

        except asyncio.TimeoutError:
            logger.error("Search timed out after %.1fs for query '%s'", self._query_timeout, query)
            raise TimeoutError(
                f"Hybrid search timed out after {self._query_timeout}s"
            )
        except (TimeoutError, ValueError):
            raise
        except Exception as exc:
            logger.exception("Unhandled pipeline failure for query '%s'", query)
            raise RuntimeError(
                f"Search pipeline failed: {type(exc).__name__}: {exc}"
            ) from exc
        finally:
            await searcher.close()

    async def warmup(self) -> None:
        logger.info("Warming up reranker model...")
        await asyncio.to_thread(self._reranker._load_model)
        logger.info("Warmup complete.")
