"""
API Reranker
============
Semantic reranking via OpenRouter's hosted Cohere rerank models.
Zero local ML dependencies — a single HTTP POST per ``rerank()`` call.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
from typing import List, Tuple

from core.models import UnifiedChunk
from retrieval.models import SearchResult

logger = logging.getLogger(__name__)

DEFAULT_API_MODEL = os.getenv("RERANK_API_MODEL", "cohere/rerank-4-fast")


def _api_key() -> str:
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is required for reranking")
    return key


class ApiReranker:
    """
    Async reranker powered by OpenRouter's hosted Cohere rerank models.

    Parameters
    ----------
    model_name:
        OpenRouter model slug (e.g. ``cohere/rerank-4-fast``).
    api_key:
        OpenRouter API key.
    base_url:
        OpenRouter base URL.
    """

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_API_MODEL,
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
    ) -> None:
        self.model_name = model_name
        self._api_key = api_key or _api_key()
        self._url = base_url.rstrip("/") + "/rerank"

    async def rerank(
        self,
        query: str,
        chunks_with_scores: List[Tuple[float, UnifiedChunk]],
        *,
        top_k: int = 5,
    ) -> List[SearchResult]:
        if not chunks_with_scores:
            return []

        chunks = [chunk for _, chunk in chunks_with_scores]
        docs = [chunk.text_content for chunk in chunks]

        import httpx

        max_retries = 3
        base_wait = 1.0

        for attempt in range(1, max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                    response = await client.post(
                        self._url,
                        headers={
                            "Authorization": f"Bearer {self._api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": self.model_name,
                            "query": query,
                            "documents": docs,
                            "top_n": top_k,
                        },
                    )

                    if response.status_code == 429:
                        retry_after = response.headers.get("retry-after")
                        wait = float(retry_after) if retry_after else base_wait * (2 ** (attempt - 1)) + random.uniform(0, 1)
                        logger.warning("Rerank API rate-limited (attempt %d/3), waiting %.1fs", attempt, wait)
                        await asyncio.sleep(wait)
                        continue

                    if response.status_code >= 500:
                        logger.warning("Rerank API server error %d (attempt %d/3)", response.status_code, attempt)
                        await asyncio.sleep(base_wait * (2 ** (attempt - 1)))
                        continue

                    if response.status_code != 200:
                        body = response.text[:500]
                        raise RuntimeError(f"Rerank API returned {response.status_code}: {body}")

                    data = response.json()
                    break

            except httpx.TimeoutException:
                logger.warning("Rerank API timeout (attempt %d/3)", attempt)
                if attempt == max_retries:
                    raise TimeoutError("Rerank API timed out after 3 attempts")
                await asyncio.sleep(base_wait * (2 ** (attempt - 1)))
            except httpx.ConnectError as exc:
                logger.warning("Rerank API connect error (attempt %d/3): %s", attempt, exc)
                if attempt == max_retries:
                    raise RuntimeError(f"Rerank API unavailable: {exc}") from exc
                await asyncio.sleep(base_wait * (2 ** (attempt - 1)))

        results_data = data.get("results", [])
        scored: List[Tuple[float, UnifiedChunk, float]] = []
        for r in results_data:
            idx = r["index"]
            rrqf = chunks_with_scores[idx][0]
            chunk = chunks[idx]
            relevance = float(r["relevance_score"])
            scored.append((relevance, chunk, rrqf))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            SearchResult(chunk=chunk, rrqf_score=rrqf, relevance_score=float(score))
            for score, chunk, rrqf in scored[:top_k]
        ]
