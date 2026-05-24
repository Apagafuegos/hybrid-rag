"""
Live External Port Adapter
===========================
Abstract port for real-time external data fetching.

All implementations produce transient ``UnifiedChunk`` objects so they slot
into the domain-agnostic retrieval pipeline without leakage.
"""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from typing import List

from core.models import UnifiedChunk, UnifiedChunkMetadata

logger = logging.getLogger(__name__)


class ExternalSearchPort(ABC):
    """
    Abstract port for fetching live, real-time context from external sources.

    Concrete subclasses (e.g. GitCommitProvider, RSSFeedProvider,
    MailingListProvider) must implement ``fetch_live_context`` and return
    freshly-minted ``UnifiedChunk`` instances.
    """

    @abstractmethod
    async def fetch_live_context(self, query: str) -> List[UnifiedChunk]:
        """
        Fetch live hits for *query* and map each hit to a ``UnifiedChunk``.

        Parameters
        ----------
        query:
            User query string.

        Returns
        -------
        List[UnifiedChunk]
            Transient chunks ordered by provider-specific relevance.
        """
        ...


class MockExternalSearchProvider(ExternalSearchPort):
    """Test double that returns synthetic chunks for pipeline validation."""

    def __init__(self, max_hits: int = 3) -> None:
        self.max_hits = max_hits

    async def fetch_live_context(self, query: str) -> List[UnifiedChunk]:
        chunks: List[UnifiedChunk] = []
        for i in range(self.max_hits):
            tokens = [w.lower() for w in query.split() if len(w) > 2]
            chunks.append(
                UnifiedChunk(
                    id=str(uuid.uuid4()),
                    text_content=f"Live result {i+1} for: {query}",
                    source_type="mock_live",
                    source_id=f"mock-live-{i+1}",
                    sparse_tokens={"tokens": tokens},
                    metadata=UnifiedChunkMetadata(
                        hierarchical_tags=["mock", "live"],
                        parent_structure=None,
                        file_path_or_url="mock://live",
                        custom_attributes={"hit_number": i + 1},
                    ),
                )
            )
        return chunks
