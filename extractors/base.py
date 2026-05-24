"""
Abstract Base Extraction Port
==============================
Domain-agnostic interface that every concrete extractor must implement.
Core infrastructure remains blind to parsing toolkits.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from core.models import UnifiedChunk


class DataExtractor(ABC):
    """
    Abstract port for data extraction.

    All concrete extractors (code, email, retail catalog, etc.) must
    implement ``extract_chunks`` and return a list of ``UnifiedChunk``
    objects that conform to the invariant contract defined in
    ``core.models``.
    """

    @abstractmethod
    def extract_chunks(self, source_path: str) -> List[UnifiedChunk]:
        """
        Extract domain-specific data from *source_path* and map each
        logical unit into a ``UnifiedChunk``.

        Parameters
        ----------
        source_path:
            Filesystem path (or URI) pointing to the raw upstream data.

        Returns
        -------
        List[UnifiedChunk]
            Ordered list of chunks ready for downstream batch ingestion.
        """
        ...
