from __future__ import annotations

from dataclasses import dataclass, field

from core.models import UnifiedChunk


@dataclass
class SearchResult:
    chunk: UnifiedChunk
    relevance_score: float = 0.0
    rrqf_score: float = 0.0
