"""
Invariant Data Contract — UnifiedChunk
========================================
Domain-agnostic schema used by every ingestion worker, storage indexer,
and retrieval port in the Hybrid RAG engine.

DO NOT add domain-specific fields here.  Use metadata.custom_attributes
for downstream specialization (Linux kernel structs, email threads, etc.).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class UnifiedChunkMetadata(BaseModel):
    """Structured metadata container with an open domain-specific escape hatch."""

    hierarchical_tags: List[str] = Field(
        default_factory=list,
        description="Ordered taxonomy tags for result filtering / faceting.",
    )
    parent_structure: Optional[str] = Field(
        default=None,
        description="Optional structural namespace (e.g. function signature, thread root).",
    )
    file_path_or_url: str = Field(
        ...,
        description="Canonical source locator.",
    )
    custom_attributes: Dict[str, Any] = Field(
        default_factory=dict,
        description="Open key-value map for domain-specific payload passthrough. "
                    "Engine layers MUST NOT validate keys inside this dict.",
    )


class UnifiedChunk(BaseModel):
    """
    Immutable data contract for a single retrievable chunk.

    All fields are required except those explicitly marked Optional.
    """

    id: str = Field(
        ...,
        description="UUID-v4 string identifying this chunk globally.",
    )
    text_content: str = Field(
        ...,
        description="Raw text block fed to embedding engines and LLM context windows.",
    )
    source_type: str = Field(
        ...,
        description="Logical domain discriminator (e.g. 'linux_kernel', 'lkml_email').",
    )
    source_id: str = Field(
        ...,
        description="Unique identifier from the original upstream data source.",
    )
    sparse_tokens: Dict[str, List[str]] = Field(
        default_factory=lambda: {"tokens": []},
        description="Exact keyword list for sparse / inverted-index retrieval.",
    )
    metadata: UnifiedChunkMetadata = Field(
        ...,
        description="Structured metadata with open custom_attributes extension point.",
    )

    model_config = {
        "frozen": True,          # Immutable at runtime
        "extra": "forbid",       # Reject unexpected top-level keys
    }
