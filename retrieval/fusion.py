"""
Reciprocal Rank Fusion (RRF) Engine
====================================
Pure mathematical implementation of RRF for unifying ranked result lists.

Input: arbitrary list of ranked ``UnifiedChunk`` lists.
Output: single deduplicated list sorted by descending RRF score.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from core.models import UnifiedChunk

logger = logging.getLogger(__name__)

# Standard rank-smoothing constant per the RRF literature.
K_CONSTANT = 60


def reciprocal_rank_fusion(
    ranked_lists: List[List[UnifiedChunk]],
    *,
    k: int = K_CONSTANT,
) -> List[Tuple[float, UnifiedChunk]]:
    """
    Fuse multiple ranked result lists into one using Reciprocal Rank Fusion.

    The RRF score for a document ``d`` is:

        RRF_Score(d) = sum_{m in M} (1 / (k + r_m(d)))

    where ``r_m(d)`` is the 1-based rank of ``d`` in list ``m``.

    Parameters
    ----------
    ranked_lists:
        Each inner list is an ordered array of ``UnifiedChunk`` objects
        representing one search modality (dense, sparse, live, etc.).
    k:
        Rank-smoothing constant (default 60).

    Returns
    -------
    List[Tuple[float, UnifiedChunk]]
        Deduplicated (RRF_score, chunk) pairs sorted by descending RRF score.
    """
    # Accumulator: chunk_id -> (cumulative_score, chunk)
    scores: Dict[str, Dict[str, Any]] = {}

    for result_list in ranked_lists:
        for rank_1based, chunk in enumerate(result_list, start=1):
            dedup_key = chunk.id if chunk.id else chunk.source_id

            if dedup_key not in scores:
                scores[dedup_key] = {"score": 0.0, "chunk": chunk}

            scores[dedup_key]["score"] += 1.0 / (k + rank_1based)

    sorted_entries = sorted(scores.values(), key=lambda e: e["score"], reverse=True)

    logger.info(
        "RRF fused %d input lists into %d unique chunks",
        len(ranked_lists),
        len(sorted_entries),
    )
    return [(entry["score"], entry["chunk"]) for entry in sorted_entries]
