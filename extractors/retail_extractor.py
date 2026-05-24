"""
Retail Data Extractor
======================
Concrete ``DataExtractor`` that reads the retail CSV dump (articles +
category hierarchy) and maps every FOOD-catalog item into a strictly
conformant ``UnifiedChunk``.

Category chains are built by walking the parent tree in
``d_categorizacion_tbl.csv``, yielding tags like::

    SUPERMERCADO > Frescos > Lácteos > Leche > Entera

Version deduplication: items sharing the same *codart* are collapsed to
the single row with the highest version (lexicographic), keeping the
most recent date on ties.
"""

from __future__ import annotations

import csv
import logging
import re
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from core.models import UnifiedChunk, UnifiedChunkMetadata
from extractors.base import DataExtractor

logger = logging.getLogger(__name__)

_SOURCE_TYPE = "retail_product"

_CSV_ARTICLES = Path.home() / "raw_data" / "d_articulos.csv"
_CSV_CATEGORIES = Path.home() / "raw_data" / "d_categorizacion_tbl.csv"

_STOPWORDS = {
    "the", "and", "for", "are", "but", "not", "you", "all", "can",
    "had", "her", "was", "one", "our", "out", "day", "get", "has",
    "him", "his", "how", "its", "may", "new", "now", "old", "see",
    "two", "who", "boy", "did", "she", "use", "way", "many",
    "de", "la", "el", "los", "las", "un", "una", "del", "en",
    "con", "por", "para", "sin", "que", "los", "las", "y", "o",
    "e", "a", "es", "se", "su", "al",
}

_SENTINEL_CODES = {"#GIFTCARD", "00000001"}


class RetailDataExtractor(DataExtractor):
    """Extract ``UnifiedChunk`` objects from the retail CSV dump.

    Parameters
    ----------
    categories_csv:
        Path to ``d_categorizacion_tbl.csv`` (category tree).
    activity_filter:
        Only process rows where ``uid == activity_filter`` (default ``FOOD``).
    """

    def __init__(
        self,
        *,
        categories_csv: str = str(_CSV_CATEGORIES),
        activity_filter: str = "FOOD",
    ) -> None:
        self._categories_csv = Path(categories_csv)
        self._activity_filter = activity_filter
        self._cat_tree: Dict[str, Tuple[str, str]] = {}  # cat_id -> (name, parent_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_chunks(self, source_path: str) -> List[UnifiedChunk]:
        articles_csv = Path(source_path)
        if not articles_csv.exists():
            raise FileNotFoundError(f"Articles CSV not found: {articles_csv}")

        self._load_category_tree()
        items = self._parse_articles(articles_csv)
        items = self._dedup_by_version(items)

        chunks: List[UnifiedChunk] = []
        for item in items:
            try:
                chunk = self._item_to_chunk(item)
                chunks.append(chunk)
            except Exception:
                logger.exception("Failed to build chunk for item %s", item.get("codart", "?"))

        logger.info(
            "RetailDataExtractor produced %d chunk(s) from %d article(s) (activity=%s).",
            len(chunks),
            len(items),
            self._activity_filter,
        )
        return chunks

    # ------------------------------------------------------------------
    # Category tree
    # ------------------------------------------------------------------

    def _load_category_tree(self) -> None:
        if not self._categories_csv.exists():
            logger.warning("Categories CSV not found at %s — category chains will be flat.",
                           self._categories_csv)
            return

        with self._categories_csv.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            for row in reader:
                if len(row) < 3:
                    continue
                if row[0].strip() != self._activity_filter:
                    continue
                cat_id = row[1].strip()
                cat_name = row[2].strip()
                parent_id = row[6].strip() if len(row) > 6 else ""
                if cat_id and cat_name:
                    self._cat_tree[cat_id] = (cat_name, parent_id)

        logger.info("Loaded %d category node(s) from %s.", len(self._cat_tree), self._categories_csv)

    def _build_chain(self, leaf_cat_id: str) -> List[str]:
        """Walk parent links upward, returning root→leaf ordered names."""
        chain: List[str] = []
        visited: set[str] = set()
        current = leaf_cat_id.strip()
        while current and current in self._cat_tree and current not in visited:
            name, parent = self._cat_tree[current]
            chain.append(name)
            visited.add(current)
            current = parent.strip() if parent else ""
        chain.reverse()
        return chain

    # ------------------------------------------------------------------
    # Article parsing
    # ------------------------------------------------------------------

    def _parse_articles(self, path: Path) -> List[Dict[str, str]]:
        items: List[Dict[str, str]] = []
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            for row in reader:
                if len(row) < 38:
                    continue
                if row[0].strip() != self._activity_filter:
                    continue
                codart = row[1].strip()
                if codart in _SENTINEL_CODES:
                    continue
                if not codart or not row[2].strip():
                    continue

                items.append({
                    "activity_uid": row[0].strip(),
                    "codart": codart,
                    "desart": row[2].strip(),
                    "formato": (row[3] or "").strip(),
                    "codfam": (row[4] or "").strip(),
                    "desfam": (row[5] or "").strip(),
                    "codseccion": (row[6] or "").strip(),
                    "dessecion": (row[7] or "").strip(),
                    "codcat": (row[8] or "").strip(),
                    "descat": (row[9] or "").strip(),
                    "codpro": (row[10] or "").strip(),
                    "despro": (row[11] or "").strip(),
                    "codimp": (row[17] or "").strip(),
                    "desimp": (row[18] or "").strip(),
                    "activo": (row[20] or "").strip(),
                    "version": (row[36] or "").strip(),
                    "desmarca": (row[37] or "").strip(),
                    "fecha_version": (row[35] or "").strip(),
                    "medida_alt_code": (row[29] or "").strip(),
                    "medida_alt_desc": (row[30] or "").strip(),
                })
        return items

    # ------------------------------------------------------------------
    # Version deduplication
    # ------------------------------------------------------------------

    @staticmethod
    def _version_sort_key(version: str) -> Tuple[int, int]:
        if not version:
            return (0, 0)
        try:
            return (1, int(version))
        except ValueError:
            return (1, 0)

    def _dedup_by_version(self, items: List[Dict[str, str]]) -> List[Dict[str, str]]:
        grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        for item in items:
            grouped[item["codart"]].append(item)

        deduped: List[Dict[str, str]] = []
        for codart, group in grouped.items():
            if len(group) == 1:
                deduped.append(group[0])
            else:
                group.sort(
                    key=lambda x: (self._version_sort_key(x["version"]), x.get("fecha_version", "")),
                    reverse=True,
                )
                deduped.append(group[0])

        removed = len(items) - len(deduped)
        if removed:
            logger.info("Version dedup removed %d older version(s) (kept latest per codart).", removed)
        return deduped

    # ------------------------------------------------------------------
    # Chunk construction
    # ------------------------------------------------------------------

    def _item_to_chunk(self, item: Dict[str, str]) -> UnifiedChunk:
        codart = item["codart"]
        desart = item["desart"]
        desfam = item["desfam"]
        dessecion = item["dessecion"]
        desmarca = item["desmarca"]
        codcat = item["codcat"]
        version = item["version"]

        chain = self._build_chain(codcat) if codcat else []

        source_id = f"{codart}"
        if version:
            source_id += f":v{version}"

        hierarchical_tags = ["retail", self._activity_filter.lower()] + [
            name.lower() for name in chain
        ]

        text_content = self._build_text(item, chain)

        tokens = self._tokenize(desart, desfam, dessecion, desmarca, *chain)

        return UnifiedChunk(
            id=str(uuid.uuid4()),
            text_content=text_content,
            source_type=_SOURCE_TYPE,
            source_id=source_id,
            sparse_tokens={"tokens": tokens},
            metadata=UnifiedChunkMetadata(
                hierarchical_tags=hierarchical_tags,
                parent_structure=chain[-1] if chain else None,
                file_path_or_url=f"csv://retail_catalog/{codart}",
                custom_attributes={
                    "codart": codart,
                    "desart": desart,
                    "desfam": desfam,
                    "dessecion": dessecion,
                    "desmarca": desmarca,
                    "category_chain": " > ".join(chain),
                },
            ),
        )

    @staticmethod
    def _build_text(item: Dict[str, str], chain: List[str]) -> str:
        parts: List[str] = []

        desc = item["desart"]
        if desc:
            parts.append(desc)

        desfam = item["desfam"]
        if desfam:
            parts.append(f"Familia: {desfam}")

        dessecion = item["dessecion"]
        if dessecion:
            parts.append(f"Sección: {dessecion}")

        if chain:
            parts.append(f"Categoría: {' > '.join(chain)}")

        desmarca = item["desmarca"]
        if desmarca:
            parts.append(f"Marca: {desmarca}")

        despro = item["despro"]
        if despro and despro != "PROVEEDOR POR DEFECTO":
            parts.append(f"Proveedor: {despro}")

        return ". ".join(parts) + "."

    @staticmethod
    def _tokenize(*texts: str) -> List[str]:
        tokens: set[str] = set()
        for text in texts:
            if not text:
                continue
            words = re.findall(r"[A-Za-zÀ-ÿ0-9_]{3,}", text)
            for w in words:
                wl = w.lower()
                if wl not in _STOPWORDS:
                    tokens.add(wl)
        return sorted(tokens)
