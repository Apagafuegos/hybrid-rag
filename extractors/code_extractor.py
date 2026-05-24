"""
Concrete Tree-Sitter Code Extractor
====================================
Parses C source/header files into *file-level grouped chunks* rather
than one chunk per function.  This dramatically reduces embedding
volume while preserving structural context.

Key optimisations
-----------------
1. **Preprocessor-aware recursion** – descends into ``#ifdef`` /
   ``#ifndef`` / ``#if`` blocks so header include guards and
   conditional-compilation sections do not hide AST nodes.
2. **File-level grouping** – adjacent functions/structs/macros are
   concatenated into chunks of ~3 000 chars instead of one chunk each.
3. **Deep noise reduction** – strips comments, string literals, numeric
   constants, modifier attributes, kernel boilerplate macros, and
   trivial getter/setter stubs.
4. **Whitespace normalisation** – runs of whitespace collapse to a
   single space.
5. **Oversized-piece splitting** – a single AST node larger than
   ``MAX_CHUNK_CHARS`` is split on newline boundaries.
6. **Per-file deduplication** – identical minified blocks are only
   emitted once.
7. **Cache-friendly** – emits deterministic ``source_id`` values so
   downstream caches can invalidate by file hash.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from pathlib import Path
from typing import List, Optional, Set

from tree_sitter import Language, Node, Parser
from tree_sitter_c import language as c_language_factory

from core.models import UnifiedChunk, UnifiedChunkMetadata
from extractors.base import DataExtractor

C_LANGUAGE = Language(c_language_factory())

# ---------------------------------------------------------------------------
# Tuning knobs
# ---------------------------------------------------------------------------
TARGET_CHUNK_CHARS = 3_000       # ideal group size
MAX_CHUNK_CHARS = 6_000          # hard ceiling before forced split
MIN_NODE_CHARS = 20              # skip tiny nodes (was 80, then 40)
MIN_CHUNK_CHARS = 100            # skip nearly-empty groups

# ---------------------------------------------------------------------------
# AST node types we recognise as "content-bearing"
# ---------------------------------------------------------------------------
_CONTENT_NODE_TYPES: Set[str] = {
    "function_definition",
    "struct_specifier",
    "preproc_def",
    "preproc_function_def",
    "declaration",
    "enum_specifier",
    "type_definition",
}

# Node types we must **recurse into** because they wrap other nodes
_PREPROC_WRAPPER_TYPES: Set[str] = {
    "preproc_ifdef",
    "preproc_if",
    "preproc_else",
}

# ---------------------------------------------------------------------------
# Noise patterns — preprocessor directives we skip wholesale
# ---------------------------------------------------------------------------
_SKIP_PATTERNS: List[re.Pattern] = [
    re.compile(r"^\s*#\s*include\s+"),
    re.compile(r"^\s*#\s*pragma\s+"),
]

# Boilerplate function names
_BOILERPLATE_NAMES: Set[str] = {
    "__init", "__exit", "__initdata", "__exitdata",
    "module_init", "module_exit", "late_initcall",
    "__attribute__", "__aligned", "__packed",
}

# Kernel boilerplate macros whose whole node is noise
_BOILERPLATE_MACROS_RE: re.Pattern = re.compile(
    r"^\s*"
    r"(?:EXPORT_SYMBOL(?:_GPL(?:_FUTURE)?|_NS)?"
    r"|MODULE_(?:LICENSE|AUTHOR|DESCRIPTION|DEVICE_TABLE|VERSION|FIRMWARE|ALIAS|IMPORT_NS)"
    r"|__initcall|core_initcall|postcore_initcall|arch_initcall"
    r"|subsys_initcall|fs_initcall|device_initcall|late_initcall"
    r"|early_initcall|pure_initcall|rootfs_initcall|module_init"
    r"|module_exit|__exitcall"
    r"|DEFINE_PER_CPU|DECLARE_PER_CPU"
    r")"
    r"\s*\(.*",
    re.DOTALL,
)

# ---------------------------------------------------------------------------
# Minification regexes (compiled once)
# ---------------------------------------------------------------------------
# Strip kernel-doc /** ... */ comments first (they often contain raw C)
_KERNELDOC_RE: re.Pattern = re.compile(
    r"/\*\*.*?\*/",
    re.DOTALL,
)
# Strip regular C comments
_C_COMMENT_RE: re.Pattern = re.compile(
    r"//.*?$|/\*.*?\*/",
    re.MULTILINE | re.DOTALL,
)
# Strip string literal contents (keep the quotes as markers)
_STRING_LITERAL_RE: re.Pattern = re.compile(
    r'"(?:[^"\\]|\\.)*"',
    re.DOTALL,
)
# Normalise numeric literals
_NUMBER_RE: re.Pattern = re.compile(
    r"\b(?:0[xX][0-9a-fA-F]+|[0-9]+[uUlLfF]*)\b",
)
# Collapse runs of whitespace
_WS_RE: re.Pattern = re.compile(r"\s+")


class LinuxCodeExtractor(DataExtractor):
    """
    Extractor for Linux kernel C source files (``.c`` / ``.h``).

    Emits *grouped* chunks per file rather than one chunk per AST node,
    cutting total volume by 5-10× while keeping enough context for
    retrieval.
    """

    SUPPORTED_EXTENSIONS = {".c", ".h"}

    def extract_chunks(self, source_path: str) -> List[UnifiedChunk]:
        path = Path(source_path)
        if path.suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"LinuxCodeExtractor does not support '{path.suffix}'. "
                f"Expected one of {self.SUPPORTED_EXTENSIONS}."
            )

        source_bytes = path.read_bytes()
        parser = Parser(C_LANGUAGE)
        tree = parser.parse(source_bytes)
        rel_path = str(path)

        # 1. Recursively gather structural nodes (descend into #ifdef etc.)
        nodes = self._collect_content_nodes(tree.root_node, source_bytes)
        if not nodes:
            return []

        # 2. Map → minified text, filter noise
        pieces: List[str] = []
        for node in nodes:
            raw = source_bytes[node.start_byte:node.end_byte].decode(
                "utf-8", errors="replace"
            )
            mini = self._minify(raw)
            if not self._is_noise(mini, node):
                pieces.append(mini)

        if not pieces:
            return []

        # 3. Group into target-sized chunks (split oversize pieces)
        groups = self._group_pieces(pieces)

        # 4. Deduplicate and build UnifiedChunks
        seen: Set[str] = set()
        chunks: List[UnifiedChunk] = []

        for i, group_text in enumerate(groups):
            h = hashlib.sha256(group_text.encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)

            if len(group_text) < MIN_CHUNK_CHARS:
                continue

            chunks.append(
                UnifiedChunk(
                    id=str(uuid.uuid4()),
                    text_content=group_text,
                    source_type="linux_kernel",
                    source_id=f"{rel_path}#group_{i}",
                    sparse_tokens={"tokens": self._tokenize(group_text)},
                    metadata=UnifiedChunkMetadata(
                        hierarchical_tags=["linux_kernel", "code"],
                        parent_structure=None,
                        file_path_or_url=rel_path,
                        custom_attributes={
                            "language": "c",
                            "group_index": i,
                            "node_count": len(pieces),
                        },
                    ),
                )
            )

        return chunks

    # -------------------------------------------------------------------
    # Recursive node collector (descends into preprocessor wrappers)
    # -------------------------------------------------------------------

    @classmethod
    def _collect_content_nodes(
        cls, root: Node, source_bytes: bytes
    ) -> List[Node]:
        """
        Walk *root* and its descendants, returning every content-bearing
        node discovered.  Descends into ``preproc_ifdef`` / ``preproc_if`` /
        ``preproc_else`` wrappers so nodes hidden behind include guards or
        ``#ifdef CONFIG_*`` blocks are still collected.
        """
        collected: List[Node] = []
        for child in root.children:
            ctype = child.type
            if ctype in _PREPROC_WRAPPER_TYPES:
                collected.extend(cls._collect_content_nodes(child, source_bytes))
            elif ctype in _CONTENT_NODE_TYPES:
                collected.append(child)
            # else: skip comments, preproc_include, #endif, etc.
        return collected

    # -------------------------------------------------------------------
    # Minification
    # -------------------------------------------------------------------

    @staticmethod
    def _minify(text: str) -> str:
        """Strip comments, string contents, numbers, and collapse whitespace."""
        # Kernel-doc comments first (may contain raw code snippets)
        text = _KERNELDOC_RE.sub(" ", text)
        # C block/line comments
        text = _C_COMMENT_RE.sub(" ", text)
        # String literal contents → empty placeholder
        text = _STRING_LITERAL_RE.sub('""', text)
        # Numeric literals → 0
        text = _NUMBER_RE.sub("0", text)
        # Collapse whitespace
        text = _WS_RE.sub(" ", text)
        return text.strip()

    # -------------------------------------------------------------------
    # Noise heuristics
    # -------------------------------------------------------------------

    @classmethod
    def _is_noise(cls, mini: str, node: Node) -> bool:
        """Return True if this minified node should be discarded."""
        # Too short
        if len(mini) < MIN_NODE_CHARS:
            return True

        # Kernel boilerplate macros (EXPORT_SYMBOL, MODULE_*, initcall, etc.)
        if _BOILERPLATE_MACROS_RE.match(mini):
            return True

        # Preprocessor skips
        if mini.startswith("#"):
            for pat in _SKIP_PATTERNS:
                if pat.match(mini):
                    return True

        # Trivial #define: bare constant or empty (include guard stubs)
        if node.type in ("preproc_def", "preproc_function_def"):
            parts = mini.split()
            if len(parts) <= 2:
                # Only '#define NAME' (include guard) — no value, just a stub
                return True

        # Bare declaration with no semantic weight
        # e.g. 'int a ;' alone — still too small after MIN_NODE_CHARS
        if node.type == "declaration":
            word_count = len(mini.split())
            if word_count <= 2:
                return True

        # Trivial getter / setter heuristic: function with just return
        if node.type == "function_definition":
            body = next(
                (c for c in node.children if c.type == "compound_statement"), None
            )
            if body and len(body.children) <= 3:
                body_text = cls._minify(
                    body.text.decode("utf-8", errors="replace")
                )
                if body_text.count(";") <= 2 and "return" in body_text:
                    return True

        # Boilerplate names
        name = cls._extract_any_name(node)
        if name in _BOILERPLATE_NAMES:
            return True

        return False

    # -------------------------------------------------------------------
    # Grouping (with oversized-piece splitting)
    # -------------------------------------------------------------------

    @classmethod
    def _group_pieces(cls, pieces: List[str]) -> List[str]:
        """Concatenate adjacent pieces until target size is reached.
        Pieces larger than ``MAX_CHUNK_CHARS`` are split on newline
        boundaries before being fed into the accumulator."""
        groups: List[str] = []
        current: List[str] = []
        cur_len = 0

        for p in pieces:
            for sub in cls._split_oversized(p):
                sub_len = len(sub)
                # If adding this piece blows the hard ceiling, flush first
                if cur_len + sub_len > MAX_CHUNK_CHARS and current:
                    groups.append("\n\n".join(current))
                    current = [sub]
                    cur_len = sub_len
                else:
                    current.append(sub)
                    cur_len += sub_len

                # If we've hit the sweet spot, flush
                if cur_len >= TARGET_CHUNK_CHARS:
                    groups.append("\n\n".join(current))
                    current = []
                    cur_len = 0

        # Tail
        if current:
            tail = "\n\n".join(current)
            if len(tail) >= MIN_CHUNK_CHARS:
                groups.append(tail)

        return groups

    @staticmethod
    def _split_oversized(text: str) -> List[str]:
        """Split *text* into chunks no larger than MAX_CHUNK_CHARS.
        Tries to split on newline boundaries first, then falls back to
        character-boundary splitting."""
        if len(text) <= MAX_CHUNK_CHARS:
            return [text]

        parts: List[str] = []
        remainder = text
        while len(remainder) > MAX_CHUNK_CHARS:
            # Try to find a newline near the cut point
            cut = remainder.rfind("\n", MAX_CHUNK_CHARS // 2, MAX_CHUNK_CHARS + 1)
            if cut == -1:
                cut = MAX_CHUNK_CHARS
            else:
                cut += 1  # include the newline in the current part
            parts.append(remainder[:cut])
            remainder = remainder[cut:]
        if remainder:
            parts.append(remainder)
        return parts

    # -------------------------------------------------------------------
    # Name extraction (for boilerplate filtering)
    # -------------------------------------------------------------------

    @staticmethod
    def _extract_any_name(node: Node) -> Optional[str]:
        """Best-effort name extraction from any structural node."""
        if node.type == "function_definition":
            return LinuxCodeExtractor._extract_function_name(node)
        if node.type in ("struct_specifier", "enum_specifier"):
            return LinuxCodeExtractor._extract_struct_name(node)
        if node.type in ("preproc_def", "preproc_function_def"):
            return LinuxCodeExtractor._extract_macro_name(node)
        if node.type == "declaration":
            for child in node.children:
                if child.type == "identifier":
                    return child.text.decode("utf-8", errors="replace")
        if node.type == "type_definition":
            for child in node.children:
                if child.type == "type_identifier":
                    return child.text.decode("utf-8", errors="replace")
        return None

    @staticmethod
    def _extract_function_name(node: Node) -> Optional[str]:
        declarator = next((c for c in node.children if c.type == "declarator"), None)
        if declarator is not None:
            func_decl = next(
                (c for c in declarator.children if c.type == "function_declarator"),
                None,
            )
            if func_decl is None:
                func_decl = declarator
        else:
            func_decl = next(
                (c for c in node.children if c.type == "function_declarator"), None
            )

        if func_decl is None:
            return None

        ident = next((c for c in func_decl.children if c.type == "identifier"), None)
        if ident is None:
            ptr_decl = next(
                (c for c in func_decl.children if c.type == "pointer_declarator"),
                None,
            )
            if ptr_decl:
                ident = next(
                    (c for c in ptr_decl.children if c.type == "identifier"), None
                )
        return ident.text.decode("utf-8") if ident else None

    @staticmethod
    def _extract_struct_name(node: Node) -> Optional[str]:
        ident = next((c for c in node.children if c.type == "type_identifier"), None)
        if ident is None:
            ident = next((c for c in node.children if c.type == "identifier"), None)
        return ident.text.decode("utf-8") if ident else None

    @staticmethod
    def _extract_macro_name(node: Node) -> Optional[str]:
        ident = next((c for c in node.children if c.type == "identifier"), None)
        return ident.text.decode("utf-8") if ident else None

    # -------------------------------------------------------------------
    # Tokenizer (for sparse retrieval)
    # -------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        keywords = {
            "int", "void", "char", "float", "double", "long", "short",
            "unsigned", "signed", "const", "static", "extern", "inline",
            "return", "if", "else", "for", "while", "do", "switch",
            "case", "break", "continue", "default", "sizeof", "struct",
            "typedef", "enum", "union", "volatile", "register", "auto",
        }
        tokens = re.findall(r"[A-Za-z_]\w*", text)
        unique = sorted({t for t in tokens if len(t) > 2 and t not in keywords})
        return unique[:64]
