# AGENTS.md — Project Scope & Architectural Context

## Project Overview
You are building a production-grade, highly scalable Hybrid RAG (Retrieval-Augmented Generation) System designed with a strict **Hexagonal Architecture (Ports & Adapters)**. 

The core system engine must remain completely domain-agnostic, invariant, and blind to the underlying nature of the data it processes. The initial data domain used to stress-test this system is the **Linux Kernel Ecosystem** (source code files via AST parsing and live mailing lists via `.mbox`), but the system must be capable of swapping to a completely different domain (e.g., retail catalogs, medical text) solely by changing peripheral adapters.

The entire engine will eventually be exposed as a **Model Context Protocol (MCP)** server deployed via Docker on a remote VPS.

---

## Technical Stack Guidelines
*   **Containerization:** Docker / Docker Compose
*   **Vector Database:** Qdrant (with on-disk indexing and `int8` Scalar Quantization)
*   **AI Gateway:** LiteLLM Proxy (OpenAI-compatible endpoints)
*   **Code Parsing:** Tree-sitter (for structural, AST-aware chunking)
*   **Reranking:** Local ONNX Cross-Encoder or VPS-hosted microservice

---

## Core Invariant Contract: The `UnifiedChunk`

Every data ingestion worker, storage indexer, and retrieval port MUST strictly communicate using this immutable data structure. Do NOT modify this schema to accommodate specific domain attributes.

```json
{
  "id": "string (uuid-v4)",
  "text_content": "string (the raw content block passed to embedding engines & LLM context)",
  "source_type": "string (e.g., linux_kernel, lkml_email, retail_product)",
  "source_id": "string (unique identifier from the original data source)",
  "sparse_tokens": {
    "tokens": ["list", "of", "exact", "keywords", "for", "sparse", "vector", "matching"]
  },
  "metadata": {
    "hierarchical_tags": ["string", "array"],
    "parent_structure": "string (optional structural namespace context, e.g. function signature)",
    "file_path_or_url": "string",
    "custom_attributes": {
      "__KEY_VALUE_PAIRS_FOR_DOMAIN_SPECIFIC_DATA_GO_HERE__": "any"
    }
  }
}
Strict Architectural Guardrails (Read Before Coding)
The Core Separation Rule: The directories handling vector database interactions (Qdrant), mathematical merging (RRF), and API delivery (MCP) must contain zero references to C-code, email formats, kernel structures, or any specific domain.

Pluggable Ingestion: All parsing logic must live inside isolated extractor classes implementing an abstract DataExtractor port: extract_chunks(input_source) -> List<UnifiedChunk>.

Memory Limits: The target deployment environment is a standard cloud VPS. Assume millions of items will be indexed. Memory-hogging array operations or unquantized in-memory storage configurations are explicit failures.

No Direct Model Dependencies: Do not hardcode specific OpenAI or Anthropic SDK clients into the core services. Route all completions and embeddings through the system's OpenAI-compatible LiteLLM proxy base URL.

Phase-by-Phase Roadmap
Phase 1: Agnostic Storage Infrastructure & Contracts
Task 1.1: Create a docker-compose.yml for Qdrant.

Task 1.2: Configure Qdrant's config.yaml to enforce int8 scalar quantization for memory optimization.

Task 1.3: Write the initialization script to set up a collection supporting dual-vector storage (Dense vector configurations + Sparse vector configurations).

Task 1.4: Standardize the code interface for UnifiedChunk.

Phase 2: Pluggable Extractor Framework & Ingestion Workers
Task 2.1: Code the DataExtractor abstract base class/interface.

Task 2.2: Build the LinuxCodeExtractor utilizing Tree-sitter to chunk C code at structural function/struct boundaries, mapping them to UnifiedChunk schemas.

Task 2.3: Build the MailingListExtractor to parse .mbox conversational threads into chunk segments.

Task 2.4: Build an asynchronous ingestion queue runner that aggregates UnifiedChunk sets, requests embeddings in batches, and performs bulk upserts into Qdrant.

Phase 3: Domain-Blind Fusion & Retrieval Engine
Task 3.1: Implement a parallel query mechanism hitting Qdrant's dense and sparse vector indices simultaneously.

Task 3.2: Write a pure mathematical implementation of Reciprocal Rank Fusion (RRF) to combine distinct rank result tables into a single sorted array.

Task 3.3: Add an asynchronous hook/port to fetch real-time data from an external feed during a search loop, mapping live payloads instantly into transient UnifiedChunk structures.

Task 3.4: Set up a lightweight, local cross-encoder reranking layer to score the top-fused chunks.

Phase 4: Standardized Protocol Delivery (MCP)
Task 4.1: Wrap the Phase 3 search service inside a Model Context Protocol (MCP) server.

Task 4.2: Expose generic tools (e.g., hybrid_search_explorer(query, domain_flags)) that pass options directly down to the infrastructure layer.

Task 4.3: Validate functionality across automated Docker deployments on the VPS by mounting the MCP server directly into local AI workspace clients.
