## System Architecture – Epstein RAG with Cognee

This document describes the technical architecture for a three-step buildout:

1. **Step 1 – Document-Only RAG**  
   Baseline retrieval-augmented generation over Epstein documents using embeddings and a vector store.
2. **Step 2 – Graph Knowledge Layer**  
   Entity, relation, and incident extraction into a knowledge graph for structured querying.
3. **Step 3 – Hybrid RAG (Graph + Vector)**  
   A combined pipeline that uses both graph traversals and vector retrieval, with query understanding and safety layers.

---

## Global Architecture Overview

At a high level the system consists of:

- **Client / API Layer**
  - `FastAPI` (or similar) that exposes:
    - `/ask` – question answering (RAG, and later hybrid).
    - `/graph/query` – graph-specific queries (Step 2+).
    - `/feedback` – record user feedback and corrections.

- **Cognee Application**
  - Coordinates ingestion, enrichment, retrieval, and generation.
  - Hosts custom loaders, transformers, and query pipelines.

- **Storage**
  - **Raw Documents Store**: file system or object storage (e.g. S3/MinIO).
  - **Metadata Store**: relational DB (e.g. Postgres) for documents, runs, feedback.
  - **Vector Store**: Qdrant, pgvector, or Chroma for embeddings and chunk metadata.
  - **Graph Store**: Cognee’s internal graph or external graph DB (e.g. Neo4j).

- **LLM / Embedding Providers**
  - Text embedding model (for chunks and queries).
  - LLM for:
    - Question understanding.
    - Entity/relation/incident extraction.
    - Answer generation.

- **Observability**
  - Centralized logging (queries, retrieval results, LLM prompts/responses).
  - Metrics (latency, error rates, usage).
  - Tracing (end-to-end spans across ingestion and query pipelines).

---

## Step 1 – Document-Only RAG Architecture

### Components

- **Ingestion Pipeline**
  - **Loader**:
    - Reads documents from a configured source (`DOCS_PATH` or S3 bucket).
    - Produces `Document` objects with:
      - `document_id`
      - `source_uri`
      - `document_type` (e.g. court filing, email, note)
      - `created_at` / `date` (if known)
      - raw text or extracted text.
  - **Parser / Normalizer**:
    - PDF → text (e.g. `pypdf`, `pdfplumber`).
    - Emails → header/body/attachments parsing.
    - OCR for scans (optional, pluggable).
    - Text cleanup (whitespace, boilerplate removal).

- **Chunking & Embeddings**
  - **Chunker Transformer**:
    - Breaks documents into chunks of \(N\) tokens (e.g. 500–800) with overlap.
    - Respects structure (headings, paragraphs) where available.
    - Outputs `Chunk` objects with:
      - `chunk_id`
      - `document_id`
      - `page` / `section` info where possible
      - text content.
  - **Embedding Transformer**:
    - Calls embedding model for each chunk.
    - Stores vectors + metadata in the vector store.

- **RAG Query Pipeline**
  - **Query Embedding**:
    - Embed user question using same embedding model as chunks.
  - **Vector Retrieval**:
    - k-NN search in the vector store with optional filters (e.g. date range).
  - **Answer Generation**:
    - Prompt LLM with:
      - User query.
      - Top-K retrieved chunks (with document and page/chunk ids).
      - Instructions to:
        - Answer strictly from provided context.
        - Include inline citations pointing to chunks/documents.

### Data Flow (Step 1)

1. **Ingestion**: Raw files → Loader → Parsed & normalized documents.
2. **Indexing**: Documents → Chunker → Embeddings → Vector store.
3. **Query**: User question → Embed → Vector search → LLM with retrieved chunks → Answer with citations.

This step establishes stable, production-ready ingestion and retrieval infrastructure that later steps will build on.

---

## Step 2 – Graph Knowledge Layer Architecture

### Data Model

- **Nodes**
  - `Person`:
    - Fields: `id`, `canonical_name`, `aliases`, `roles`, `notes`.
  - `Organization`:
    - Fields: `id`, `name`, `type`, `jurisdiction`.
  - `Location`:
    - Fields: `id`, `name`, `address`, `geo` (optional).
  - `Incident`:
    - Fields: `id`, `title`, `description`, `date`, `date_range`, `status`.
  - `Document`:
    - Fields: `id`, `source_uri`, `document_type`, `date`, `hash`.

- **Edges**
  - `KNOWS` / `ASSOCIATED_WITH` (Person–Person).
  - `AFFILIATED_WITH` (Person–Organization).
  - `OCCURRED_AT` (Incident–Location).
  - `INVOLVES` (Incident–Person/Organization).
  - `MENTIONED_IN` (Entity/Incident–Document or Chunk).

Each node/edge carries:
- Evidence references (document id, page, chunk id, text span).
- Confidence score and extraction run id.

### Extraction & Graph-Build Pipeline

- **Entity Extraction Transformer**
  - Input: Chunks from Step 1 (and their metadata).
  - Output: Candidate entities with:
    - Type, canonical name, aliases.
    - Source chunk ids and text spans.

- **Relation & Incident Extraction Transformer**
  - Input: Chunks (possibly grouped by document or window).
  - Output:
    - Relations: `from_entity`, `to_entity`, `type`, evidence chunks, confidence.
    - Incidents: `id` (or generated), `participants`, `location`, `date`, `description`, evidence chunks, confidence.

- **Entity Resolution & Graph Updater**
  - Resolves duplicates and aliases:
    - String similarity, context, and possibly human-validated corrections.
  - Upserts nodes and edges into the graph store.
  - Maintains backlinks from graph objects to source chunks and documents.

### Graph Query API

- **Graph Traversal Service**
  - Provides high-level methods:
    - `get_associates(person_id, depth=1)`
    - `get_incidents_for_person(person_id, date_range?)`
    - `get_incident_details(incident_id)`
  - Encapsulates the underlying graph database.

- **HTTP Endpoints**
  - `/graph/query`:
    - Accepts structured queries (e.g. JSON specifying entities, relation types, filters).
    - Returns:
      - Nodes and edges.
      - Evidence references (chunk/document ids).
  - `/graph/search`:
    - Given a name or label, returns candidate nodes for disambiguation.

### Data Flow (Step 2)

1. **Reuse Step 1 outputs**: Use existing chunks and metadata as input.
2. **Run extraction**: Apply entity and relation/incident transformers.
3. **Update graph**: Upsert entities, relations, and incidents with evidence.
4. **Serve graph queries**: Clients (or later, the RAG pipeline) call graph services.

---

## Step 3 – Hybrid RAG Architecture

### Query Understanding Layer

- **Classifier**
  - LLM-based classification of incoming queries into types:
    - `RELATION_QUERY`, `INCIDENT_QUERY`, `GENERAL_FACTUAL`, `SUMMARY`, etc.
  - Extracted elements:
    - Entity names, dates, locations, incident descriptors.

- **Router**
  - Based on classification and extracted elements:
    - Decides what combination of:
      - Graph traversal.
      - Vector search.
      - Direct LLM answer (for meta/system questions).

### Hybrid Retrieval Layer

- **Graph Retrieval**
  - For relation/incident-focused queries:
    - Resolve entity names to graph node ids (via `/graph/search`).
    - Perform bounded-depth traversals to collect:
      - People, organizations, incidents, locations linked to query.
    - Retrieve associated evidence references (chunks/documents).

- **Vector Retrieval (from Step 1)**
  - For all query types:
    - Embed query with optional hints (e.g. tags like “incident”, “relation”).
    - Run vector search, optionally filtered by:
      - Document type.
      - Date range.
      - Documents linked to specific graph nodes (for focused search).

- **Evidence Aggregation**
  - Normalize and merge results from:
    - Graph (facts + evidence references).
    - Vector store (chunks + similarity scores).
  - Rank by:
    - Graph proximity.
    - Vector similarity.
    - Source reliability and confidence.
  - Select top-K chunks and top-N graph facts for the final LLM call.

### Answer Generation Layer

- **Prompt Construction**
  - Build a structured prompt that includes:
    - User query and any interpreted intent.
    - Graph facts (entities, relations, incidents) with ids.
    - Text chunks (with document + page/chunk ids).
    - Explicit instructions:
      - Use only provided information.
      - Prefer corroborated facts.
      - Clearly indicate uncertainty or lack of evidence.
      - Always return citations (document/page/chunk, graph ids).

- **Response Schema**
  - LLM output should be structured as:
    - Natural language answer.
    - Machine-readable list of citations.
    - Optional structured summary (e.g. list of people, incidents) for UI rendering.

### Safety, Access Control, and Observability

These concerns span all steps but become critical in the hybrid system:

- **Safety**
  - Prompt-level guardrails to avoid:
    - Speculative accusations beyond evidence.
    - Hallucinated relationships or incidents.
  - Optional moderation layer to filter or redact content.

- **Access Control**
  - Authentication (e.g. JWT/OIDC) for all APIs.
  - Role-based authorization:
    - Admins (full access).
    - Analysts (limited data, full query).
    - Read-only (constrained views).

- **Observability**
  - Logs:
    - Incoming queries.
    - Retrieval results (graph + vector).
    - Prompts and responses (with redaction as required).
  - Metrics:
    - Latency (per stage, per endpoint).
    - Error rates.
    - LLM usage and cost.
  - Traces:
    - Spans for ingestion, indexing, and query flows, from API entry to LLM and back.

---

## Implementation Phasing Summary

- **Step 1 – Document-Only RAG**
  - Build ingestion, chunking, embedding, vector store, and basic RAG API.
  - Focus: correctness, stability, and observability of core pipeline.

- **Step 2 – Graph Knowledge Layer**
  - Build extraction transformers, entity resolution, and graph store.
  - Expose graph query APIs and start answering relation/incident questions structurally.

- **Step 3 – Hybrid RAG**
  - Add query understanding, hybrid retrieval, and combined answer generation.
  - Harden safety, access control, evaluation, and production operations.

This staged approach ensures you always have a working, deployable system while incrementally adding sophistication and capabilities.

