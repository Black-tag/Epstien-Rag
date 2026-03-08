## Epstein RAG with Cognee – Three-Step Plan

This repository describes a production-ready plan for building a Retrieval-Augmented Generation (RAG) system over Epstein-related documents using **Cognee** in three incremental, deployable steps:

- **Step 1 – Document-Only RAG**: Ingest documents, build embeddings, and answer questions from retrieved chunks.
- **Step 2 – Graph Knowledge Layer**: Extract entities, relations, and incidents into a graph for structured querying.
- **Step 3 – Hybrid RAG (Graph + Vector)**: Combine graph traversals with vector search for more precise, explainable answers.

The goal is to answer **relation** questions (who/what is connected to whom/what) and **incident** questions (what happened, where, when, who was involved) with **traceable sources**, **safety controls**, and **observability**.

---

## High-Level Roadmap

- **Step 1 – Generate answers from documents (baseline RAG)**
  - Ingest Epstein documents into Cognee.
  - Parse and clean text, chunk documents, and generate embeddings.
  - Store chunks in a vector database and expose a simple RAG API to generate answers using only document chunks.

- **Step 2 – Add graph knowledge (knowledge graph)**
  - Run entity and relation extraction over chunks.
  - Build a graph of people, organizations, locations, and incidents, with edges backed by evidence from documents.
  - Expose graph-first query capability for relation/incident questions independent of the RAG layer.

- **Step 3 – Hybrid RAG (graph + vector)**
  - Use query understanding to route to graph traversal, vector search, or both.
  - Merge graph results and vector-retrieved chunks into a consolidated evidence set for the LLM.
  - Enforce safety, access control, and provide full citations and graph snippets in answers.

See `ARCHITECTURE.md` for detailed architecture for each step.

---

## Step 1 – Document-Only RAG (Baseline)

**Goal**: Quickly stand up a basic but production-quality RAG that answers questions only from documents, with solid ingestion and retrieval.

- **Ingestion**
  - Configure Cognee to read Epstein documents from a local folder or object store.
  - Implement parsing for PDFs, emails, and other formats, plus OCR for scanned documents where needed.
  - Normalize text (whitespace, encoding, boilerplate removal) and attach rich metadata (document type, date, source).

- **Chunking & Embeddings**
  - Apply structure-aware chunking (e.g. section/paragraph boundaries where available) with token-based limits and overlap.
  - Generate embeddings using a production-ready model (OpenAI, local model, or provider configured in `.env`).
  - Store chunks and embeddings in a vector database (Qdrant, pgvector, or Cognee’s configured backend).

- **Answering**
  - Implement a simple RAG chain in Cognee:
    - Embed the user query.
    - Retrieve top-K relevant chunks.
    - Call an LLM with instructions to answer **only from those chunks**, with citations.
  - Expose an `/ask` endpoint (e.g. via `FastAPI`) for external clients.

Deliverable of Step 1: a working, containerized service that can answer factual questions from the corpus with citations from the underlying documents.

---

## Step 2 – Graph Knowledge Layer

**Goal**: Model and store **entities**, **relations**, and **incidents** so relation/incident questions can be answered structurally, not just via free-text RAG.

- **Data Model**
  - Entities: `Person`, `Organization`, `Location`, `Incident`, `Document`.
  - Relations:
    - `KNOWS` / `ASSOCIATED_WITH` (Person–Person)
    - `AFFILIATED_WITH` (Person–Organization)
    - `OCCURRED_AT` (Incident–Location)
    - `INVOLVES` (Incident–Person / Organization)
    - `MENTIONED_IN` (Entity / Incident–Document)
  - Each node/edge stores source references (document id, page, chunk id, text span) and confidence scores.

- **Extraction Pipeline**
  - Add Cognee transformers that:
    - Perform NER and entity normalization (canonical name + aliases).
    - Extract relations and incidents from one or more related chunks via LLM prompts.
    - Produce structured JSON (entities, relations, incidents) for ingestion into the graph.

- **Graph Storage and Access**
  - Store the graph in Cognee’s internal graph layer or an external graph DB (e.g. Neo4j) behind a clean abstraction.
  - Implement graph traversal APIs:
    - Given person X, list direct associates and incidents.
    - Given an incident id, list participants, date, and location.
  - Expose a `/graph/query` endpoint that returns graph answers + supporting evidence (linked chunks/documents).

Deliverable of Step 2: a graph-backed knowledge layer where relation and incident questions can be answered from structured data with precise references back to documents.

---

## Step 3 – Hybrid RAG (Graph + Vector)

**Goal**: Combine graph and vector retrieval so the system can:
- Answer **structured** relation/incident queries via the graph.
- Answer **unstructured** or narrative questions via vector search.
- Use both to provide the LLM with the most relevant and trustworthy evidence.

- **Query Understanding**
  - Use an LLM to classify the query:
    - `RELATION_QUERY`, `INCIDENT_QUERY`, `GENERAL_FACTUAL`, `SUMMARY`, etc.
  - Extract candidate entities, dates, and locations for use as filters.

- **Hybrid Retrieval**
  - For relation/incident queries:
    - Resolve entities to graph nodes and perform bounded-depth traversals.
    - Collect associated incidents, people, organizations, and locations.
  - In parallel, perform vector search over chunks, filtered by any known entities/dates.
  - Merge and rank evidence from both graph and vector sources.

- **Answer Synthesis**
  - Build prompts that:
    - Explicitly list graph-derived facts and vector-retrieved context.
    - Instruct the LLM to:
      - Prefer corroborated information.
      - Clearly label uncertain or conflicting information.
      - Always return citations (document + page/chunk + graph ids).
  - Extend the `/ask` endpoint to return:
    - Natural language answer.
    - Evidence set.
    - Any graph substructure used to derive the answer.

Deliverable of Step 3: a production-ready hybrid RAG that leverages both graph structure and document retrieval to answer complex questions with strong traceability and safety.

---

## Operational Concerns (All Steps)

- **Safety & Compliance**
  - Guardrails in prompts to avoid speculation beyond evidence.
  - PII and sensitive-content handling: redaction or masking in responses and logs as required.
  - Role-based access control on APIs and audit logs for all queries and answers.

- **Testing & Evaluation**
  - Maintain an evaluation set of questions and expected answers/relations.
  - Regularly run automated evaluations on retrieval quality and answer accuracy.

- **Deployment & Observability**
  - Containerize components and deploy via Docker / Kubernetes.
  - Centralized logging and metrics (latency, error rates, retrieval stats, LLM usage).
  - Feature flags or configuration for switching models, stores, and thresholds without redeploying code.

