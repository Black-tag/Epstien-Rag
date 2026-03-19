"""
Storage package.

Provides Postgres-backed persistence for:
- Ingested documents
- Ingested chunks
- pgvector embeddings (stored directly on the chunks table)
"""

from .postgres import (
    PostgresConfig,
    store_documents_and_chunks,
    update_chunk_embeddings,
    load_all_chunks,
    load_chunks_without_embeddings,
)

__all__ = [
    "PostgresConfig",
    "store_documents_and_chunks",
    "update_chunk_embeddings",
    "load_all_chunks",
    "load_chunks_without_embeddings",
]
