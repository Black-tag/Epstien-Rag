"""
Storage package.

Currently provides Postgres-backed persistence for:
- Ingested documents
- Ingested chunks
"""

from .postgres import PostgresConfig, store_documents_and_chunks

__all__ = [
    "PostgresConfig",
    "store_documents_and_chunks",
]

