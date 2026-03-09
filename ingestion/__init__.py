"""
Ingestion package

Provides reusable, LangChain-based ingestion utilities for:
- Multi-format document loading
- Metadata extraction
- Text normalization & deduplication
- Chunking with fixed length + overlap
- Traceability back to original files
"""

from .pipeline import (
    IngestionConfig,
    IngestionState,
    IngestionResult,
    ingest_repository,
)

__all__ = [
    "IngestionConfig",
    "IngestionState",
    "IngestionResult",
    "ingest_repository",
]

