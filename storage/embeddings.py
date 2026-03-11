import json
import logging
from typing import Iterable, Sequence

import psycopg
from langchain_core.documents import Document

from .postgres import PostgresConfig, _connect, _ensure_schema


logger = logging.getLogger(__name__)


def _ensure_embeddings_schema(conn: psycopg.Connection) -> None:
    """
    Create the chunk_embeddings table if it does not already exist.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chunk_embeddings (
                id BIGSERIAL PRIMARY KEY,
                document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                embedding JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_document_id
            ON chunk_embeddings (document_id);
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_doc_chunk
            ON chunk_embeddings (document_id, chunk_index);
            """
        )
    conn.commit()


def store_chunk_embeddings(
    chunks: Sequence[Document],
    embeddings: Sequence[Sequence[float]],
    cfg: PostgresConfig,
) -> None:
    """
    Persist chunk-level embeddings to Postgres.

    Strategy:
    - Resolve document_id for each chunk via its source_path metadata.
    - Upsert (document_id, chunk_index) -> embedding in chunk_embeddings.
    """
    if not chunks or not embeddings:
        logger.info("No chunks or embeddings provided; nothing to persist.")
        return

    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Number of chunks ({len(chunks)}) does not match number of embeddings ({len(embeddings)})"
        )

    logger.info(
        "Persisting embeddings for %d chunks to Postgres database '%s'",
        len(chunks),
        cfg.dbname,
    )

    with _connect(cfg) as conn:
        # Ensure base document / chunk tables exist
        _ensure_schema(conn)
        # Ensure embeddings table exists
        _ensure_embeddings_schema(conn)

        with conn.cursor() as cur:
            for chunk, vector in zip(chunks, embeddings):
                metadata = chunk.metadata or {}
                source_path = metadata.get("source_path")
                if not source_path:
                    logger.warning(
                        "Skipping embedding for chunk without source_path in metadata."
                    )
                    continue

                chunk_index = metadata.get("chunk_index")
                if chunk_index is None:
                    logger.warning(
                        "Skipping embedding for chunk without chunk_index in metadata (source_path=%s).",
                        source_path,
                    )
                    continue

                # Look up the associated document_id
                cur.execute(
                    """
                    SELECT id FROM documents
                    WHERE source_path = %s;
                    """,
                    (source_path,),
                )
                row = cur.fetchone()
                if row is None:
                    logger.warning(
                        "No document row found for source_path=%s; skipping embedding.",
                        source_path,
                    )
                    continue

                document_id = row[0]

                # Upsert embedding for this (document_id, chunk_index)
                cur.execute(
                    """
                    INSERT INTO chunk_embeddings (document_id, chunk_index, embedding, created_at)
                    VALUES (%s, %s, %s::jsonb, NOW())
                    ON CONFLICT (id) DO NOTHING;
                    """,
                    (
                        document_id,
                        int(chunk_index),
                        json.dumps(vector),
                    ),
                )

        conn.commit()

    logger.info("Chunk embeddings persistence complete.")

