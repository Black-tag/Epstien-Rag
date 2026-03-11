import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Sequence

import psycopg
from langchain_core.documents import Document


logger = logging.getLogger(__name__)


@dataclass
class PostgresConfig:
    """
    Connection configuration for the Postgres backing store.

    Defaults match a typical local development instance:
    - host: localhost
    - port: 5432
    - user: postgres
    - password: postgres
    - dbname: epstien_files_db
    """

    host: str = os.getenv("PGHOST", "localhost")
    port: int = int(os.getenv("PGPORT", "5432"))
    user: str = os.getenv("PGUSER", "postgres")
    password: str = os.getenv("PGPASSWORD", "postgres")
    dbname: str = os.getenv("PGDATABASE", "epstien_files_db")


def _connect(cfg: PostgresConfig) -> psycopg.Connection:
    return psycopg.connect(
        host=cfg.host,
        port=cfg.port,
        user=cfg.user,
        password=cfg.password,
        dbname=cfg.dbname,
    )


def _ensure_schema(conn: psycopg.Connection) -> None:
    """
    Create the documents and chunks tables if they do not already exist.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                source_path TEXT UNIQUE NOT NULL,
                file_name TEXT NOT NULL,
                extension TEXT NOT NULL,
                last_modified TIMESTAMPTZ,
                content_hash CHAR(64) NOT NULL,
                raw_text TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id BIGSERIAL PRIMARY KEY,
                document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                metadata JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chunks_document_id
            ON chunks (document_id);
            """
        )
    conn.commit()


def store_documents_and_chunks(
    documents: Sequence[Document],
    chunks: Sequence[Document],
    cfg: PostgresConfig,
) -> None:
    """
    Persist normalized documents and their chunks to Postgres.

    Strategy:
    - Upsert one row per source_path in documents.
    - For each (possibly new/updated) document, delete existing chunks and insert fresh ones.
    """
    if not documents and not chunks:
        logger.info("No documents or chunks to persist to Postgres.")
        return

    logger.info(
        "Persisting %d documents and %d chunks to Postgres database '%s'",
        len(documents),
        len(chunks),
        cfg.dbname,
    )

    with _connect(cfg) as conn:
        _ensure_schema(conn)

        # Map source_path -> document_id
        path_to_id: dict[str, int] = {}

        with conn.cursor() as cur:
            # Upsert documents
            for doc in documents:
                metadata = doc.metadata or {}
                source_path = metadata.get("source_path")
                if not source_path:
                    logger.warning("Skipping document without source_path in metadata.")
                    continue

                file_name = metadata.get("file_name") or source_path.rsplit("/", 1)[-1]
                extension = metadata.get("extension") or ""
                last_modified_str = metadata.get("last_modified")
                last_modified = (
                    datetime.fromisoformat(last_modified_str)
                    if last_modified_str
                    else None
                )
                content_hash = metadata.get("content_hash")
                if not content_hash:
                    # Fallback: hash of content for robustness; ingestion already computes hashes
                    import hashlib

                    content_hash = hashlib.sha256(
                        doc.page_content.encode("utf-8")
                    ).hexdigest()

                cur.execute(
                    """
                    INSERT INTO documents (source_path, file_name, extension, last_modified, content_hash, raw_text, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, NOW(), NOW())
                    ON CONFLICT (source_path)
                    DO UPDATE SET
                        file_name = EXCLUDED.file_name,
                        extension = EXCLUDED.extension,
                        last_modified = EXCLUDED.last_modified,
                        content_hash = EXCLUDED.content_hash,
                        raw_text = EXCLUDED.raw_text,
                        updated_at = NOW()
                    RETURNING id;
                    """,
                    (
                        source_path,
                        file_name,
                        extension,
                        last_modified,
                        content_hash,
                        doc.page_content,
                    ),
                )
                document_id = cur.fetchone()[0]
                path_to_id[source_path] = document_id

            # Clear existing chunks for these documents to avoid stale data
            if path_to_id:
                cur.execute(
                    """
                    DELETE FROM chunks
                    WHERE document_id = ANY(%s);
                    """,
                    (list(path_to_id.values()),),
                )

            # Insert chunks
            for chunk in chunks:
                metadata = chunk.metadata or {}
                source_path = metadata.get("source_path")
                if not source_path:
                    logger.warning("Skipping chunk without source_path in metadata.")
                    continue

                document_id = path_to_id.get(source_path)
                if document_id is None:
                    logger.warning(
                        "No document_id found for chunk source_path=%s; skipping.", source_path
                    )
                    continue

                chunk_index = metadata.get("chunk_index", 0)

                cur.execute(
                    """
                    INSERT INTO chunks (document_id, chunk_index, content, metadata, created_at)
                    VALUES (%s, %s, %s, %s::jsonb, NOW());
                    """,
                    (
                        document_id,
                        int(chunk_index),
                        chunk.page_content,
                        json.dumps(metadata, ensure_ascii=False),
                    ),
                )

        conn.commit()

    logger.info("Postgres persistence complete.")


def load_all_chunks(
    cfg: PostgresConfig,
    limit: int | None = None,
) -> list[Document]:
    """
    Load all chunk rows from Postgres as LangChain Document objects.

    This is used by the embedding-only pipeline to generate embeddings
    for chunks that have already been ingested.
    """
    logger.info(
        "Loading chunks from Postgres database '%s'%s",
        cfg.dbname,
        f" (limit={limit})" if limit is not None else "",
    )

    with _connect(cfg) as conn:
        with conn.cursor() as cur:
            if limit is not None:
                cur.execute(
                    """
                    SELECT content, metadata
                    FROM chunks
                    ORDER BY document_id, chunk_index
                    LIMIT %s;
                    """,
                    (int(limit),),
                )
            else:
                cur.execute(
                    """
                    SELECT content, metadata
                    FROM chunks
                    ORDER BY document_id, chunk_index;
                    """
                )

            rows = cur.fetchall()

    documents: list[Document] = []
    for content, metadata in rows:
        # metadata is JSONB; psycopg returns it as a Python dict
        doc = Document(page_content=content or "", metadata=metadata or {})
        documents.append(doc)

    logger.info("Loaded %d chunks from Postgres for embedding.", len(documents))
    return documents

