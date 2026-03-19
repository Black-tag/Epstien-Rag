"""
storage/postgres.py
-------------------
Postgres-backed persistence for documents, chunks, and pgvector embeddings.

Schema overview
---------------
documents       – one row per source file, upserted on source_path
chunks          – one row per text chunk; carries a nullable
                  vector(N) embedding column managed by pgvector

All vector similarity search is handled via a pgvector HNSW index on
chunks.embedding.  No external vector store (Chroma etc.) is required.

Prerequisites
-------------
* PostgreSQL with the pgvector extension installed:
      CREATE EXTENSION IF NOT EXISTS vector;
  _ensure_schema() runs this automatically, but the shared library must
  already be present on the server (pgvector >= 0.5.0 recommended).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Sequence, Tuple

import psycopg
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PostgresConfig:
    """
    Connection parameters for the Postgres backing store.

    Every field falls back to the standard PG* environment variables so the
    dataclass can be used without explicit arguments in a 12-factor app.
    """

    host: str = os.getenv("PGHOST", "localhost")
    port: int = int(os.getenv("PGPORT", "5432"))
    user: str = os.getenv("PGUSER", "postgres")
    password: str = os.getenv("PGPASSWORD", "postgres")
    dbname: str = os.getenv("PGDATABASE", "epstien_files_db")


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------


def _connect(cfg: PostgresConfig) -> psycopg.Connection:
    """Open and return a new psycopg3 connection."""
    return psycopg.connect(
        host=cfg.host,
        port=cfg.port,
        user=cfg.user,
        password=cfg.password,
        dbname=cfg.dbname,
    )


def _vec_literal(embedding: Sequence[float]) -> str:
    """
    Serialise a Python float sequence to a Postgres vector literal.

    Example: [0.1, -0.2, 0.3]  →  '[0.1,-0.2,0.3]'

    Using a cast (%s::vector) means we need no extra Python adapter and
    the pgvector extension handles parsing on the server side.
    """
    return "[" + ",".join(repr(float(v)) for v in embedding) + "]"


# ---------------------------------------------------------------------------
# Schema management
# ---------------------------------------------------------------------------


def _ensure_schema(conn: psycopg.Connection, embedding_dimensions: int = 1024) -> None:
    """
    Idempotent schema bootstrap – safe to call on every connection.

    Operations performed (all guarded with IF NOT EXISTS / IF NOT EXISTS):
      1. Enable the pgvector extension.
      2. Create the ``documents`` table.
      3. Create the ``chunks`` table with a nullable vector column.
      4. ALTER TABLE to add the embedding column when upgrading an existing
         schema that was created before pgvector was introduced.
      5. Create a B-tree index on chunks.document_id.
      6. Create an HNSW cosine-similarity index on chunks.embedding
         (partial index: only rows where embedding IS NOT NULL).
    """
    with conn.cursor() as cur:
        # 1. pgvector extension -------------------------------------------------
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # 2. documents ----------------------------------------------------------
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id            SERIAL       PRIMARY KEY,
                source_path   TEXT         UNIQUE NOT NULL,
                file_name     TEXT         NOT NULL,
                extension     TEXT         NOT NULL,
                last_modified TIMESTAMPTZ,
                content_hash  CHAR(64)     NOT NULL,
                raw_text      TEXT,
                created_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
                updated_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW()
            );
            """
        )

        # 3. chunks (fresh install includes the vector column) ------------------
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id           BIGSERIAL    PRIMARY KEY,
                document_id  INTEGER      NOT NULL
                                 REFERENCES documents(id) ON DELETE CASCADE,
                chunk_index  INTEGER      NOT NULL,
                content      TEXT         NOT NULL,
                metadata     JSONB        NOT NULL,
                embedding    vector({embedding_dimensions}),
                created_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW()
            );
            """
        )

        # 4. Upgrade path: add embedding column to pre-pgvector schema ----------
        cur.execute(
            f"""
            ALTER TABLE chunks
                ADD COLUMN IF NOT EXISTS embedding vector({embedding_dimensions});
            """
        )

        # 5. B-tree index for document → chunk joins ----------------------------
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chunks_document_id
                ON chunks (document_id);
            """
        )

        # 6. HNSW index for ANN cosine search ----------------------------------
        # Partial so that the index stays lean while embedding is back-filled.
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
                ON chunks USING hnsw (embedding vector_cosine_ops)
                WHERE embedding IS NOT NULL;
            """
        )

    conn.commit()


# ---------------------------------------------------------------------------
# Write path
# ---------------------------------------------------------------------------


def store_documents_and_chunks(
    documents: Sequence[Document],
    chunks: Sequence[Document],
    cfg: PostgresConfig,
    embedding_dimensions: int = 1024,
) -> List[Tuple[int, str]]:
    """
    Persist normalised documents and their text chunks to Postgres.

    Returns
    -------
    list[tuple[int, str]]
        One entry per successfully inserted chunk row, in insertion order:
        ``(chunk_db_id, chunk_content)``.

        This is handed directly to :func:`update_chunk_embeddings` so the
        pipeline never needs a second round-trip to reload chunk data.

    Strategy
    --------
    * Upsert one row per ``source_path`` in *documents*.
    * For each touched document delete its existing chunks then re-insert –
      keeps the table clean after re-ingestion without leaving stale rows.
    * Embeddings are NOT written here; the column starts NULL and is filled
      by a subsequent call to :func:`update_chunk_embeddings`.
    """
    if not documents and not chunks:
        logger.info("No documents or chunks to persist – skipping Postgres write.")
        return []

    logger.info(
        "Persisting %d document(s) and %d chunk(s) to Postgres database '%s'.",
        len(documents),
        len(chunks),
        cfg.dbname,
    )

    inserted: List[Tuple[int, str]] = []

    with _connect(cfg) as conn:
        _ensure_schema(conn, embedding_dimensions=embedding_dimensions)

        # source_path → documents.id
        path_to_doc_id: dict[str, int] = {}

        with conn.cursor() as cur:
            # ------------------------------------------------------------------
            # Upsert documents
            # ------------------------------------------------------------------
            for doc in documents:
                meta = doc.metadata or {}
                source_path: Optional[str] = meta.get("source_path")
                if not source_path:
                    logger.warning(
                        "Skipping document without source_path in metadata."
                    )
                    continue

                file_name: str = meta.get("file_name") or source_path.rsplit("/", 1)[-1]
                extension: str = meta.get("extension") or ""

                last_modified_raw = meta.get("last_modified")
                last_modified: Optional[datetime] = (
                    datetime.fromisoformat(last_modified_raw)
                    if last_modified_raw
                    else None
                )

                content_hash: str = meta.get("content_hash") or hashlib.sha256(
                    doc.page_content.encode("utf-8")
                ).hexdigest()

                cur.execute(
                    """
                    INSERT INTO documents
                        (source_path, file_name, extension, last_modified,
                         content_hash, raw_text, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, NOW(), NOW())
                    ON CONFLICT (source_path) DO UPDATE SET
                        file_name     = EXCLUDED.file_name,
                        extension     = EXCLUDED.extension,
                        last_modified = EXCLUDED.last_modified,
                        content_hash  = EXCLUDED.content_hash,
                        raw_text      = EXCLUDED.raw_text,
                        updated_at    = NOW()
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
                row = cur.fetchone()
                if row:
                    path_to_doc_id[source_path] = row[0]

            # ------------------------------------------------------------------
            # Delete stale chunks for the documents we just upserted
            # ------------------------------------------------------------------
            if path_to_doc_id:
                cur.execute(
                    "DELETE FROM chunks WHERE document_id = ANY(%s);",
                    (list(path_to_doc_id.values()),),
                )

            # ------------------------------------------------------------------
            # Insert chunks (embedding stays NULL until update_chunk_embeddings)
            # ------------------------------------------------------------------
            for chunk in chunks:
                meta = chunk.metadata or {}
                source_path = meta.get("source_path")
                if not source_path:
                    logger.warning("Skipping chunk without source_path in metadata.")
                    continue

                doc_id = path_to_doc_id.get(source_path)
                if doc_id is None:
                    logger.warning(
                        "No document_id for chunk source_path=%s; skipping.",
                        source_path,
                    )
                    continue

                chunk_index: int = int(meta.get("chunk_index", 0))

                cur.execute(
                    """
                    INSERT INTO chunks
                        (document_id, chunk_index, content, metadata, created_at)
                    VALUES (%s, %s, %s, %s::jsonb, NOW())
                    RETURNING id;
                    """,
                    (
                        doc_id,
                        chunk_index,
                        chunk.page_content,
                        json.dumps(meta, ensure_ascii=False),
                    ),
                )
                new_id_row = cur.fetchone()
                if new_id_row:
                    inserted.append((new_id_row[0], chunk.page_content))

        conn.commit()

    logger.info(
        "Postgres write complete: %d document(s), %d chunk(s) inserted.",
        len(path_to_doc_id),
        len(inserted),
    )
    return inserted


def update_chunk_embeddings(
    chunk_db_ids: List[int],
    embeddings: List[List[float]],
    cfg: PostgresConfig,
) -> None:
    """
    Write pre-computed embedding vectors back into the ``chunks`` table.

    Each ``chunk_db_ids[i]`` is updated with ``embeddings[i]`` via a
    parameterised ``UPDATE … SET embedding = %s::vector``.  The cast lets
    psycopg pass a plain string and have Postgres parse it as a vector,
    so no extra Python adapter (numpy / pgvector) is required.

    Parameters
    ----------
    chunk_db_ids:
        Primary-key values of the chunk rows to update, in the same order
        as *embeddings*.
    embeddings:
        One vector per chunk – must be the same length as *chunk_db_ids*.
    cfg:
        Postgres connection configuration.

    Raises
    ------
    ValueError
        If the lengths of *chunk_db_ids* and *embeddings* do not match.
    """
    if not chunk_db_ids:
        logger.info("update_chunk_embeddings: no chunks to update.")
        return

    if len(chunk_db_ids) != len(embeddings):
        raise ValueError(
            f"chunk_db_ids length ({len(chunk_db_ids)}) != "
            f"embeddings length ({len(embeddings)})"
        )

    logger.info(
        "Writing %d embedding vector(s) to Postgres chunks table.",
        len(chunk_db_ids),
    )

    with _connect(cfg) as conn:
        with conn.cursor() as cur:
            for chunk_id, embedding in zip(chunk_db_ids, embeddings):
                cur.execute(
                    "UPDATE chunks SET embedding = %s::vector WHERE id = %s;",
                    (_vec_literal(embedding), chunk_id),
                )
        conn.commit()

    logger.info("Embedding update complete for %d chunk(s).", len(chunk_db_ids))


# ---------------------------------------------------------------------------
# Read path
# ---------------------------------------------------------------------------


def load_all_chunks(
    cfg: PostgresConfig,
    limit: Optional[int] = None,
) -> List[Document]:
    """
    Load chunk rows from Postgres as LangChain ``Document`` objects.

    Used by ad-hoc / diagnostic tooling.  The full pipeline uses the
    ``(chunk_db_id, content)`` pairs returned by
    :func:`store_documents_and_chunks` directly, so this function is not
    called in the hot path.

    Parameters
    ----------
    cfg:
        Postgres connection configuration.
    limit:
        Optional cap on the number of rows returned (ordered by
        document_id, chunk_index).
    """
    logger.info(
        "Loading chunks from Postgres '%s'%s.",
        cfg.dbname,
        f" (limit={limit})" if limit is not None else "",
    )

    with _connect(cfg) as conn:
        with conn.cursor() as cur:
            if limit is not None:
                cur.execute(
                    """
                    SELECT content, metadata
                    FROM   chunks
                    ORDER  BY document_id, chunk_index
                    LIMIT  %s;
                    """,
                    (int(limit),),
                )
            else:
                cur.execute(
                    """
                    SELECT content, metadata
                    FROM   chunks
                    ORDER  BY document_id, chunk_index;
                    """
                )
            rows = cur.fetchall()

    docs: List[Document] = [
        Document(page_content=content or "", metadata=meta or {})
        for content, meta in rows
    ]
    logger.info("Loaded %d chunk(s) from Postgres.", len(docs))
    return docs


def load_file_hashes(cfg: PostgresConfig) -> dict[str, str]:
    """
    Return a mapping of ``source_path → content_hash`` for every row
    currently in the ``documents`` table.

    This is used by the ingestion pipeline as a **Postgres-backed
    deduplication state store**, replacing the fragile JSON state file.

    On the first call against an empty database the function returns an
    empty dict, which causes every file to be treated as new.  Subsequent
    calls reflect whatever was last upserted by
    :func:`store_documents_and_chunks`.
    """
    logger.info(
        "Loading known file hashes from Postgres database '%s'.", cfg.dbname
    )
    try:
        with _connect(cfg) as conn:
            # Ensure the schema exists so this can be called safely before
            # the first ingest run (e.g. during startup health checks).
            _ensure_schema(conn)
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT source_path, content_hash FROM documents;"
                )
                rows = cur.fetchall()
    except Exception as exc:
        # A DB connection failure at this stage should not crash the server;
        # return an empty dict so the pipeline treats everything as new and
        # logs the error for the operator to investigate.
        logger.error(
            "Could not load file hashes from Postgres ('%s'): %s – "
            "treating all files as new.",
            cfg.dbname,
            exc,
        )
        return {}

    hashes = {row[0]: row[1] for row in rows if row[0] and row[1]}
    logger.info(
        "Loaded %d known file hash(es) from Postgres.", len(hashes)
    )
    return hashes


def load_chunks_without_embeddings(
    cfg: PostgresConfig,
    limit: Optional[int] = None,
) -> List[Tuple[int, str]]:
    """
    Return ``(chunk_db_id, content)`` for every chunk whose embedding is
    still NULL.

    Useful for back-filling embeddings after a schema migration or after
    an interrupted embedding run.

    Parameters
    ----------
    cfg:
        Postgres connection configuration.
    limit:
        Optional cap on the number of rows returned.
    """
    logger.info(
        "Loading un-embedded chunks from Postgres '%s'%s.",
        cfg.dbname,
        f" (limit={limit})" if limit is not None else "",
    )

    with _connect(cfg) as conn:
        with conn.cursor() as cur:
            if limit is not None:
                cur.execute(
                    """
                    SELECT id, content
                    FROM   chunks
                    WHERE  embedding IS NULL
                    ORDER  BY document_id, chunk_index
                    LIMIT  %s;
                    """,
                    (int(limit),),
                )
            else:
                cur.execute(
                    """
                    SELECT id, content
                    FROM   chunks
                    WHERE  embedding IS NULL
                    ORDER  BY document_id, chunk_index;
                    """
                )
            rows = cur.fetchall()

    result: List[Tuple[int, str]] = [(row[0], row[1] or "") for row in rows]
    logger.info(
        "Found %d chunk(s) without embeddings in Postgres.", len(result)
    )
    return result
