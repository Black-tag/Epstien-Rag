import argparse
import logging
import os
from pathlib import Path
from typing import List, Tuple

from langchain_core.documents import Document

from config.loader import load_properties
from ingestion import IngestionConfig, IngestionState, ingest_repository
from ingestion.embeddings import embed_chunks_with_ollama
from storage import (
    PostgresConfig,
    store_documents_and_chunks,
    update_chunk_embeddings,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _run_ingest(args: argparse.Namespace) -> None:
    """
    Run the full ingestion pipeline:

        1. ETL  – scan the document repository, parse, chunk, deduplicate
        2. SQL  – upsert documents + chunks to Postgres (embedding = NULL)
        3. Embed – generate vectors via Ollama for every inserted chunk
        4. pgvector – UPDATE chunks SET embedding = %s::vector

    Configuration priority for every value:
        1. CLI argument
        2. config/properties.yaml
        3. Environment variable  (repository path only)
        4. Hard-coded default
    """
    project_root = Path(__file__).resolve().parent

    properties: dict = load_properties()
    ingestion_cfg: dict = properties.get("ingestion", {}) or {}
    db_cfg: dict = properties.get("database", {}) or {}
    ollama_cfg: dict = properties.get("ollama", {}) or {}
    embedding_cfg: dict = properties.get("embedding", {}) or {}

    embedding_dimensions: int = int(embedding_cfg.get("dimensions", 1024))

    # ------------------------------------------------------------------
    # Resolve repository path
    # ------------------------------------------------------------------
    env_repo = os.getenv("EPSTEIN_DOCS_PATH")

    if args.repository:
        repository_path = Path(args.repository).expanduser().resolve()
    elif env_repo:
        repository_path = Path(env_repo).expanduser().resolve()
    elif ingestion_cfg.get("repository_path"):
        repository_path = (
            Path(ingestion_cfg["repository_path"]).expanduser().resolve()
        )
    else:
        repository_path = project_root / "epstein-documents"

    if not repository_path.exists() and not args.repository and not env_repo:
        logger.info(
            "Default repository path %s does not exist – creating it.",
            repository_path,
        )
        repository_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Resolve state file
    # ------------------------------------------------------------------
    if args.state_file:
        state_file = Path(args.state_file).resolve()
    elif ingestion_cfg.get("state_file"):
        state_file = (project_root / ingestion_cfg["state_file"]).resolve()
    else:
        state_file = project_root / "data" / "ingestion_state.json"

    # ------------------------------------------------------------------
    # Chunking parameters
    # ------------------------------------------------------------------
    chunk_size: int = args.chunk_size or int(ingestion_cfg.get("chunk_size", 800))
    chunk_overlap: int = args.chunk_overlap or int(
        ingestion_cfg.get("chunk_overlap", 200)
    )

    # ------------------------------------------------------------------
    # Step 1 – ETL
    # ------------------------------------------------------------------
    config = IngestionConfig(
        repository_path=repository_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        state_file=state_file,
    )

    state = IngestionState.load(config.state_file)

    logger.info("Ingesting repository at %s", repository_path)
    logger.info("State file: %s", state_file)

    output = ingest_repository(config=config, state=state)

    documents: list[Document] = output.get("documents", [])
    chunks: list[Document] = output.get("chunks", [])

    logger.info("ETL complete – %d document(s), %d chunk(s).", len(documents), len(chunks))

    if not documents and not chunks:
        logger.info("Nothing to process – exiting.")
        return

    # ------------------------------------------------------------------
    # Step 2 – Persist documents + chunks to Postgres
    #          store_documents_and_chunks returns (chunk_db_id, content)
    #          pairs in insertion order so we never need a reload step.
    # ------------------------------------------------------------------
    pg_cfg = PostgresConfig(
        host=args.db_host or db_cfg.get("host", "localhost"),
        port=args.db_port or int(db_cfg.get("port", 5432)),
        user=args.db_user or db_cfg.get("user", "postgres"),
        password=args.db_password or db_cfg.get("password", "postgres"),
        dbname=args.db_name or db_cfg.get("name", "epstien_files_db"),
    )

    chunk_records: List[Tuple[int, str]] = store_documents_and_chunks(
        documents=documents,
        chunks=chunks,
        cfg=pg_cfg,
        embedding_dimensions=embedding_dimensions,
    )

    logger.info(
        "Postgres write complete – %d document(s), %d chunk(s) inserted.",
        len(documents),
        len(chunk_records),
    )

    if not chunk_records:
        logger.info("No chunk rows were inserted – skipping embedding step.")
        return

    # ------------------------------------------------------------------
    # Step 3 – Generate embeddings via Ollama
    # ------------------------------------------------------------------
    ollama_model: str = ollama_cfg.get("embedding_model", "snowflake-arctic-embed")
    ollama_endpoint: str = ollama_cfg.get(
        "endpoint", "http://localhost:11434/api/embed"
    )
    batch_size: int = int(embedding_cfg.get("batch_size", 64))

    chunk_db_ids: List[int] = [db_id for db_id, _ in chunk_records]
    embed_docs: List[Document] = [
        Document(page_content=content) for _, content in chunk_records
    ]

    logger.info(
        "Requesting embeddings from Ollama model '%s' "
        "(chunks=%d, batch_size=%d).",
        ollama_model,
        len(embed_docs),
        batch_size,
    )

    try:
        embeddings = embed_chunks_with_ollama(
            embed_docs,
            model=ollama_model,
            endpoint=ollama_endpoint,
            batch_size=batch_size,
        )
    except Exception:
        logger.exception(
            "Ollama embedding failed. "
            "Documents and chunks are persisted in Postgres "
            "but embeddings were NOT written. "
            "Re-run ingestion to retry the embedding step."
        )
        raise

    logger.info(
        "Received %d embedding vector(s) from Ollama.", len(embeddings)
    )

    # ------------------------------------------------------------------
    # Step 4 – Write embeddings to Postgres via pgvector
    #          UPDATE chunks SET embedding = %s::vector WHERE id = %s
    # ------------------------------------------------------------------
    update_chunk_embeddings(
        chunk_db_ids=chunk_db_ids,
        embeddings=embeddings,
        cfg=pg_cfg,
    )

    logger.info(
        "Pipeline complete – %d doc(s) | %d chunk(s) | %d embedding(s) written to Postgres.",
        len(documents),
        len(chunk_records),
        len(embeddings),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Epstein RAG – full ingestion pipeline CLI.\n\n"
            "Runs ETL → Postgres → Ollama embeddings → pgvector in one shot."
        ),
    )

    # ------------------------------------------------------------------
    # ingest subcommand  (the only command; embed is retired)
    # ------------------------------------------------------------------
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser(
        "ingest",
        help=(
            "Run the full pipeline: parse documents, chunk, persist to "
            "Postgres, generate Ollama embeddings, store via pgvector."
        ),
    )
    ingest_parser.add_argument(
        "-r",
        "--repository",
        type=str,
        default=None,
        help="Path to the documents repository (default: ./epstein-documents).",
    )
    ingest_parser.add_argument(
        "--state-file",
        type=str,
        default=None,
        help="Path to ingestion state JSON file (default: ./data/ingestion_state.json).",
    )
    ingest_parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Chunk size in characters (default from properties.yaml, fallback 800).",
    )
    ingest_parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Chunk overlap in characters (default from properties.yaml, fallback 200).",
    )
    ingest_parser.add_argument(
        "--db-host",
        type=str,
        default=None,
        help="Postgres host (default: localhost).",
    )
    ingest_parser.add_argument(
        "--db-port",
        type=int,
        default=None,
        help="Postgres port (default: 5432).",
    )
    ingest_parser.add_argument(
        "--db-user",
        type=str,
        default=None,
        help="Postgres user (default: postgres).",
    )
    ingest_parser.add_argument(
        "--db-password",
        type=str,
        default=None,
        help="Postgres password (default: postgres).",
    )
    ingest_parser.add_argument(
        "--db-name",
        type=str,
        default=None,
        help="Postgres database name (default: epstien_files_db).",
    )
    ingest_parser.set_defaults(func=_run_ingest)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
