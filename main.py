import argparse
import logging
import os
from pathlib import Path

from config.loader import load_properties
from ingestion import IngestionConfig, IngestionState, ingest_repository
from ingestion.embeddings import embed_chunks_with_ollama
from storage import PostgresConfig, store_documents_and_chunks
from storage.chroma_store import upsert_chunk_embeddings
from storage.postgres import load_all_chunks


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _run_ingest(args: argparse.Namespace) -> None:
    """
    Run ingestion over the Epstein documents repository.

    By default, it looks for documents under:
        ./epstein-documents
    and stores ingestion state under:
        ./data/ingestion_state.json
    """
    project_root = Path(__file__).resolve().parent

    # Load properties.yaml for central configuration (reusable helper)
    properties: dict = load_properties()

    ingestion_cfg = properties.get("ingestion", {}) or {}
    db_cfg = properties.get("database", {}) or {}

    # Resolve repository path with precedence:
    # 1. CLI argument
    # 2. EPSTEIN_DOCS_PATH environment variable
    # 3. ingestion.repository_path from properties.yaml
    # 4. Default: ./epstein-documents under project root
    env_repo = os.getenv("EPSTEIN_DOCS_PATH")
    if args.repository:
        repository_path = Path(args.repository).expanduser().resolve()
    elif env_repo:
        repository_path = Path(env_repo).expanduser().resolve()
    elif ingestion_cfg.get("repository_path"):
        repository_path = Path(ingestion_cfg["repository_path"]).expanduser().resolve()
    else:
        repository_path = project_root / "epstein-documents"

    # For the default path (no CLI arg and no env), create the directory if missing
    if not repository_path.exists() and not args.repository and not env_repo:
        logger.info(
            "Default repository path %s does not exist. Creating it now.",
            repository_path,
        )
        repository_path.mkdir(parents=True, exist_ok=True)

    # State file: CLI arg > properties.yaml > default path
    if args.state_file:
        state_file = Path(args.state_file).resolve()
    elif ingestion_cfg.get("state_file"):
        state_file = (project_root / ingestion_cfg["state_file"]).resolve()
    else:
        state_file = project_root / "data" / "ingestion_state.json"

    # Chunking config: CLI args override properties.yaml, which override hard defaults
    chunk_size = args.chunk_size or int(ingestion_cfg.get("chunk_size", 800))
    chunk_overlap = args.chunk_overlap or int(ingestion_cfg.get("chunk_overlap", 200))

    config = IngestionConfig(
        repository_path=repository_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        state_file=state_file,
    )

    state = IngestionState.load(config.state_file)
    logger.info("Ingesting repository at %s", repository_path)
    logger.info("Using state file: %s", state_file)

    output = ingest_repository(config=config, state=state)

    documents = output.get("documents", [])
    chunks = output.get("chunks", [])

    logger.info("Ingestion complete.")
    logger.info("Documents ready for indexing: %d", len(documents))
    logger.info("Chunks ready for embedding: %d", len(chunks))

    # Persist normalized documents and chunks to Postgres (ingestion only)
    pg_cfg = PostgresConfig(
        host=args.db_host or db_cfg.get("host", "localhost"),
        port=args.db_port or int(db_cfg.get("port", 5432)),
        user=args.db_user or db_cfg.get("user", "postgres"),
        password=args.db_password or db_cfg.get("password", "postgres"),
        dbname=args.db_name or db_cfg.get("name", "epstien_files_db"),
    )
    store_documents_and_chunks(documents=documents, chunks=chunks, cfg=pg_cfg)


def _run_embed_only(args: argparse.Namespace) -> None:
    """
    Run embeddings + Chroma indexing using chunks that have already
    been ingested and stored in Postgres.

    This lets you:
    - Ingest once (batch or streaming) and embed later.
    - Re-embed all chunks when you change embedding model or settings.
    """
    project_root = Path(__file__).resolve().parent

    properties: dict = load_properties()
    db_cfg = properties.get("database", {}) or {}
    ollama_cfg = properties.get("ollama", {}) or {}
    chroma_cfg = properties.get("chroma", {}) or {}
    embedding_cfg = properties.get("embedding", {}) or {}

    # Build Postgres config (CLI overrides properties.yaml)
    pg_cfg = PostgresConfig(
        host=args.db_host or db_cfg.get("host", "localhost"),
        port=args.db_port or int(db_cfg.get("port", 5432)),
        user=args.db_user or db_cfg.get("user", "postgres"),
        password=args.db_password or db_cfg.get("password", "postgres"),
        dbname=args.db_name or db_cfg.get("name", "epstien_files_db"),
    )

    # Load existing chunks from Postgres
    max_chunks = args.max_chunks
    chunks = load_all_chunks(cfg=pg_cfg, limit=max_chunks)
    if not chunks:
        logger.info("No chunks found in Postgres; nothing to embed.")
        return

    ollama_model = ollama_cfg.get("embedding_model", "snowflake-arctic-embed")
    ollama_endpoint = ollama_cfg.get("endpoint", "http://localhost:11434/api/embed")
    batch_size = int(embedding_cfg.get("batch_size", 64))

    try:
        embeddings = embed_chunks_with_ollama(
            chunks,
            model=ollama_model,
            endpoint=ollama_endpoint,
            batch_size=batch_size,
        )
    except Exception:
        logger.exception(
            "Failed to generate embeddings for chunks; skipping embedding persistence."
        )
        return

    chroma_dir = chroma_cfg.get("persist_directory", "./data/chroma")
    collection_name = chroma_cfg.get("collection_name", "epstein_chunks")

    upsert_chunk_embeddings(
        chunks=chunks,
        embeddings=embeddings,
        persist_directory=project_root / chroma_dir,
        collection_name=collection_name,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Epstein RAG – ingestion and orchestration CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Ingestion command (documents + chunks only)
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Run ingestion over the Epstein documents repository (no embeddings)",
    )
    ingest_parser.add_argument(
        "-r",
        "--repository",
        type=str,
        default=None,
        help="Path to the documents repository (default: ./epstein-documents)",
    )
    ingest_parser.add_argument(
        "--state-file",
        type=str,
        default=None,
        help="Path to ingestion state JSON file (default: ./data/ingestion_state.json)",
    )
    ingest_parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Chunk size (characters) for splitting documents",
    )
    ingest_parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Chunk overlap (characters) for splitting documents",
    )
    ingest_parser.add_argument(
        "--db-host",
        type=str,
        default=None,
        help="Postgres host (default: localhost)",
    )
    ingest_parser.add_argument(
        "--db-port",
        type=int,
        default=None,
        help="Postgres port (default: 5432)",
    )
    ingest_parser.add_argument(
        "--db-user",
        type=str,
        default=None,
        help="Postgres user (default: postgres)",
    )
    ingest_parser.add_argument(
        "--db-password",
        type=str,
        default=None,
        help="Postgres password (default: postgres)",
    )
    ingest_parser.add_argument(
        "--db-name",
        type=str,
        default=None,
        help="Postgres database name (default: epstien-files_db)",
    )
    ingest_parser.set_defaults(func=_run_ingest)

    # Embedding-only command (for already-ingested chunks)
    embed_parser = subparsers.add_parser(
        "embed",
        help="Generate embeddings and index into Chroma using existing chunks in Postgres",
    )
    embed_parser.add_argument(
        "--db-host",
        type=str,
        default=None,
        help="Postgres host (default: localhost)",
    )
    embed_parser.add_argument(
        "--db-port",
        type=int,
        default=None,
        help="Postgres port (default: 5432)",
    )
    embed_parser.add_argument(
        "--db-user",
        type=str,
        default=None,
        help="Postgres user (default: postgres)",
    )
    embed_parser.add_argument(
        "--db-password",
        type=str,
        default=None,
        help="Postgres password (default: postgres)",
    )
    embed_parser.add_argument(
        "--db-name",
        type=str,
        default=None,
        help="Postgres database name (default: epstien-files_db)",
    )
    embed_parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Maximum number of chunks to embed in this run (for streaming/batch control).",
    )
    embed_parser.set_defaults(func=_run_embed_only)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()