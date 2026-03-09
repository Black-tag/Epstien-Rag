import argparse
import logging
import os
from pathlib import Path

from ingestion import IngestionConfig, IngestionState, ingest_repository
from storage import PostgresConfig, store_documents_and_chunks


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

    # Resolve repository path with precedence:
    # 1. CLI argument
    # 2. EPSTEIN_DOCS_PATH environment variable
    # 3. Default: ./epstein-documents under project root
    env_repo = os.getenv("EPSTEIN_DOCS_PATH")
    if args.repository:
        repository_path = Path(args.repository).expanduser().resolve()
    elif env_repo:
        repository_path = Path(env_repo).expanduser().resolve()
    else:
        repository_path = project_root / "epstein-documents"

    # For the default path (no CLI arg and no env), create the directory if missing
    if not repository_path.exists() and not args.repository and not env_repo:
        logger.info(
            "Default repository path %s does not exist. Creating it now.",
            repository_path,
        )
        repository_path.mkdir(parents=True, exist_ok=True)

    state_file = (
        Path(args.state_file).resolve()
        if args.state_file
        else project_root / "data" / "ingestion_state.json"
    )

    config = IngestionConfig(
        repository_path=repository_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
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

    # Persist to Postgres
    pg_cfg = PostgresConfig(
        host=args.db_host,
        port=args.db_port,
        user=args.db_user,
        password=args.db_password,
        dbname=args.db_name,
    )
    store_documents_and_chunks(documents=documents, chunks=chunks, cfg=pg_cfg)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Epstein RAG – ingestion and orchestration CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Ingestion command
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Run ingestion over the Epstein documents repository",
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
        default=800,
        help="Chunk size (characters) for splitting documents",
    )
    ingest_parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Chunk overlap (characters) for splitting documents",
    )
    ingest_parser.add_argument(
        "--db-host",
        type=str,
        default="localhost",
        help="Postgres host (default: localhost)",
    )
    ingest_parser.add_argument(
        "--db-port",
        type=int,
        default=5432,
        help="Postgres port (default: 5432)",
    )
    ingest_parser.add_argument(
        "--db-user",
        type=str,
        default="postgres",
        help="Postgres user (default: postgres)",
    )
    ingest_parser.add_argument(
        "--db-password",
        type=str,
        default="postgres",
        help="Postgres password (default: postgres)",
    )
    ingest_parser.add_argument(
        "--db-name",
        type=str,
        default="epstien_files_db",
        help="Postgres database name (default: epstien-files_db)",
    )
    ingest_parser.set_defaults(func=_run_ingest)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()