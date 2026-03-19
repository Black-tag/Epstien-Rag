"""
Ingest router – POST /api/ingest

Runs the complete pipeline in a single background job:

  1. [ETL]       Scan the document repository, parse, chunk, deduplicate
  2. [Postgres]  Upsert documents + chunks (embedding column starts NULL)
  3. [Ollama]    Generate embeddings for the inserted chunk texts
  4. [pgvector]  UPDATE chunks SET embedding = %s::vector for each chunk

Returns 202 Accepted immediately with a job_id.
Poll GET /api/jobs/{job_id} for progress and the final result.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Tuple

from fastapi import APIRouter, BackgroundTasks
from langchain_core.documents import Document

from api.models.requests import IngestRequest
from api.models.responses import JobResponse
from api.services.job_manager import JobStatus, job_manager
from config.loader import load_properties
from ingestion import IngestionConfig, IngestionState, ingest_repository
from ingestion.embeddings import embed_chunks_with_ollama
from storage import PostgresConfig, store_documents_and_chunks, update_chunk_embeddings

router = APIRouter(prefix="/api", tags=["Ingestion"])
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Background task
# ---------------------------------------------------------------------------


def _ingest_task(job_id: str, req: IngestRequest) -> None:
    """
    Full pipeline worker.

    Configuration priority for every setting:
        1. Explicit field in the request body
        2. config/properties.yaml
        3. Environment variable  (repository path only)
        4. Hard-coded default
    """
    job_manager.update_job(job_id, JobStatus.RUNNING)

    try:
        project_root = Path(__file__).resolve().parents[2]

        properties: dict = load_properties()
        ingestion_cfg: dict = properties.get("ingestion", {}) or {}
        db_cfg: dict = properties.get("database", {}) or {}
        ollama_cfg: dict = properties.get("ollama", {}) or {}
        embedding_cfg: dict = properties.get("embedding", {}) or {}

        embedding_dimensions: int = int(embedding_cfg.get("dimensions", 1024))

        # ----------------------------------------------------------------
        # Step 1 – resolve repository path
        # ----------------------------------------------------------------
        env_repo = os.getenv("EPSTEIN_DOCS_PATH")

        if req.repository:
            repository_path = Path(req.repository).expanduser().resolve()
        elif env_repo:
            repository_path = Path(env_repo).expanduser().resolve()
        elif ingestion_cfg.get("repository_path"):
            repository_path = (
                Path(ingestion_cfg["repository_path"]).expanduser().resolve()
            )
        else:
            repository_path = project_root / "epstein-documents"

        if not repository_path.exists() and not req.repository and not env_repo:
            logger.info(
                "Job %s: default repository path %s not found – creating it.",
                job_id,
                repository_path,
            )
            repository_path.mkdir(parents=True, exist_ok=True)

        # ----------------------------------------------------------------
        # Step 2 – resolve state file
        # ----------------------------------------------------------------
        if req.state_file:
            state_file = Path(req.state_file).resolve()
        elif ingestion_cfg.get("state_file"):
            state_file = (project_root / ingestion_cfg["state_file"]).resolve()
        else:
            state_file = project_root / "data" / "ingestion_state.json"

        # ----------------------------------------------------------------
        # Step 3 – chunking parameters
        # ----------------------------------------------------------------
        chunk_size: int = req.chunk_size or int(ingestion_cfg.get("chunk_size", 800))
        chunk_overlap: int = req.chunk_overlap or int(
            ingestion_cfg.get("chunk_overlap", 200)
        )

        # ----------------------------------------------------------------
        # Step 4 – ETL: scan, parse, chunk, deduplicate
        # ----------------------------------------------------------------
        config = IngestionConfig(
            repository_path=repository_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            state_file=state_file,
        )
        state = IngestionState.load(config.state_file)

        logger.info(
            "Job %s: running ETL on %s (chunk_size=%d, chunk_overlap=%d).",
            job_id,
            repository_path,
            chunk_size,
            chunk_overlap,
        )

        output = ingest_repository(config=config, state=state)
        documents: list[Document] = output.get("documents", [])
        chunks: list[Document] = output.get("chunks", [])

        logger.info(
            "Job %s: ETL produced %d document(s) and %d chunk(s).",
            job_id,
            len(documents),
            len(chunks),
        )

        if not documents and not chunks:
            logger.info("Job %s: nothing to process – marking complete.", job_id)
            job_manager.update_job(
                job_id,
                JobStatus.COMPLETED,
                result={"documents_count": 0, "chunks_count": 0, "chunks_embedded": 0},
            )
            return

        # ----------------------------------------------------------------
        # Step 5 – persist documents + chunks to Postgres
        #           store_documents_and_chunks returns the (db_id, content)
        #           pair for every successfully inserted chunk row so we
        #           never need to reload from the database in this pipeline.
        # ----------------------------------------------------------------
        pg_cfg = PostgresConfig(
            host=req.db_host or db_cfg.get("host", "localhost"),
            port=req.db_port or int(db_cfg.get("port", 5432)),
            user=req.db_user or db_cfg.get("user", "postgres"),
            password=req.db_password or db_cfg.get("password", "postgres"),
            dbname=req.db_name or db_cfg.get("name", "epstien_files_db"),
        )

        chunk_records: List[Tuple[int, str]] = store_documents_and_chunks(
            documents=documents,
            chunks=chunks,
            cfg=pg_cfg,
            embedding_dimensions=embedding_dimensions,
        )

        logger.info(
            "Job %s: %d document(s) and %d chunk(s) persisted to Postgres.",
            job_id,
            len(documents),
            len(chunk_records),
        )

        # ----------------------------------------------------------------
        # Step 6 – generate embeddings via Ollama
        #           We wrap the stored chunk contents in lightweight Document
        #           objects so embed_chunks_with_ollama can access .page_content.
        #           The chunk_db_ids list is kept in the same order so that
        #           chunk_db_ids[i] matches embeddings[i] exactly.
        # ----------------------------------------------------------------
        chunks_embedded = 0

        if chunk_records:
            chunk_db_ids: List[int] = [db_id for db_id, _ in chunk_records]
            embed_docs: List[Document] = [
                Document(page_content=content) for _, content in chunk_records
            ]

            ollama_model: str = ollama_cfg.get(
                "embedding_model", "snowflake-arctic-embed"
            )
            ollama_endpoint: str = ollama_cfg.get(
                "endpoint", "http://localhost:11434/api/embed"
            )
            batch_size: int = int(embedding_cfg.get("batch_size", 64))

            logger.info(
                "Job %s: requesting embeddings from Ollama model '%s' "
                "(chunks=%d, batch_size=%d).",
                job_id,
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
            except Exception as embed_exc:
                # Postgres write succeeded; flag the embedding failure clearly
                # so the operator knows the embedding column was not updated.
                logger.exception(
                    "Job %s: Ollama embedding failed – %s", job_id, embed_exc
                )
                job_manager.update_job(
                    job_id,
                    JobStatus.FAILED,
                    error=(
                        f"Documents and chunks were persisted to Postgres "
                        f"({len(chunk_records)} chunk(s)) but Ollama embedding "
                        f"failed: {embed_exc}"
                    ),
                )
                return

            logger.info(
                "Job %s: received %d embedding vector(s) from Ollama.",
                job_id,
                len(embeddings),
            )

            # ----------------------------------------------------------------
            # Step 7 – write embeddings back to Postgres via pgvector
            #           UPDATE chunks SET embedding = %s::vector WHERE id = %s
            # ----------------------------------------------------------------
            update_chunk_embeddings(
                chunk_db_ids=chunk_db_ids,
                embeddings=embeddings,
                cfg=pg_cfg,
            )

            chunks_embedded = len(embeddings)

            logger.info(
                "Job %s: %d embedding vector(s) written to Postgres chunks.embedding.",
                job_id,
                chunks_embedded,
            )
        else:
            logger.info(
                "Job %s: no chunk rows were inserted – skipping embedding step.",
                job_id,
            )

        # ----------------------------------------------------------------
        # Step 8 – mark job complete
        # ----------------------------------------------------------------
        job_manager.update_job(
            job_id,
            JobStatus.COMPLETED,
            result={
                "documents_count": len(documents),
                "chunks_count": len(chunk_records),
                "chunks_embedded": chunks_embedded,
            },
        )

        logger.info(
            "Job %s COMPLETED: %d doc(s) | %d chunk(s) | %d embedding(s).",
            job_id,
            len(documents),
            len(chunk_records),
            chunks_embedded,
        )

    except Exception as exc:
        logger.exception("Job %s (ingest) raised an unhandled exception: %s", job_id, exc)
        job_manager.update_job(job_id, JobStatus.FAILED, error=str(exc))


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.post(
    "/ingest",
    response_model=JobResponse,
    status_code=202,
    summary="Run the full ingestion + embedding pipeline",
    description=(
        "Executes the **complete pipeline** as a background job:\n\n"
        "1. **ETL** – scan the document repository, parse every file, chunk and deduplicate\n"
        "2. **Postgres** – upsert documents and chunks (embedding starts as NULL)\n"
        "3. **Ollama** – generate vector embeddings for every inserted chunk\n"
        "4. **pgvector** – `UPDATE chunks SET embedding = …` for each chunk\n\n"
        "Returns `202 Accepted` immediately with a `job_id`.  "
        "Poll **`GET /api/jobs/{job_id}`** to track progress and retrieve the "
        "final counts (`documents_count`, `chunks_count`, `chunks_embedded`) "
        "once the job reaches `completed` status.\n\n"
        "All request-body fields are **optional** – omitted values fall back to "
        "`config/properties.yaml`, environment variables, or built-in defaults."
    ),
    responses={
        202: {
            "description": "Job accepted and queued.",
            "content": {
                "application/json": {
                    "example": {
                        "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                        "status": "pending",
                        "message": "Ingestion job accepted and queued.",
                        "result": None,
                        "error": None,
                    }
                }
            },
        }
    },
)
def start_ingest(
    req: IngestRequest,
    background_tasks: BackgroundTasks,
) -> JobResponse:
    """
    Accept an ingestion request, enqueue it as a background job, and return
    immediately with the assigned job_id.
    """
    job = job_manager.create_job("ingest")
    background_tasks.add_task(_ingest_task, job.job_id, req)

    logger.info("Ingest job %s accepted.", job.job_id)

    return JobResponse(
        job_id=job.job_id,
        status=JobStatus.PENDING,
        message="Ingestion job accepted and queued.",
    )
