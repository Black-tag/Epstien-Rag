import logging
from pathlib import Path
from typing import Sequence, Sequence as Seq

import chromadb
from langchain_core.documents import Document


logger = logging.getLogger(__name__)


def upsert_chunk_embeddings(
    chunks: Sequence[Document],
    embeddings: Seq[Seq[float]],
    persist_directory: Path,
    collection_name: str,
) -> None:
    """
    Store chunk embeddings in a persistent Chroma collection.

    - Uses each chunk's metadata (chunk_id, source_path, file_name, section, etc.).
    - Does not recompute embeddings; uses the provided vectors.
    """
    if not chunks or not embeddings:
        logger.info("No chunks or embeddings provided; nothing to upsert into Chroma.")
        return

    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Number of chunks ({len(chunks)}) does not match number of embeddings ({len(embeddings)})"
        )

    persist_directory = persist_directory.expanduser().resolve()
    persist_directory.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Upserting %d chunk embeddings into Chroma at %s (collection=%s)",
        len(chunks),
        persist_directory,
        collection_name,
    )

    client = chromadb.PersistentClient(path=str(persist_directory))
    collection = client.get_or_create_collection(name=collection_name)

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []
    vectors: list[list[float]] = []

    for chunk, vector in zip(chunks, embeddings):
        metadata = dict(chunk.metadata or {})
        source_path = metadata.get("source_path", "")
        file_name = metadata.get("file_name", "")
        chunk_id = metadata.get("chunk_id")
        chunk_index = metadata.get("chunk_index", 0)

        # Build a stable, unique id per chunk
        if chunk_id:
            doc_id = f"{source_path}::{chunk_id}"
        else:
            doc_id = f"{source_path}::chunk-{chunk_index}"

        ids.append(doc_id)
        documents.append(chunk.page_content or "")
        metadatas.append(metadata)
        vectors.append(list(vector))

    # Chroma expects flat lists; it handles upsert automatically when ids already exist.
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=vectors,
    )

    logger.info("Chroma upsert complete for %d chunk embeddings.", len(ids))

