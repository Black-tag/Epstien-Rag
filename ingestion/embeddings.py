import logging
from typing import Iterable, List, Sequence

import requests
import tiktoken
from langchain_core.documents import Document


logger = logging.getLogger(__name__)

# Use the same tokenizer family as the rest of the pipeline
_ENCODING = tiktoken.get_encoding("cl100k_base")


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """
    Truncate text to at most max_tokens using the shared tokenizer.
    """
    if max_tokens <= 0:
        return text

    token_ids = _ENCODING.encode(text)
    if len(token_ids) <= max_tokens:
        return text
    truncated_ids = token_ids[:max_tokens]
    return _ENCODING.decode(truncated_ids)


def embed_chunks_with_ollama(
    chunks: Sequence[Document],
    model: str = "snowflake-arctic-embed",
    endpoint: str = "http://localhost:11434/api/embed",
    batch_size: int = 64,
    max_tokens: int = 512,
) -> List[List[float]]:
    """
    Generate embeddings for a sequence of chunk Documents using a local
    Ollama instance running a Snowflake Arctic Embed model.

    - Supports batching so very large corpora do not exceed request limits.
    - The order of the returned embeddings matches the order of the input
      chunks.
    """
    if not chunks:
        return []

    if batch_size <= 0:
        batch_size = 64

    all_embeddings: List[List[float]] = []

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]

        texts: List[str] = []
        for idx, chunk in enumerate(batch):
            raw_text = (chunk.page_content or "").strip()
            if not raw_text:
                logger.warning(
                    "Skipping empty chunk at global index %d (batch offset %d)",
                    start + idx,
                    idx,
                )
                continue

            # Ensure we never exceed the model's context window
            text = _truncate_to_tokens(raw_text, max_tokens=max_tokens)
            texts.append(text)

        # If all chunks in this batch were empty after cleaning, skip the call
        if not texts:
            logger.info(
                "No non-empty texts in batch [%d:%d); skipping Ollama call for this batch.",
                start,
                start + len(batch),
            )
            continue

        payload = {
            "model": model,
            "input": texts,
        }

        try:
            logger.info(
                "Requesting embeddings for batch [%d:%d) (size=%d) from Ollama model '%s'",
                start,
                start + len(batch),
                len(batch),
                model,
            )
            resp = requests.post(endpoint, json=payload, timeout=300)
            if resp.status_code != 200:
                logger.error(
                    "Ollama embed API returned %s with body: %s",
                    resp.status_code,
                    resp.text,
                )
                resp.raise_for_status()
        except Exception as exc:
            logger.exception(
                "Failed to call Ollama embed API for batch starting at %d: %s",
                start,
                exc,
            )
            raise

        data = resp.json()

        # Ollama's embed API returns an "embeddings" field that is a list of
        # vectors, one per input item.
        embeddings: Iterable[Iterable[float]] = data.get("embeddings") or []
        embeddings_list: List[List[float]] = [list(vec) for vec in embeddings]

        if len(embeddings_list) != len(batch):
            logger.error(
                "Mismatched embeddings for batch starting at %d: expected %d vectors, got %d",
                start,
                len(batch),
                len(embeddings_list),
            )
            raise RuntimeError(
                f"Expected {len(batch)} embeddings, got {len(embeddings_list)} for batch starting at {start}"
            )

        all_embeddings.extend(embeddings_list)

    logger.info("Successfully generated %d embeddings from Ollama", len(all_embeddings))
    return all_embeddings

