"""
ingestion/embeddings.py
-----------------------
Ollama embedding client with multi-layer truncation and automatic retry.

Truncation layers (applied in order for every chunk text):
  1. Hard character cap  – cheap, model-agnostic safety net.
  2. Token cap (cl100k_base proxy) – keeps the tiktoken count conservative
     to account for the fact that snowflake-arctic-embed uses a BERT-style
     tokenizer that produces more tokens than cl100k_base for the same text.

Retry strategy:
  If a full batch is rejected by Ollama with a 400 context-length error,
  the batch is automatically decomposed and each item is retried
  individually with an even harder character cap (_FALLBACK_MAX_CHARS).
  This guarantees that the pipeline completes regardless of chunk size or
  document content.
"""

from __future__ import annotations

import logging
from typing import List, Sequence

import requests
import tiktoken
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tokenizer (used as a proxy – not the model's own tokenizer)
# ---------------------------------------------------------------------------

_ENCODING = tiktoken.get_encoding("cl100k_base")

# ---------------------------------------------------------------------------
# Truncation constants
# ---------------------------------------------------------------------------

# Conservative token budget using cl100k_base.
# snowflake-arctic-embed's BERT tokenizer routinely produces 1.5-2x the
# number of tokens that cl100k_base does for the same text, so 256 proxy
# tokens gives comfortable headroom inside the model's 512-token window.
_DEFAULT_MAX_TOKENS: int = 256

# Hard character cap applied before tokenisation – a cheap first filter.
# 2 000 characters is a generous upper bound; typical 800-char chunks are
# well below this.
_DEFAULT_MAX_CHARS: int = 2_000

# Even harder cap used when a batch fails with a context-length 400 error
# and we fall back to per-item retry.  500 characters ≈ 2–3 sentences and
# will fit inside virtually any embedding model's context window.
_FALLBACK_MAX_CHARS: int = 500


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _truncate(text: str, max_chars: int, max_tokens: int) -> str:
    """
    Truncate *text* so it fits within both the character and token budgets.

    Parameters
    ----------
    text:
        Raw input string.
    max_chars:
        Hard character limit applied first (O(1) operation).
    max_tokens:
        Token limit using the cl100k_base proxy tokeniser.
        Disabled when <= 0.

    Returns
    -------
    str
        Truncated text, guaranteed to be <= max_chars characters and
        <= max_tokens cl100k_base tokens.
    """
    if not text:
        return text

    # Layer 1 – character cap
    if len(text) > max_chars:
        text = text[:max_chars]

    # Layer 2 – token cap
    if max_tokens > 0:
        token_ids = _ENCODING.encode(text)
        if len(token_ids) > max_tokens:
            text = _ENCODING.decode(token_ids[:max_tokens])

    return text


def _is_context_length_error(response: requests.Response) -> bool:
    """Return True when Ollama signals that the input is too long."""
    if response.status_code != 400:
        return False
    body = response.text.lower()
    return "context" in body or "input length" in body or "exceeds" in body


def _post_embed(
    texts: List[str],
    model: str,
    endpoint: str,
    timeout: int,
) -> List[List[float]]:
    """
    POST one embed request to Ollama and return the raw embedding vectors.

    Raises
    ------
    requests.HTTPError
        On any non-200 response (caller decides whether to retry).
    """
    resp = requests.post(
        endpoint,
        json={"model": model, "input": texts},
        timeout=timeout,
    )
    if resp.status_code != 200:
        logger.error(
            "Ollama embed API returned %s: %s",
            resp.status_code,
            resp.text,
        )
        resp.raise_for_status()

    data = resp.json()
    return [list(vec) for vec in (data.get("embeddings") or [])]


def _embed_with_fallback(
    texts: List[str],
    model: str,
    endpoint: str,
    timeout: int,
    batch_start: int,
) -> List[List[float]]:
    """
    Embed *texts* as a single batch.  If Ollama rejects the batch with a
    context-length 400 error, fall back to embedding each text individually
    using ``_FALLBACK_MAX_CHARS`` truncation.

    Parameters
    ----------
    texts:
        Pre-truncated texts for this batch (1:1 with the batch's chunks).
    model / endpoint / timeout:
        Forwarded to Ollama.
    batch_start:
        Absolute offset of the first item in this batch, used for logging.

    Returns
    -------
    list[list[float]]
        One embedding vector per input text, in the same order.

    Raises
    ------
    requests.HTTPError
        If a context-length error persists even after per-item fallback
        truncation, or if Ollama returns a different kind of error.
    RuntimeError
        If the number of returned vectors does not match len(texts).
    """
    batch_end = batch_start + len(texts)

    # ------------------------------------------------------------------
    # Attempt 1 – full batch
    # ------------------------------------------------------------------
    try:
        logger.info(
            "Requesting embeddings for batch [%d:%d) (size=%d) from Ollama model '%s'.",
            batch_start,
            batch_end,
            len(texts),
            model,
        )
        vectors = _post_embed(texts, model, endpoint, timeout)

    except requests.HTTPError as exc:
        resp = exc.response
        if resp is not None and _is_context_length_error(resp):
            # ----------------------------------------------------------
            # Attempt 2 – per-item with hard fallback truncation
            # ----------------------------------------------------------
            logger.warning(
                "Batch [%d:%d) rejected (context-length error). "
                "Retrying each item individually with hard char cap=%d.",
                batch_start,
                batch_end,
                _FALLBACK_MAX_CHARS,
            )
            vectors = []
            for i, text in enumerate(texts):
                fallback_text = _truncate(
                    text,
                    max_chars=_FALLBACK_MAX_CHARS,
                    max_tokens=0,  # char cap is enough at this level
                )
                logger.debug(
                    "  Item %d/%d: original_len=%d fallback_len=%d.",
                    i + 1,
                    len(texts),
                    len(text),
                    len(fallback_text),
                )
                item_vectors = _post_embed([fallback_text], model, endpoint, timeout)
                if not item_vectors:
                    raise RuntimeError(
                        f"Ollama returned no embedding for item {batch_start + i} "
                        f"even after fallback truncation."
                    )
                vectors.extend(item_vectors)

            logger.info(
                "Per-item fallback succeeded for batch [%d:%d).",
                batch_start,
                batch_end,
            )
        else:
            # Not a context-length error – propagate immediately.
            raise

    # ------------------------------------------------------------------
    # Sanity-check: Ollama must return exactly one vector per input text.
    # ------------------------------------------------------------------
    if len(vectors) != len(texts):
        raise RuntimeError(
            f"Embedding count mismatch for batch [{batch_start}:{batch_end}): "
            f"sent {len(texts)} texts, received {len(vectors)} vectors."
        )

    return vectors


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def embed_chunks_with_ollama(
    chunks: Sequence[Document],
    model: str = "snowflake-arctic-embed",
    endpoint: str = "http://localhost:11434/api/embed",
    batch_size: int = 64,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    max_chars: int = _DEFAULT_MAX_CHARS,
) -> List[List[float]]:
    """
    Generate embeddings for *chunks* using a local Ollama instance.

    Every chunk text is first truncated to ``max_chars`` characters then to
    ``max_tokens`` cl100k_base tokens before being sent to the model.  If a
    batch is still rejected with a context-length error, the batch is retried
    one item at a time with a hard ``_FALLBACK_MAX_CHARS`` cap, ensuring the
    pipeline always completes.

    The returned list is in the **same order** as *chunks* and has the
    **same length** – one vector per chunk.

    Parameters
    ----------
    chunks:
        Input documents whose ``page_content`` will be embedded.
    model:
        Ollama model name (must be running locally).
    endpoint:
        Full URL of the Ollama ``/api/embed`` endpoint.
    batch_size:
        Number of texts sent per HTTP request.  Smaller values reduce the
        chance of hitting the context window on the first attempt.
    max_tokens:
        cl100k_base token budget per text before sending.  Default 256 gives
        a safety margin for BERT-style tokenisers that produce more tokens.
    max_chars:
        Hard character cap applied before tokenisation.  Default 2 000.

    Returns
    -------
    list[list[float]]
        Embedding vectors, one per input chunk, in input order.
    """
    if not chunks:
        return []

    if batch_size <= 0:
        batch_size = 64

    all_embeddings: List[List[float]] = []

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]

        # ------------------------------------------------------------------
        # Prepare texts: truncate and maintain strict 1:1 with batch items.
        # Empty chunks are replaced with a single space so the index
        # correspondence between chunk_db_ids and embeddings is never broken.
        # ------------------------------------------------------------------
        texts: List[str] = []
        for idx, chunk in enumerate(batch):
            raw = (chunk.page_content or "").strip()
            if not raw:
                logger.warning(
                    "Chunk at global index %d is empty; using placeholder for embedding.",
                    start + idx,
                )
                raw = " "

            texts.append(
                _truncate(raw, max_chars=max_chars, max_tokens=max_tokens)
            )

        # ------------------------------------------------------------------
        # Call Ollama with automatic fallback on context-length errors.
        # ------------------------------------------------------------------
        batch_embeddings = _embed_with_fallback(
            texts=texts,
            model=model,
            endpoint=endpoint,
            timeout=300,
            batch_start=start,
        )

        all_embeddings.extend(batch_embeddings)

    logger.info(
        "Embedding complete: %d vector(s) generated from Ollama model '%s'.",
        len(all_embeddings),
        model,
    )
    return all_embeddings
