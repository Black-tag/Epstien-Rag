import hashlib
import json
import logging
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import tiktoken
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredFileLoader,
)
from langchain_core.documents import Document


logger = logging.getLogger(__name__)

# Global tokenizer for consistent token counting
_ENCODING = tiktoken.get_encoding("cl100k_base")


@dataclass
class IngestionConfig:
    """Configuration for the ingestion pipeline."""

    repository_path: Path
    chunk_size: int = 800
    chunk_overlap: int = 200
    encoding: str = "utf-8"
    # File extensions to process; keys without dot for convenience
    allowed_extensions: Sequence[str] = field(
        default_factory=lambda: ["pdf", "txt", "md", "html", "eml", "docx"]
    )
    # Path to a JSON file where ingestion state is stored (for incremental runs)
    state_file: Optional[Path] = None


@dataclass
class IngestionState:
    """
    Tracks what has already been ingested for incremental / scheduled runs.

    - file_hashes: content hash per absolute path
    - last_run_at: timestamp string of last successful ingestion
    """

    file_hashes: Dict[str, str] = field(default_factory=dict)
    last_run_at: Optional[str] = None

    @classmethod
    def load(cls, path: Optional[Path]) -> "IngestionState":
        if path is None or not path.exists():
            return cls()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Failed to load ingestion state from %s; starting fresh", path)
            return cls()
        return cls(
            file_hashes=raw.get("file_hashes", {}),
            last_run_at=raw.get("last_run_at"),
        )

    def save(self, path: Optional[Path]) -> None:
        if path is None:
            return
        payload = asdict(self)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


@dataclass
class IngestionResult:
    """Summary of an ingestion run."""

    processed_files: int
    skipped_files: int
    new_documents: int
    new_chunks: int


def _compute_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_text(text: str) -> str:
    """Basic text normalization (line endings and blank lines)."""
    # Strip BOMs and normalize line endings
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse repeated blank lines
    lines = [line.rstrip() for line in normalized.split("\n")]
    cleaned_lines: List[str] = []
    previous_blank = False
    for line in lines:
        is_blank = len(line.strip()) == 0
        if is_blank and previous_blank:
            continue
        cleaned_lines.append(line)
        previous_blank = is_blank
    return "\n".join(cleaned_lines).strip()


def clean_text(text: str) -> str:
    """
    Clean text before chunking:
    - normalize whitespace and blank lines
    - collapse multiple newlines
    - remove repeated page headers/footers when they are detected as
      short, frequently repeated, non-sentence lines.
    """
    normalized = _normalize_text(text)
    lines = normalized.split("\n")

    # Count line frequencies to detect repeated headers/footers
    freq: Dict[str, int] = {}
    for line in lines:
        key = line.strip()
        if not key:
            continue
        freq[key] = freq.get(key, 0) + 1

    removable: set[str] = set()
    for key, count in freq.items():
        # Heuristic for header/footer-like lines:
        # - appears on at least 3 lines
        # - relatively short
        # - does not contain obvious sentence punctuation
        if (
            count >= 3
            and 5 <= len(key) <= 80
            and not any(punct in key for punct in ".!?")
        ):
            removable.add(key)

    cleaned_lines: List[str] = []
    for line in lines:
        if line.strip() in removable:
            continue
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)
    return _normalize_text(cleaned)


EMAIL_HEADER_KEYS = {
    "from": "from",
    "to": "to",
    "cc": "cc",
    "bcc": "bcc",
    "subject": "subject",
    "sent": "date",
    "date": "date",
    "attachments": "attachments",
    "sensitivity": "sensitivity",
    "importance": "importance",
}


def extract_email_headers(text: str) -> tuple[Dict[str, str], str]:
    """
    Extract a leading email-style header block into metadata and
    return (headers, body_text_without_headers).

    Only treats the top block as headers if at least one known key is found.
    """
    lines = text.splitlines()
    header_lines: List[str] = []
    body_start_index = 0
    for idx, line in enumerate(lines):
        # Header block ends at the first completely blank line
        if line.strip() == "":
            body_start_index = idx + 1
            break
        header_lines.append(line)

    headers: Dict[str, str] = {}
    for line in header_lines:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key_norm = key.strip().lower()
        mapped = EMAIL_HEADER_KEYS.get(key_norm)
        if mapped:
            headers[mapped] = value.strip()

    # If we did not detect any known header keys, treat the text as-is
    if not headers:
        return {}, text

    body_lines = lines[body_start_index:]
    body_text = "\n".join(body_lines).lstrip("\n")
    return headers, body_text


def _token_count(text: str) -> int:
    """Return an approximate token count using tiktoken."""
    if not text:
        return 0
    # cl100k_base is compatible with most modern chat/embedding models
    return len(_ENCODING.encode(text))


def _is_header_like_line(line: str) -> bool:
    """
    Heuristic to detect header-like lines within a chunk.
    """
    stripped = line.strip()
    if not stripped:
        return False

    # Strong signals: known email header prefixes
    lowered = stripped.lower()
    for prefix in (
        "from:",
        "to:",
        "cc:",
        "bcc:",
        "subject:",
        "sent:",
        "date:",
        "attachments:",
        "re:",
        "fw:",
    ):
        if lowered.startswith(prefix):
            return True

    # Generic "Key: Value" style with no sentence punctuation
    if ":" in stripped:
        key, _ = stripped.split(":", 1)
        if key.isalpha() and not any(p in stripped for p in ".!?"):
            return True

    return False


def is_valid_chunk(text: str) -> bool:
    """
    Decide whether a candidate chunk is valuable enough to keep.

    Rules:
    - discard if length < 120 characters
    - discard if majority of non-empty lines are header-like
    - discard if there is not at least one full sentence
    """
    stripped = text.strip()
    if len(stripped) < 120:
        return False

    lines = [ln.strip() for ln in stripped.splitlines() if ln.strip()]
    if not lines:
        return False

    header_like = sum(1 for ln in lines if _is_header_like_line(ln))
    if header_like / len(lines) > 0.6:
        return False

    # Require at least one "full sentence" via presence of punctuation
    if not re.search(r"[.!?]\s", stripped):
        return False

    return True


def paragraph_chunk(
    text: str,
    min_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
) -> List[str]:
    """
    Paragraph-aware chunking:
    - split on double newlines into paragraphs
    - merge paragraphs within a section until reaching ~max_tokens
    - ensure consecutive chunks overlap by overlap_tokens
    - if a single paragraph is longer than max_tokens, split it manually
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    chunks: List[str] = []
    current_tokens: List[str] = []

    def flush_current() -> None:
        nonlocal current_tokens
        if not current_tokens:
            return
        chunk_text = " ".join(current_tokens).strip()
        if chunk_text:
            chunks.append(chunk_text)
        # Prepare overlap for next chunk
        if overlap_tokens > 0 and current_tokens:
            overlap = current_tokens[-overlap_tokens:]
        else:
            overlap = []
        current_tokens = list(overlap)

    for para in paragraphs:
        # Handle extremely large paragraphs by hard-splitting them
        if _token_count(para) > max_tokens:
            words = para.split()
            for i in range(0, len(words), max_tokens):
                segment = " ".join(words[i : i + max_tokens])
                if segment.strip():
                    chunks.append(segment.strip())
            continue

        para_tokens = para.split()
        # If adding this paragraph would exceed max_tokens, flush first
        if current_tokens and _token_count(" ".join(current_tokens + para_tokens)) > max_tokens:
            flush_current()

        current_tokens.extend(para_tokens)

        # If we've reached at least min_tokens, we can flush opportunistically
        if _token_count(" ".join(current_tokens)) >= min_tokens:
            flush_current()

    # Flush any remaining tokens
    if current_tokens:
        flush_current()

    return chunks


def filter_newsletter_noise(line: str) -> bool:
    """
    Return True if the line looks like newsletter UI / boilerplate that
    should be removed from content (share buttons, unsubscribe, etc.).
    """
    lowered = line.strip().lower()
    if not lowered:
        return False

    noise_phrases = [
        "share",
        "subscribe",
        "privacy policy",
        "unsubscribe",
        "listen to",
        "videos of the day",
        "good reads from elsewhere",
        "social media speed read",
        "view this email in your browser",
        "manage your subscriptions",
        "update your preferences",
    ]
    return any(phrase in lowered for phrase in noise_phrases)


def detect_section_heading(line: str) -> Optional[str]:
    """
    Detect section headings such as:
    - ALL CAPS lines (with letters)
    - short lines ending with ':'
    - but excluding numbered list items.
    """
    stripped = line.strip()
    if not stripped:
        return None

    if len(stripped) > 100:
        return None

    # Exclude obvious numbered list items (handled elsewhere)
    if re.match(r"^\d+[\).\)]\s", stripped):
        return None

    # ALL CAPS with at least one letter
    if stripped.isupper() and any(c.isalpha() for c in stripped):
        return stripped

    # Headings ending with ":" that aren't too long
    if stripped.endswith(":"):
        return stripped.rstrip(":")

    return None


def extract_title(text: str) -> tuple[Optional[str], str]:
    """
    Extract a likely document title near the top of the text.

    Heuristics:
    - short line (< 120 chars)
    - not ALL CAPS
    - appears in the first ~20 non-empty lines
    - surrounded by blank lines when possible
    """
    lines = text.splitlines()
    title: Optional[str] = None
    title_index: Optional[int] = None
    non_empty_seen = 0

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        non_empty_seen += 1
        if non_empty_seen > 20:
            break

        if len(stripped) < 120 and not stripped.isupper():
            # Heuristic: treat first suitable line as title
            title = stripped
            title_index = idx
            break

    if title is None or title_index is None:
        return None, text

    # Remove the title line from the body text
    remaining_lines = lines[:title_index] + lines[title_index + 1 :]
    body = "\n".join(remaining_lines).lstrip("\n")
    return title, body


def _extension(path: Path) -> str:
    return path.suffix.lstrip(".").lower()


def _load_file(path: Path, encoding: str) -> List[Document]:
    """Load a single file into LangChain Documents, with metadata."""
    ext = _extension(path)

    if ext == "pdf":
        loader = PyPDFLoader(str(path))
    elif ext in {"txt", "md"}:
        loader = TextLoader(str(path), encoding=encoding)
    else:
        # Fallback to unstructured loader for other document types
        loader = UnstructuredFileLoader(str(path))

    docs = loader.load()

    for doc in docs:
        # Enrich metadata for traceability
        metadata = dict(doc.metadata or {})
        metadata.update(
            {
                "source_path": str(path.resolve()),
                "file_name": path.name,
                "extension": ext,
                "last_modified": datetime.fromtimestamp(
                    path.stat().st_mtime
                ).isoformat(),
            }
        )
        # Rough document type tagging; can be refined later
        if ext == "eml":
            metadata.setdefault("document_type", "email")
        elif ext in {"md"}:
            metadata.setdefault("document_type", "markdown")
        else:
            metadata.setdefault("document_type", "generic")

        doc.metadata = metadata
    return docs


def _deduplicate_documents(
    docs: Sequence[Document],
    state: IngestionState,
) -> List[Document]:
    """
    Deduplicate documents based on content hash.

    This operates at the file-level granularity by default (same path & hash),
    but can be extended to chunk-level deduplication if needed.
    """
    deduped: List[Document] = []
    for doc in docs:
        source_path = doc.metadata.get("source_path")
        if not source_path:
            deduped.append(doc)
            continue
        text_hash = _compute_text_hash(doc.page_content)
        previous_hash = state.file_hashes.get(source_path)
        if previous_hash == text_hash:
            logger.info("Skipping unchanged document: %s", source_path)
            continue
        state.file_hashes[source_path] = text_hash
        deduped.append(doc)
    return deduped


def _chunk_documents(
    docs: Sequence[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    """
    Chunk documents using structure-aware, paragraph-based logic.

    Assumes:
    - doc.page_content has already been cleaned
    - email headers have already been extracted into doc.metadata
    """
    chunked_with_ids: List[Document] = []

    # Interpret chunk_size / chunk_overlap as token-based parameters
    max_tokens = max(chunk_size, 200)
    min_tokens = max(int(max_tokens * 2 / 3), 100)
    overlap_tokens = max(chunk_overlap, 0)

    for doc in docs:
        base_metadata = dict(doc.metadata or {})
        # Ensure we preserve existing metadata and core fields
        base_metadata.setdefault("source", base_metadata.get("source_path"))
        extension = base_metadata.get("extension", "")
        document_type = base_metadata.get("document_type", "generic")

        # Start from the cleaned body that already has email headers removed
        text = doc.page_content or ""

        # Detect a document-level title near the top, if any
        title, body_without_title = extract_title(text)
        if title:
            base_metadata.setdefault("title", title)
            working_text = body_without_title
        else:
            working_text = text

        # Split into lines for section / heading analysis
        all_lines = working_text.splitlines()

        sections: List[Dict[str, object]] = []
        current_section_name: Optional[str] = None
        current_section_lines: List[str] = []

        def flush_section() -> None:
            nonlocal current_section_lines, current_section_name
            if current_section_lines:
                sections.append(
                    {
                        "name": current_section_name,
                        "lines": list(current_section_lines),
                    }
                )
                current_section_lines = []

        for line in all_lines:
            if filter_newsletter_noise(line):
                continue
            heading = detect_section_heading(line)
            if heading:
                # Start a new section at headings
                flush_section()
                current_section_name = heading
                continue
            current_section_lines.append(line)

        flush_section()

        # Fallback if we never detected any heading: treat the whole body as one section
        if not sections:
            sections.append({"name": None, "lines": all_lines})

        # Chunk within each section independently to avoid mixing topics
        chunk_index = 0
        for section in sections:
            section_name = section.get("name")
            lines = [ln for ln in section.get("lines", []) if ln is not None]

            # Build section text and keep numbered list items separate
            numbered_pattern = re.compile(r"^\s*\d+[\).\)]\s+")
            current_paragraph_lines: List[str] = []

            def flush_paragraph_block() -> None:
                nonlocal current_paragraph_lines, chunk_index
                if not current_paragraph_lines:
                    return
                block_text = "\n".join(current_paragraph_lines).strip()
                current_paragraph_lines = []
                if not block_text:
                    return
                # Paragraph-aware chunking within this block
                for chunk_text in paragraph_chunk(
                    block_text,
                    min_tokens=min_tokens,
                    max_tokens=max_tokens,
                    overlap_tokens=overlap_tokens,
                ):
                    if not is_valid_chunk(chunk_text):
                        continue
                    metadata = dict(base_metadata)
                    if section_name:
                        metadata["section"] = section_name
                    metadata["chunk_index"] = chunk_index
                    metadata["chunk_id"] = f"{metadata.get('file_name')}_{chunk_index}"
                    chunk_index += 1
                    chunk_content = chunk_text
                    if section_name:
                        chunk_content = f"Section: {section_name}\n\n{chunk_text}"
                    chunk_doc = Document(page_content=chunk_content, metadata=metadata)
                    chunked_with_ids.append(chunk_doc)

            for ln in lines:
                stripped = ln.strip()
                if not stripped:
                    # Blank line separates paragraphs
                    flush_paragraph_block()
                    continue

                # Numbered list items: each becomes its own chunk
                if numbered_pattern.match(stripped):
                    flush_paragraph_block()
                    # A numbered item may itself be long; respect token limits
                    item_text = stripped
                    # Split very long items using token-based splitting
                    if _token_count(item_text) > max_tokens:
                        words = item_text.split()
                        for i in range(0, len(words), max_tokens):
                            seg = " ".join(words[i : i + max_tokens]).strip()
                            if not is_valid_chunk(seg):
                                continue
                            metadata = dict(base_metadata)
                            if section_name:
                                metadata["section"] = section_name
                            metadata["chunk_index"] = chunk_index
                            metadata["chunk_id"] = f"{metadata.get('file_name')}_{chunk_index}"
                            chunk_index += 1
                            chunk_content = seg
                            if section_name:
                                chunk_content = f"Section: {section_name}\n\n{seg}"
                            chunk_doc = Document(
                                page_content=chunk_content,
                                metadata=metadata,
                            )
                            chunked_with_ids.append(chunk_doc)
                    else:
                        if is_valid_chunk(item_text):
                            metadata = dict(base_metadata)
                            if section_name:
                                metadata["section"] = section_name
                            metadata["chunk_index"] = chunk_index
                            metadata["chunk_id"] = f"{metadata.get('file_name')}_{chunk_index}"
                            chunk_index += 1
                            chunk_content = item_text
                            if section_name:
                                chunk_content = f"Section: {section_name}\n\n{item_text}"
                            chunk_doc = Document(
                                page_content=chunk_content,
                                metadata=metadata,
                            )
                            chunked_with_ids.append(chunk_doc)
                    continue

                # Regular narrative line: accumulate into paragraph block
                current_paragraph_lines.append(ln)

            # Flush any remaining paragraph text for this section
            flush_paragraph_block()

    return chunked_with_ids


def ingest_repository(
    config: IngestionConfig,
    state: Optional[IngestionState] = None,
) -> Mapping[str, List[Document]]:
    """
    Ingest all supported documents from a repository path.

    Returns a mapping:
        {
            "documents": [normalized, deduplicated Document objects],
            "chunks": [chunked Document objects ready for embedding],
        }

    This is designed as a reusable interface that can be called from:
    - Batch jobs (cron / scheduled tasks)
    - Streaming or event-driven triggers (e.g. on new file arrival)
    """
    logger.info("Starting ingestion for repository: %s", config.repository_path)

    repository_path = config.repository_path
    if not repository_path.exists() or not repository_path.is_dir():
        raise ValueError(f"Repository path does not exist or is not a directory: {repository_path}")

    if state is None:
        state = IngestionState.load(config.state_file)

    allowed_exts = {ext.lower().lstrip(".") for ext in config.allowed_extensions}

    all_docs: List[Document] = []
    processed_files = 0
    skipped_files = 0

    for path in repository_path.rglob("*"):
        if not path.is_file():
            continue
        ext = _extension(path)
        if ext not in allowed_exts:
            continue
        try:
            logger.info("Loading file: %s", path)
            raw_docs = _load_file(path, encoding=config.encoding)
        except Exception as exc:
            logger.exception("Failed to load file %s: %s", path, exc)
            skipped_files += 1
            continue

        # Normalize and clean text content before deduplication and chunking
        for doc in raw_docs:
            # Basic normalization and removal of repeated headers/footers
            cleaned = clean_text(doc.page_content)
            # Extract any leading email headers into metadata and remove them from body text
            headers, body_without_headers = extract_email_headers(cleaned)
            if headers:
                metadata = dict(doc.metadata or {})
                metadata.update(headers)
                doc.metadata = metadata
                doc.page_content = body_without_headers
            else:
                doc.page_content = cleaned

        all_docs.extend(raw_docs)
        processed_files += 1

    logger.info("Loaded %d documents from %d files", len(all_docs), processed_files)

    # Deduplicate based on content hash + path
    deduped_docs = _deduplicate_documents(all_docs, state=state)
    new_documents = len(deduped_docs)
    logger.info(
        "Deduplication complete. %d new/changed documents, %d skipped files",
        new_documents,
        skipped_files,
    )

    # Chunk documents for downstream embedding
    chunks = _chunk_documents(
        docs=deduped_docs,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )

    logger.info("Generated %d chunks from %d documents", len(chunks), new_documents)

    # Update and persist state for future incremental runs
    state.last_run_at = datetime.utcnow().isoformat() + "Z"
    state.save(config.state_file)

    return {
        "documents": deduped_docs,
        "chunks": chunks,
    }

