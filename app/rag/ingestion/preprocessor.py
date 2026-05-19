"""Text cleaning and chunking for RAG ingestion."""

import re
from typing import Iterator


def clean_text(text: str) -> str:
    """Normalize whitespace and strip HTML-like artifacts."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[str]:
    """Split text into overlapping chunks by character count."""
    text = clean_text(text)
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - chunk_overlap
        if start < 0:
            start = 0
        if end >= len(text):
            break

    return chunks


def iter_chunks(
    documents: list[dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> Iterator[dict]:
    """Yield chunk dicts with metadata from source documents."""
    for doc in documents:
        body = doc.get("content") or doc.get("summary") or ""
        title = doc.get("title", "")
        url = doc.get("url", "")
        source = doc.get("source", "unknown")

        for i, chunk in enumerate(
            chunk_text(body, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        ):
            yield {
                "text": f"{title}\n\n{chunk}" if title else chunk,
                "metadata": {
                    "title": title,
                    "url": url,
                    "source": source,
                    "chunk_index": i,
                },
            }
