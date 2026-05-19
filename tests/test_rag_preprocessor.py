"""Tests for RAG ingestion preprocessing (clean_text, chunk_text, iter_chunks)."""

from app.rag.ingestion.preprocessor import chunk_text, clean_text, iter_chunks


def test_clean_text_strips_html_and_whitespace():
    raw = "<p>Hello <b>world</b></p>\n\n   How are    you?  "
    cleaned = clean_text(raw)
    assert cleaned == "Hello world How are you?"


def test_clean_text_handles_empty():
    assert clean_text("") == ""
    assert clean_text(None) == ""  # type: ignore[arg-type]


def test_chunk_text_short_returns_single_chunk():
    text = "This is a short blog post."
    chunks = chunk_text(text, chunk_size=1000, chunk_overlap=200)
    assert chunks == [text]


def test_chunk_text_long_text_splits_with_overlap():
    text = "A" * 2500
    chunks = chunk_text(text, chunk_size=1000, chunk_overlap=200)

    assert len(chunks) >= 3
    assert all(len(c) <= 1000 for c in chunks)
    # Overlap means consecutive chunks share content
    assert chunks[0][-200:] == chunks[1][:200]


def test_chunk_text_empty_returns_empty():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_iter_chunks_yields_metadata():
    docs = [
        {
            "title": "LangGraph 101",
            "url": "https://example.com/langgraph",
            "source": "https://example.com/rss",
            "content": "LangGraph is a library for building stateful agents.",
        }
    ]
    chunks = list(iter_chunks(docs, chunk_size=1000, chunk_overlap=200))

    assert len(chunks) == 1
    chunk = chunks[0]
    assert "LangGraph 101" in chunk["text"]
    assert chunk["metadata"]["title"] == "LangGraph 101"
    assert chunk["metadata"]["url"] == "https://example.com/langgraph"
    assert chunk["metadata"]["source"] == "https://example.com/rss"
    assert chunk["metadata"]["chunk_index"] == 0


def test_iter_chunks_uses_summary_when_content_missing():
    docs = [
        {
            "title": "Fallback Doc",
            "summary": "Summary used because content is missing.",
        }
    ]
    chunks = list(iter_chunks(docs))
    assert len(chunks) == 1
    assert "Summary used" in chunks[0]["text"]


def test_iter_chunks_skips_empty_bodies():
    docs = [{"title": "Only title", "content": ""}]
    chunks = list(iter_chunks(docs))
    assert chunks == []


def test_iter_chunks_long_doc_produces_multiple_indices():
    docs = [
        {
            "title": "Long Post",
            "url": "https://example.com/long",
            "source": "feed",
            "content": "word " * 800,  # ~4000 chars
        }
    ]
    chunks = list(iter_chunks(docs, chunk_size=1000, chunk_overlap=200))
    assert len(chunks) >= 3
    indices = [c["metadata"]["chunk_index"] for c in chunks]
    assert indices == list(range(len(chunks)))
