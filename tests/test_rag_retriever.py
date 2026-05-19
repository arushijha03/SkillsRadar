"""Tests for PineconeRetriever with mocked Pinecone + embeddings."""

from unittest.mock import MagicMock

import pytest

from app.rag.retriever import PineconeRetriever


def _build_retriever_with_matches(matches: list[dict]) -> PineconeRetriever:
    """Construct a retriever wired to mocked Pinecone + embeddings."""
    retriever = PineconeRetriever(api_key="test-key", index_name="skillradar-test")

    mock_embeddings = MagicMock()
    mock_embeddings.embed_query.return_value = [0.0] * 1536
    retriever.embeddings = mock_embeddings

    mock_index = MagicMock()
    mock_index.query.return_value = {"matches": matches}

    mock_client = MagicMock()
    mock_client.Index.return_value = mock_index
    retriever._client = mock_client

    return retriever


def test_retrieve_returns_normalized_chunks():
    matches = [
        {
            "id": "v1",
            "score": 0.91,
            "metadata": {
                "title": "LangGraph guide",
                "url": "https://blog.langchain.dev/lg",
                "source": "https://blog.langchain.dev/rss/",
                "text": "LangGraph is a stateful agent framework.",
            },
        },
        {
            "id": "v2",
            "score": 0.74,
            "metadata": {
                "title": "Pinecone tips",
                "url": "https://pinecone.io/tips",
                "source": "https://pinecone.io/blog/rss.xml",
                "text": "Use serverless for low traffic indexes.",
            },
        },
    ]
    retriever = _build_retriever_with_matches(matches)

    results = retriever.retrieve("LangGraph agent")
    assert len(results) == 2
    assert results[0]["title"] == "LangGraph guide"
    assert results[0]["score"] == 0.91
    assert results[0]["text"].startswith("LangGraph")
    assert results[1]["url"] == "https://pinecone.io/tips"


def test_retrieve_respects_top_k_override():
    retriever = _build_retriever_with_matches([])
    retriever.retrieve("query", top_k=3)

    query_kwargs = retriever._client.Index.return_value.query.call_args.kwargs
    assert query_kwargs["top_k"] == 3
    assert query_kwargs["include_metadata"] is True


def test_retrieve_uses_default_top_k():
    retriever = _build_retriever_with_matches([])
    retriever.retrieve("query")

    query_kwargs = retriever._client.Index.return_value.query.call_args.kwargs
    assert query_kwargs["top_k"] == retriever.top_k  # default 8


def test_retrieve_handles_missing_matches_key():
    retriever = PineconeRetriever(api_key="test-key")
    retriever.embeddings = MagicMock(embed_query=MagicMock(return_value=[0.0] * 1536))

    mock_index = MagicMock()
    mock_index.query.return_value = {}  # no "matches"

    mock_client = MagicMock()
    mock_client.Index.return_value = mock_index
    retriever._client = mock_client

    assert retriever.retrieve("anything") == []


def test_retrieve_handles_missing_metadata():
    retriever = _build_retriever_with_matches(
        [{"id": "v1", "score": 0.5}]  # no metadata
    )
    results = retriever.retrieve("q")
    assert results == [
        {"text": "", "title": "", "url": "", "source": "", "score": 0.5}
    ]


def test_client_requires_api_key(monkeypatch):
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    retriever = PineconeRetriever(api_key="")
    with pytest.raises(ValueError, match="PINECONE_API_KEY"):
        _ = retriever.client


def test_format_for_llm_empty_chunks():
    retriever = PineconeRetriever(api_key="test-key")
    out = retriever.format_for_llm([])
    assert "No relevant community content" in out


def test_format_for_llm_includes_scores_and_urls():
    chunks = [
        {
            "title": "LangGraph guide",
            "url": "https://example.com/lg",
            "source": "feed",
            "text": "LangGraph notes...",
            "score": 0.91,
        },
        {
            "title": "Pinecone tips",
            "url": "https://example.com/pc",
            "source": "feed",
            "text": "Pinecone notes...",
            "score": 0.74,
        },
    ]
    retriever = PineconeRetriever(api_key="test-key")
    out = retriever.format_for_llm(chunks)

    assert "[1]" in out and "[2]" in out
    assert "0.910" in out and "0.740" in out
    assert "https://example.com/lg" in out
    assert "https://example.com/pc" in out
    assert "---" in out  # separator between chunks
