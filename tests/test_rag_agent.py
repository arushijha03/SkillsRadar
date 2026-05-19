"""Tests for RAGAgent with mocked retriever and LLM."""

import json
from unittest.mock import MagicMock

from app.agents.rag_agent import RAGAgent


def _mock_llm_response(payload):
    """Build a mocked LLM that returns ``payload`` (string or list) as ``.content``."""
    if not isinstance(payload, str):
        payload = json.dumps(payload)
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = payload
    mock_llm.invoke.return_value = mock_response
    return mock_llm


def test_rag_agent_returns_topics_on_happy_path():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        {"text": "LangGraph notes", "title": "lg", "url": "u", "source": "s", "score": 0.9}
    ]
    mock_retriever.format_for_llm.return_value = "[1] (score=0.900) lg\nSource: u\nLangGraph notes"

    topics_json = [
        {"topic": "LangGraph", "momentum": 0.82, "community_score": 0.82},
        {"topic": "Pinecone", "momentum": 0.71, "community_score": 0.71},
    ]
    agent = RAGAgent(llm=_mock_llm_response(topics_json), retriever=mock_retriever)

    result = agent.run(role="ML Engineer", domain="NLP")

    assert result["error"] is None
    assert result["chunks_retrieved"] == 1
    assert len(result["topics"]) == 2
    assert result["topics"][0]["topic"] == "LangGraph"

    # Query includes role and domain
    call_args = mock_retriever.retrieve.call_args
    assert "ML Engineer" in call_args.args[0]
    assert "NLP" in call_args.args[0]


def test_rag_agent_handles_empty_chunks():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []
    mock_retriever.format_for_llm.return_value = "No relevant community content found in the vector store."

    agent = RAGAgent(llm=_mock_llm_response([]), retriever=mock_retriever)
    result = agent.run(role="Data Scientist")

    assert result["topics"] == []
    assert result["chunks_retrieved"] == 0
    assert "ingest_blogs" in (result["error"] or "")


def test_rag_agent_handles_retriever_exception():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.side_effect = RuntimeError("pinecone unreachable")

    agent = RAGAgent(llm=_mock_llm_response([]), retriever=mock_retriever)
    result = agent.run(role="ML Engineer")

    assert result["topics"] == []
    assert result["chunks_retrieved"] == 0
    assert "pinecone unreachable" in (result["error"] or "")


def test_rag_agent_strips_markdown_fences():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        {"text": "x", "title": "t", "url": "u", "source": "s", "score": 0.5}
    ]
    mock_retriever.format_for_llm.return_value = "ctx"

    fenced = '```json\n[{"topic": "LangGraph", "momentum": 0.8, "community_score": 0.8}]\n```'
    agent = RAGAgent(llm=_mock_llm_response(fenced), retriever=mock_retriever)

    result = agent.run(role="ML Engineer")
    assert len(result["topics"]) == 1
    assert result["topics"][0]["topic"] == "LangGraph"


def test_rag_agent_handles_topics_wrapped_in_object():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        {"text": "x", "title": "t", "url": "u", "source": "s", "score": 0.5}
    ]
    mock_retriever.format_for_llm.return_value = "ctx"

    wrapped = {"topics": [{"topic": "RAG", "momentum": 0.6, "community_score": 0.6}]}
    agent = RAGAgent(llm=_mock_llm_response(wrapped), retriever=mock_retriever)

    result = agent.run(role="ML Engineer")
    assert len(result["topics"]) == 1
    assert result["topics"][0]["topic"] == "RAG"


def test_rag_agent_returns_empty_on_invalid_json():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        {"text": "x", "title": "t", "url": "u", "source": "s", "score": 0.5}
    ]
    mock_retriever.format_for_llm.return_value = "ctx"

    agent = RAGAgent(llm=_mock_llm_response("not json at all"), retriever=mock_retriever)
    result = agent.run(role="ML Engineer")

    assert result["topics"] == []
    assert result["chunks_retrieved"] == 1


def test_rag_agent_query_omits_domain_when_none():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []
    mock_retriever.format_for_llm.return_value = ""

    agent = RAGAgent(llm=_mock_llm_response([]), retriever=mock_retriever)
    agent.run(role="Backend Engineer")

    query = mock_retriever.retrieve.call_args.args[0]
    assert "Backend Engineer" in query
    assert "None" not in query
