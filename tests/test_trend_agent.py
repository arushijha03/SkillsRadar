"""Tests for TrendAgent with mocked arXiv, Hacker News, and LLM calls."""

import json
from unittest.mock import MagicMock

from app.agents.trend_agent import TrendAgent


def _mock_llm_response(payload):
    if not isinstance(payload, str):
        payload = json.dumps(payload)
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = payload
    mock_llm.invoke.return_value = mock_response
    return mock_llm


def _mock_services():
    arxiv = MagicMock()
    arxiv.search_papers.return_value = [
        {
            "title": "Agentic RAG",
            "summary": "Research summary",
            "published": "2026-05-01T00:00:00+00:00",
            "categories": ["cs.AI"],
            "url": "https://arxiv.org/abs/1",
        }
    ]
    arxiv.format_for_llm.return_value = "1. Agentic RAG"

    hn = MagicMock()
    hn.search_stories.return_value = [
        {
            "title": "Show HN: Agentic RAG",
            "url": "https://example.com",
            "points": 100,
            "num_comments": 20,
            "created_at": "2026-05-01",
            "story_text": "",
        }
    ]
    hn.format_for_llm.return_value = "1. [100 pts] Show HN: Agentic RAG"
    return arxiv, hn


def test_trend_agent_returns_technologies_on_happy_path():
    arxiv, hn = _mock_services()
    llm = _mock_llm_response(
        [
            {
                "technology": "Agentic RAG",
                "signal_strength": 0.84,
                "time_horizon": "6mo",
                "evidence": "Recent papers and HN discussions",
            }
        ]
    )

    agent = TrendAgent(llm=llm, arxiv_service=arxiv, hn_service=hn)
    result = agent.run(role="ML Engineer", domain="NLP", horizon="6mo")

    assert result["error"] is None
    assert result["papers_found"] == 1
    assert result["stories_found"] == 1
    assert result["technologies"][0]["technology"] == "Agentic RAG"
    arxiv.search_papers.assert_called_once_with("ML Engineer NLP", days_back=90)
    hn.search_stories.assert_called_once_with("ML Engineer NLP")


def test_trend_agent_omits_domain_when_none():
    arxiv, hn = _mock_services()
    agent = TrendAgent(
        llm=_mock_llm_response([]),
        arxiv_service=arxiv,
        hn_service=hn,
    )

    agent.run(role="Backend Engineer")

    arxiv.search_papers.assert_called_once_with("Backend Engineer", days_back=90)
    hn.search_stories.assert_called_once_with("Backend Engineer")


def test_trend_agent_continues_when_arxiv_fails():
    arxiv, hn = _mock_services()
    arxiv.search_papers.side_effect = RuntimeError("arxiv unavailable")
    llm = _mock_llm_response(
        [
            {
                "technology": "Vector Databases",
                "signal_strength": 0.72,
                "time_horizon": "6mo",
                "evidence": "HN discussions",
            }
        ]
    )

    result = TrendAgent(llm=llm, arxiv_service=arxiv, hn_service=hn).run(
        role="ML Engineer"
    )

    assert result["papers_found"] == 0
    assert result["stories_found"] == 1
    assert result["technologies"][0]["technology"] == "Vector Databases"


def test_trend_agent_continues_when_hn_fails():
    arxiv, hn = _mock_services()
    hn.search_stories.side_effect = RuntimeError("hn unavailable")

    result = TrendAgent(
        llm=_mock_llm_response([]),
        arxiv_service=arxiv,
        hn_service=hn,
    ).run(role="ML Engineer")

    assert result["papers_found"] == 1
    assert result["stories_found"] == 0
    assert result["technologies"] == []


def test_trend_agent_strips_markdown_fences():
    arxiv, hn = _mock_services()
    fenced = """```json
[
  {
    "technology": "Small Language Models",
    "signal_strength": 0.66,
    "time_horizon": "12mo",
    "evidence": "Papers"
  }
]
```"""

    result = TrendAgent(
        llm=_mock_llm_response(fenced),
        arxiv_service=arxiv,
        hn_service=hn,
    ).run(role="ML Engineer")

    assert result["technologies"][0]["technology"] == "Small Language Models"


def test_trend_agent_handles_wrapped_technologies_object():
    arxiv, hn = _mock_services()
    wrapped = {
        "technologies": [
            {
                "technology": "LLM Observability",
                "signal_strength": 0.61,
                "time_horizon": "6mo",
                "evidence": "Tooling discussions",
            }
        ]
    }

    result = TrendAgent(
        llm=_mock_llm_response(wrapped),
        arxiv_service=arxiv,
        hn_service=hn,
    ).run(role="ML Engineer")

    assert result["technologies"][0]["technology"] == "LLM Observability"


def test_trend_agent_returns_empty_on_invalid_json():
    arxiv, hn = _mock_services()
    result = TrendAgent(
        llm=_mock_llm_response("not json"),
        arxiv_service=arxiv,
        hn_service=hn,
    ).run(role="ML Engineer")

    assert result["technologies"] == []
