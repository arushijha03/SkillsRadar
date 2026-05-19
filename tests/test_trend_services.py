"""Tests for arXiv and Hacker News trend service adapters."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.services.arxiv_service import ArxivService
from app.services.hn_service import HackerNewsService


def _paper(title: str, days_old: int = 1):
    return SimpleNamespace(
        title=title,
        summary="A paper about retrieval augmented generation and agents." * 20,
        published=datetime.now(timezone.utc) - timedelta(days=days_old),
        categories=["cs.AI", "cs.CL"],
        entry_id=f"https://arxiv.org/abs/{title.lower().replace(' ', '-')}",
    )


@patch("app.services.arxiv_service.arxiv.Search")
def test_arxiv_search_filters_old_papers(mock_search_class):
    mock_search = MagicMock()
    mock_search.results.return_value = [
        _paper("Recent RAG Agents", days_old=2),
        _paper("Old Agent Paper", days_old=120),
    ]
    mock_search_class.return_value = mock_search

    service = ArxivService(max_results=5)
    papers = service.search_papers("RAG agents", days_back=90)

    assert len(papers) == 1
    assert papers[0]["title"] == "Recent RAG Agents"
    assert papers[0]["categories"] == ["cs.AI", "cs.CL"]
    mock_search_class.assert_called_once()


@patch("app.services.arxiv_service.arxiv.Search")
def test_arxiv_search_handles_naive_published_dates(mock_search_class):
    paper = _paper("Naive Date Paper")
    paper.published = datetime.now()  # noqa: DTZ005 - intentionally naive
    mock_search = MagicMock()
    mock_search.results.return_value = [paper]
    mock_search_class.return_value = mock_search

    papers = ArxivService().search_papers("agents")
    assert len(papers) == 1
    assert papers[0]["published"].endswith("+00:00")


def test_arxiv_format_for_llm_empty():
    assert "No recent arXiv papers" in ArxivService().format_for_llm([])


def test_arxiv_format_for_llm_includes_title_categories_and_summary():
    papers = [
        {
            "title": "RAG Agents",
            "summary": "Combining retrieval and agents.",
            "categories": ["cs.AI", "cs.CL"],
            "url": "https://arxiv.org/abs/1234",
        }
    ]

    formatted = ArxivService().format_for_llm(papers)
    assert "RAG Agents" in formatted
    assert "cs.AI" in formatted
    assert "Combining retrieval" in formatted


@patch("app.services.hn_service.requests.get")
def test_hn_search_stories_normalizes_hits(mock_get):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "hits": [
            {
                "title": "Show HN: Agent framework",
                "url": "https://example.com/agent",
                "points": 120,
                "num_comments": 42,
                "created_at": "2026-05-01T00:00:00Z",
                "story_text": "Some text",
                "objectID": "123",
            },
            {
                "title": "Ask HN: RAG in production",
                "url": None,
                "points": 50,
                "num_comments": 10,
                "created_at": "2026-05-02T00:00:00Z",
                "objectID": "456",
            },
        ]
    }
    mock_get.return_value = mock_response

    stories = HackerNewsService(timeout=5).search_stories("agents", hits_per_page=2)

    assert len(stories) == 2
    assert stories[0]["title"] == "Show HN: Agent framework"
    assert stories[0]["points"] == 120
    assert stories[1]["url"] == "https://news.ycombinator.com/item?id=456"
    mock_response.raise_for_status.assert_called_once()


def test_hn_format_for_llm_empty():
    assert "No Hacker News stories" in HackerNewsService().format_for_llm([])


def test_hn_format_for_llm_includes_points_title_and_url():
    stories = [
        {
            "title": "Show HN: RAG Tool",
            "url": "https://example.com/rag",
            "points": 99,
            "num_comments": 12,
            "created_at": "2026-05-01",
            "story_text": "",
        }
    ]

    formatted = HackerNewsService().format_for_llm(stories)
    assert "[99 pts] Show HN: RAG Tool" in formatted
    assert "https://example.com/rag" in formatted
