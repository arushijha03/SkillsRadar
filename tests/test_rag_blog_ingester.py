"""Tests for BlogIngester RSS feed parsing (with mocked feedparser)."""

from types import SimpleNamespace
from unittest.mock import patch

from app.rag.ingestion.blog_ingester import DEFAULT_FEEDS, BlogIngester


def _make_entry(
    title: str = "Title",
    link: str = "https://example.com/post",
    published: str = "2026-01-01",
    summary: str | None = None,
    content_value: str | None = None,
):
    """Build a feedparser-like entry object."""
    attrs = {"title": title, "link": link, "published": published}
    if content_value is not None:
        attrs["content"] = [{"value": content_value}]
    if summary is not None:
        attrs["summary"] = summary
    return SimpleNamespace(**attrs)


@patch("app.rag.ingestion.blog_ingester.feedparser.parse")
def test_fetch_feed_extracts_content_field(mock_parse):
    mock_parse.return_value = SimpleNamespace(
        entries=[
            _make_entry(
                title="LangGraph 0.2 release",
                link="https://blog.langchain.dev/langgraph-02",
                content_value="<p>LangGraph now supports streaming.</p>",
            )
        ]
    )

    articles = BlogIngester().fetch_feed("https://blog.langchain.dev/rss/")

    assert len(articles) == 1
    article = articles[0]
    assert article["title"] == "LangGraph 0.2 release"
    assert article["url"] == "https://blog.langchain.dev/langgraph-02"
    assert "streaming" in article["content"]
    assert article["source"] == "https://blog.langchain.dev/rss/"


@patch("app.rag.ingestion.blog_ingester.feedparser.parse")
def test_fetch_feed_falls_back_to_summary(mock_parse):
    mock_parse.return_value = SimpleNamespace(
        entries=[_make_entry(title="Post", summary="Summary body")]
    )

    articles = BlogIngester().fetch_feed("feed-url")

    assert len(articles) == 1
    assert articles[0]["content"] == "Summary body"


@patch("app.rag.ingestion.blog_ingester.feedparser.parse")
def test_fetch_feed_respects_max_entries(mock_parse):
    mock_parse.return_value = SimpleNamespace(
        entries=[
            _make_entry(title=f"Post {i}", summary=f"body {i}") for i in range(10)
        ]
    )

    articles = BlogIngester().fetch_feed("feed", max_entries=3)
    assert len(articles) == 3


@patch("app.rag.ingestion.blog_ingester.feedparser.parse")
def test_fetch_feed_skips_entries_without_title_or_content(mock_parse):
    empty_entry = SimpleNamespace()  # no title, no content
    mock_parse.return_value = SimpleNamespace(
        entries=[empty_entry, _make_entry(title="Real post", summary="text")]
    )

    articles = BlogIngester().fetch_feed("feed")
    assert len(articles) == 1
    assert articles[0]["title"] == "Real post"


@patch("app.rag.ingestion.blog_ingester.feedparser.parse")
def test_fetch_all_aggregates_feeds(mock_parse):
    mock_parse.side_effect = [
        SimpleNamespace(entries=[_make_entry(title="A", summary="a")]),
        SimpleNamespace(entries=[_make_entry(title="B", summary="b")]),
    ]

    ingester = BlogIngester(feed_urls=["feed1", "feed2"])
    all_articles = ingester.fetch_all(max_entries_per_feed=10)

    assert len(all_articles) == 2
    assert {a["title"] for a in all_articles} == {"A", "B"}


@patch("app.rag.ingestion.blog_ingester.feedparser.parse")
def test_fetch_all_continues_when_feed_raises(mock_parse):
    mock_parse.side_effect = [
        Exception("network error"),
        SimpleNamespace(entries=[_make_entry(title="OK", summary="ok")]),
    ]

    ingester = BlogIngester(feed_urls=["broken-feed", "good-feed"])
    all_articles = ingester.fetch_all()

    assert len(all_articles) == 1
    assert all_articles[0]["title"] == "OK"


def test_default_feed_list_is_non_empty():
    assert len(DEFAULT_FEEDS) >= 5
    assert all(url.startswith("http") for url in DEFAULT_FEEDS)
