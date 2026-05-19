"""RSS feed ingestion for tech blogs."""

import logging
from typing import Any

import feedparser

logger = logging.getLogger(__name__)

DEFAULT_FEEDS = [
    "https://blog.langchain.dev/rss/",
    "https://openai.com/blog/rss.xml",
    "https://huggingface.co/blog/feed.xml",
    "https://www.pinecone.io/blog/rss.xml",
    "https://blog.replit.com/feed.xml",
    "https://simonwillison.net/atom/everything/",
    "https://martinfowler.com/feed.atom",
    "https://aws.amazon.com/blogs/machine-learning/feed/",
    "https://blog.google/technology/ai/rss/",
    "https://stackoverflow.blog/feed/",
]


class BlogIngester:
    """Fetch and parse articles from tech blog RSS feeds."""

    def __init__(self, feed_urls: list[str] | None = None):
        self.feed_urls = feed_urls or DEFAULT_FEEDS

    def fetch_all(self, max_entries_per_feed: int = 50) -> list[dict[str, Any]]:
        """Fetch articles from all configured feeds."""
        all_articles: list[dict[str, Any]] = []

        for url in self.feed_urls:
            try:
                articles = self.fetch_feed(url, max_entries=max_entries_per_feed)
                all_articles.extend(articles)
                logger.info("Fetched %d entries from %s", len(articles), url)
            except Exception as exc:
                logger.warning("Skipping feed %s: %s", url, exc)

        return all_articles

    def fetch_feed(
        self,
        feed_url: str,
        max_entries: int = 50,
    ) -> list[dict[str, Any]]:
        """Parse a single RSS/Atom feed."""
        parsed = feedparser.parse(feed_url)
        entries = parsed.entries or []

        articles: list[dict[str, Any]] = []
        for entry in entries[:max_entries]:
            content = ""
            if hasattr(entry, "content") and entry.content:
                content = entry.content[0].get("value", "")
            elif hasattr(entry, "summary"):
                content = entry.summary or ""
            elif hasattr(entry, "description"):
                content = entry.description or ""

            title = getattr(entry, "title", "") or ""
            link = getattr(entry, "link", "") or ""
            published = getattr(entry, "published", "") or ""

            if not content and not title:
                continue

            articles.append(
                {
                    "title": title,
                    "url": link,
                    "published": published,
                    "content": content,
                    "source": feed_url,
                }
            )

        return articles
