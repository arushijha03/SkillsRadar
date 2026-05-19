"""Hacker News Algolia API client."""

from typing import Any

import requests


class HackerNewsService:
    """Fetch trending HN stories via Algolia search API."""

    BASE_URL = "https://hn.algolia.com/api/v1"

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def search_stories(
        self,
        query: str,
        tags: str = "story",
        hits_per_page: int = 20,
    ) -> list[dict[str, Any]]:
        """Search HN stories by keyword."""
        url = f"{self.BASE_URL}/search"
        params = {
            "query": query,
            "tags": tags,
            "hitsPerPage": hits_per_page,
        }

        response = requests.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()

        hits = payload.get("hits") or []
        stories = []
        for hit in hits:
            stories.append(
                {
                    "title": hit.get("title", ""),
                    "url": hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID', '')}",
                    "points": hit.get("points", 0),
                    "num_comments": hit.get("num_comments", 0),
                    "created_at": hit.get("created_at", ""),
                    "story_text": (hit.get("story_text") or "")[:500],
                }
            )
        return stories

    def format_for_llm(self, stories: list[dict[str, Any]]) -> str:
        """Format HN stories as context for the Trend agent."""
        if not stories:
            return "No Hacker News stories found for this query."

        lines = []
        for i, s in enumerate(stories[:15], 1):
            lines.append(
                f"{i}. [{s['points']} pts] {s['title']}\n"
                f"   URL: {s['url']}"
            )
        return "\n".join(lines)
