"""arXiv API wrapper for recent research papers."""

from datetime import datetime, timedelta, timezone
from typing import Any

import arxiv


class ArxivService:
    """Search arXiv for papers related to a role/domain."""

    def __init__(self, max_results: int = 15):
        self.max_results = max_results

    def search_papers(
        self,
        query: str,
        days_back: int = 90,
    ) -> list[dict[str, Any]]:
        """
        Search arXiv for papers in the last N days.

        Uses arxiv.Search (v2.0+ API).
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)

        search = arxiv.Search(
            query=f"all:{query}",
            max_results=self.max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

        papers: list[dict[str, Any]] = []
        for result in search.results():
            published = result.published
            if published.tzinfo is None:
                published = published.replace(tzinfo=timezone.utc)
            if published < cutoff:
                continue

            papers.append(
                {
                    "title": result.title,
                    "summary": result.summary[:1500],
                    "published": published.isoformat(),
                    "categories": result.categories,
                    "url": result.entry_id,
                }
            )

        return papers

    def format_for_llm(self, papers: list[dict[str, Any]]) -> str:
        """Format paper list as context for the Trend agent."""
        if not papers:
            return "No recent arXiv papers found for this query."

        lines = []
        for i, p in enumerate(papers[:10], 1):
            lines.append(
                f"{i}. {p['title']}\n"
                f"   Categories: {', '.join(p['categories'][:3])}\n"
                f"   Summary: {p['summary'][:400]}..."
            )
        return "\n\n".join(lines)
