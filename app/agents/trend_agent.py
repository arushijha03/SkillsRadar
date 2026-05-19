"""Trend Agent — arXiv + HackerNews emerging technology signals."""

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.services.arxiv_service import ArxivService
from app.services.hn_service import HackerNewsService

logger = logging.getLogger(__name__)

TREND_SYSTEM_PROMPT = """You are a technology foresight analyst. Given arXiv papers and Hacker News
discussions, identify emerging technologies and skills relevant to the target role.

Return ONLY valid JSON array (no markdown):
[
  {
    "technology": "Retrieval-Augmented Generation",
    "signal_strength": 0.75,
    "time_horizon": "6mo",
    "evidence": "Multiple arXiv papers and HN front-page discussions in last 90 days"
  }
]

Rules:
- signal_strength: 0.0 to 1.0
- time_horizon: "6mo" or "12mo"
- Include 8-12 technologies with brief evidence strings
"""


class TrendAgent:
    """Analyze arXiv and HN for emerging tech signals."""

    def __init__(
        self,
        llm: ChatOpenAI | None = None,
        arxiv_service: ArxivService | None = None,
        hn_service: HackerNewsService | None = None,
    ):
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.arxiv = arxiv_service or ArxivService()
        self.hn = hn_service or HackerNewsService()

    def run(
        self,
        role: str,
        domain: str | None = None,
        horizon: str = "6mo",
    ) -> dict[str, Any]:
        """Execute Trend agent pipeline."""
        search_query = role
        if domain:
            search_query = f"{role} {domain}"

        try:
            papers = self.arxiv.search_papers(search_query, days_back=90)
            arxiv_context = self.arxiv.format_for_llm(papers)
        except Exception as exc:
            logger.error("arXiv search failed: %s", exc)
            papers = []
            arxiv_context = f"arXiv unavailable: {exc}"

        try:
            stories = self.hn.search_stories(search_query)
            hn_context = self.hn.format_for_llm(stories)
        except Exception as exc:
            logger.error("HN search failed: %s", exc)
            stories = []
            hn_context = f"Hacker News unavailable: {exc}"

        messages = [
            SystemMessage(content=TREND_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Target role: {role}\n"
                    f"Domain: {domain or 'general'}\n"
                    f"Analysis horizon: {horizon}\n\n"
                    f"## arXiv (last 90 days, {len(papers)} papers)\n"
                    f"{arxiv_context}\n\n"
                    f"## Hacker News\n"
                    f"{hn_context}"
                )
            ),
        ]

        response = self.llm.invoke(messages)
        technologies = self._parse_technologies(response.content)

        return {
            "technologies": technologies,
            "papers_found": len(papers),
            "stories_found": len(stories),
            "error": None,
        }

    def _parse_technologies(self, content: str) -> list[dict]:
        text = content.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        try:
            techs = json.loads(text)
            if isinstance(techs, dict) and "technologies" in techs:
                techs = techs["technologies"]
        except json.JSONDecodeError:
            logger.warning("Failed to parse Trend agent JSON")
            return []

        return techs if isinstance(techs, list) else []
