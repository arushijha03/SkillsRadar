"""Synthesis Agent — merges signals, computes SVS, generates report."""

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.models.schemas import SkillSignal
from app.utils.scoring import merge_skill_signals

logger = logging.getLogger(__name__)

SYNTHESIS_SYSTEM_PROMPT = """You are a career intelligence analyst writing a skill development report.

Given merged skill signals with SVS scores and source data, write a clear, actionable
markdown report for someone targeting this role.

Structure:
# SkillRadar Report: {role}
## Executive Summary
(2-3 sentences)

## Top Skills — In Demand Now (SVS ≥ 0.7)
(table or bullet list with skill, SVS, why it matters)

## Rising Skills — Next 6 Months (SVS 0.4–0.7)
(bullet list with learning recommendations)

## Emerging Skills — 12 Month Horizon (SVS < 0.4)
(bullet list with early-stage signals)

## Recommended Learning Path
(numbered priorities for the next 90 days)

## Data Sources
(brief note on JD count, community chunks, arXiv/HN signals)

Be specific, data-driven, and encouraging. Use markdown formatting.
"""


class SynthesisAgent:
    """Merge agent outputs, compute SVS, generate GPT-4o report."""

    def __init__(self, llm: ChatOpenAI | None = None):
        self.llm = llm or ChatOpenAI(model="gpt-4o", temperature=0.3)

    def run(
        self,
        role: str,
        horizon: str,
        jd_result: dict[str, Any],
        rag_result: dict[str, Any],
        trend_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute synthesis pipeline."""
        jd_skills = jd_result.get("skills") or []
        rag_topics = rag_result.get("topics") or []
        trend_techs = trend_result.get("technologies") or []

        merged = merge_skill_signals(jd_skills, rag_topics, trend_techs)
        signals = [
            SkillSignal(
                skill=m["skill"],
                svs_score=m["svs_score"],
                category=m["category"],
                sources=m["sources"],
            )
            for m in merged[:25]
        ]

        report = self._generate_report(
            role=role,
            horizon=horizon,
            signals=signals,
            jd_result=jd_result,
            rag_result=rag_result,
            trend_result=trend_result,
        )

        return {
            "signals": [s.model_dump() for s in signals],
            "report_markdown": report,
        }

    def _generate_report(
        self,
        role: str,
        horizon: str,
        signals: list[SkillSignal],
        jd_result: dict,
        rag_result: dict,
        trend_result: dict,
    ) -> str:
        signals_text = "\n".join(
            f"- {s.skill}: SVS={s.svs_score:.3f}, category={s.category}, "
            f"sources={', '.join(s.sources)}"
            for s in signals[:20]
        )

        context = (
            f"Role: {role}\n"
            f"Horizon: {horizon}\n"
            f"JD count: {jd_result.get('total_jds', 0)}\n"
            f"RAG chunks: {rag_result.get('chunks_retrieved', 0)}\n"
            f"arXiv papers: {trend_result.get('papers_found', 0)}\n"
            f"HN stories: {trend_result.get('stories_found', 0)}\n\n"
            f"Merged skill signals:\n{signals_text or 'No signals merged.'}"
        )

        messages = [
            SystemMessage(content=SYNTHESIS_SYSTEM_PROMPT),
            HumanMessage(content=context),
        ]

        try:
            response = self.llm.invoke(messages)
            return response.content or self._fallback_report(role, signals)
        except Exception as exc:
            logger.error("Synthesis LLM failed: %s", exc)
            return self._fallback_report(role, signals)

    def _fallback_report(self, role: str, signals: list[SkillSignal]) -> str:
        lines = [f"# SkillRadar Report: {role}", "", "## Top Skills", ""]
        for s in signals[:15]:
            lines.append(
                f"- **{s.skill}** — SVS: {s.svs_score:.3f} ({s.category})"
            )
        return "\n".join(lines)
