"""JD Agent — extracts skills from live job descriptions."""

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.services.jsearch_service import JSearchService

logger = logging.getLogger(__name__)

JD_SYSTEM_PROMPT = """You are a job market analyst. Given job descriptions for a target role,
extract tools, technologies, frameworks, and skills mentioned.

Return ONLY valid JSON array with this schema (no markdown):
[
  {
    "skill": "Python",
    "frequency": 12,
    "frequency_normalized": 0.85,
    "category": "programming_language"
  }
]

Rules:
- frequency = count of JDs mentioning this skill
- frequency_normalized = frequency / total_jds (0.0 to 1.0)
- Include at least 10 skills if data allows
- Normalize skill names (e.g. "K8s" -> "Kubernetes")
"""


class JDAgent:
    """Fetch JDs via JSearch and extract skills with GPT-4o-mini."""

    def __init__(self, llm: ChatOpenAI | None = None):
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.jsearch = JSearchService()

    def run(
        self,
        role: str,
        company: str | None = None,
        domain: str | None = None,
    ) -> dict[str, Any]:
        """Execute JD agent pipeline."""
        query_role = role
        if domain:
            query_role = f"{role} {domain}"

        try:
            descriptions = self.jsearch.fetch_job_descriptions(
                role=query_role,
                company=company,
                limit=20,
            )
        except Exception as exc:
            logger.error("JSearch failed: %s", exc)
            return {
                "skills": [],
                "total_jds": 0,
                "error": str(exc),
            }

        if not descriptions:
            return {
                "skills": [],
                "total_jds": 0,
                "error": "No job descriptions returned. Check RapidAPI JSearch subscription.",
            }

        combined = "\n\n---\n\n".join(
            f"JD {i+1}:\n{d[:3000]}" for i, d in enumerate(descriptions[:20])
        )

        messages = [
            SystemMessage(content=JD_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Role: {role}\n"
                    f"Total JDs: {len(descriptions)}\n\n"
                    f"Job descriptions:\n{combined}"
                )
            ),
        ]

        response = self.llm.invoke(messages)
        skills = self._parse_skills(response.content, len(descriptions))

        return {
            "skills": skills,
            "total_jds": len(descriptions),
            "error": None,
        }

    def _parse_skills(self, content: str, total_jds: int) -> list[dict]:
        """Parse LLM JSON response."""
        text = content.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        try:
            skills = json.loads(text)
            if isinstance(skills, dict) and "skills" in skills:
                skills = skills["skills"]
        except json.JSONDecodeError:
            logger.warning("Failed to parse JD agent JSON: %s", content[:200])
            return []

        if not isinstance(skills, list):
            return []

        for skill in skills:
            freq = skill.get("frequency", 1)
            if "frequency_normalized" not in skill and total_jds > 0:
                skill["frequency_normalized"] = min(freq / total_jds, 1.0)

        return skills
