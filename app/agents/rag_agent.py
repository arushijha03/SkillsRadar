"""RAG Agent — queries Pinecone for community skill trends."""

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.rag.retriever import PineconeRetriever

logger = logging.getLogger(__name__)

RAG_SYSTEM_PROMPT = """You are a tech community analyst. Given retrieved blog/article chunks,
identify skills and topics trending in the developer community for the target role.

Return ONLY valid JSON array (no markdown):
[
  {
    "topic": "LangGraph",
    "momentum": 0.82,
    "community_score": 0.82
  }
]

Rules:
- momentum and community_score are 0.0 to 1.0 based on relevance and recency signals
- Include 8-15 topics
- community_score should reflect how prominently the topic appears in sources
"""


class RAGAgent:
    """Retrieve from Pinecone and synthesize community trends."""

    def __init__(
        self,
        llm: ChatOpenAI | None = None,
        retriever: PineconeRetriever | None = None,
    ):
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.retriever = retriever or PineconeRetriever()

    def run(
        self,
        role: str,
        domain: str | None = None,
    ) -> dict[str, Any]:
        """Execute RAG agent pipeline."""
        query = f"{role} skills technologies trends"
        if domain:
            query += f" {domain}"

        try:
            chunks = self.retriever.retrieve(query)
            context = self.retriever.format_for_llm(chunks)
        except Exception as exc:
            logger.error("Pinecone retrieval failed: %s", exc)
            return {
                "topics": [],
                "chunks_retrieved": 0,
                "error": str(exc),
            }

        if not chunks:
            return {
                "topics": [],
                "chunks_retrieved": 0,
                "error": "No chunks in Pinecone. Run scripts/ingest_blogs.py first.",
            }

        messages = [
            SystemMessage(content=RAG_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Target role: {role}\n"
                    f"Domain: {domain or 'general'}\n\n"
                    f"Retrieved community content:\n{context}"
                )
            ),
        ]

        response = self.llm.invoke(messages)
        topics = self._parse_topics(response.content)

        return {
            "topics": topics,
            "chunks_retrieved": len(chunks),
            "error": None,
        }

    def _parse_topics(self, content: str) -> list[dict]:
        text = content.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        try:
            topics = json.loads(text)
            if isinstance(topics, dict) and "topics" in topics:
                topics = topics["topics"]
        except json.JSONDecodeError:
            logger.warning("Failed to parse RAG agent JSON")
            return []

        return topics if isinstance(topics, list) else []
