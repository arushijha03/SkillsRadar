"""LangGraph orchestrator — routes JD → RAG → Trend → Synthesis."""

import logging
import uuid
from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

from app.agents.jd_agent import JDAgent
from app.agents.rag_agent import RAGAgent
from app.agents.synthesis_agent import SynthesisAgent
from app.agents.trend_agent import TrendAgent
from app.models.schemas import AnalysisResponse, SkillSignal

logger = logging.getLogger(__name__)


class SkillRadarState(TypedDict):
    role: str
    company: Optional[str]
    domain: Optional[str]
    horizon: str
    run_id: str
    jd_result: Optional[dict]
    rag_result: Optional[dict]
    trend_result: Optional[dict]
    synthesis_result: Optional[dict]
    error: Optional[str]


def _jd_node(state: SkillRadarState) -> dict:
    agent = JDAgent()
    try:
        result = agent.run(
            role=state["role"],
            company=state.get("company"),
            domain=state.get("domain"),
        )
        return {"jd_result": result}
    except Exception as exc:
        logger.exception("JD agent failed")
        return {"jd_result": None, "error": f"JD agent: {exc}"}


def _rag_node(state: SkillRadarState) -> dict:
    agent = RAGAgent()
    try:
        result = agent.run(
            role=state["role"],
            domain=state.get("domain"),
        )
        return {"rag_result": result}
    except Exception as exc:
        logger.exception("RAG agent failed")
        return {"rag_result": None, "error": f"RAG agent: {exc}"}


def _trend_node(state: SkillRadarState) -> dict:
    agent = TrendAgent()
    try:
        result = agent.run(
            role=state["role"],
            domain=state.get("domain"),
            horizon=state.get("horizon", "6mo"),
        )
        return {"trend_result": result}
    except Exception as exc:
        logger.exception("Trend agent failed")
        return {"trend_result": None, "error": f"Trend agent: {exc}"}


def _synthesis_node(state: SkillRadarState) -> dict:
    agent = SynthesisAgent()
    try:
        result = agent.run(
            role=state["role"],
            horizon=state.get("horizon", "6mo"),
            jd_result=state.get("jd_result") or {"skills": []},
            rag_result=state.get("rag_result") or {"topics": []},
            trend_result=state.get("trend_result") or {"technologies": []},
        )
        return {"synthesis_result": result}
    except Exception as exc:
        logger.exception("Synthesis agent failed")
        return {"synthesis_result": None, "error": f"Synthesis agent: {exc}"}


def build_graph() -> StateGraph:
    """Build LangGraph state machine: jd → rag → trend → synthesis → END."""
    graph = StateGraph(SkillRadarState)

    graph.add_node("jd_agent", _jd_node)
    graph.add_node("rag_agent", _rag_node)
    graph.add_node("trend_agent", _trend_node)
    graph.add_node("synthesis_agent", _synthesis_node)

    graph.set_entry_point("jd_agent")
    graph.add_edge("jd_agent", "rag_agent")
    graph.add_edge("rag_agent", "trend_agent")
    graph.add_edge("trend_agent", "synthesis_agent")
    graph.add_edge("synthesis_agent", END)

    return graph


def run_analysis(
    role: str,
    company: str | None = None,
    domain: str | None = None,
    horizon: str = "6mo",
) -> AnalysisResponse:
    """Run full 4-agent pipeline and return API response."""
    run_id = str(uuid.uuid4())

    initial_state: SkillRadarState = {
        "role": role,
        "company": company,
        "domain": domain,
        "horizon": horizon,
        "run_id": run_id,
        "jd_result": None,
        "rag_result": None,
        "trend_result": None,
        "synthesis_result": None,
        "error": None,
    }

    workflow = build_graph()
    app = workflow.compile()
    final_state = app.invoke(initial_state)

    synthesis = final_state.get("synthesis_result") or {}
    signals_raw = synthesis.get("signals") or []
    report = synthesis.get("report_markdown") or "Analysis could not be completed."

    skills = [
        SkillSignal(
            skill=s["skill"],
            svs_score=s["svs_score"],
            category=s["category"],
            sources=s.get("sources", []),
        )
        for s in signals_raw
    ]

    return AnalysisResponse(
        role=role,
        skills=skills,
        report_markdown=report,
        run_id=run_id,
    )
