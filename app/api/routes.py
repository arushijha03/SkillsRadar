"""API v1 routes."""

import logging

from fastapi import APIRouter, HTTPException

from app.agents.jd_agent import JDAgent
from app.agents.orchestrator import run_analysis
from app.agents.rag_agent import RAGAgent
from app.agents.trend_agent import TrendAgent
from app.models.schemas import AnalysisRequest, AnalysisResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["analysis"])


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_skills(request: AnalysisRequest) -> AnalysisResponse:
    """
    Run full SkillRadar multi-agent pipeline.

    Orchestrates JD → RAG → Trend → Synthesis agents via LangGraph.
    """
    try:
        return run_analysis(
            role=request.role,
            company=request.company,
            domain=request.domain,
            horizon=request.horizon,
        )
    except Exception as exc:
        logger.exception("Analysis failed for role=%s", request.role)
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {exc}",
        ) from exc


@router.post("/jd-only")
async def jd_only(role: str, company: str | None = None, domain: str | None = None):
    """Day 1 endpoint — JD agent only (for testing)."""
    agent = JDAgent()
    try:
        return agent.run(role=role, company=company, domain=domain)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/rag-only")
async def rag_only(role: str, domain: str | None = None):
    """Day 2 endpoint — RAG agent only (queries Pinecone for community trends)."""
    agent = RAGAgent()
    try:
        return agent.run(role=role, domain=domain)
    except Exception as exc:
        logger.exception("RAG agent failed for role=%s", role)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/trend-only")
async def trend_only(role: str, domain: str | None = None, horizon: str = "6mo"):
    """Day 3 endpoint — Trend agent only (queries arXiv + Hacker News)."""
    agent = TrendAgent()
    try:
        return agent.run(role=role, domain=domain, horizon=horizon)
    except Exception as exc:
        logger.exception("Trend agent failed for role=%s", role)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
