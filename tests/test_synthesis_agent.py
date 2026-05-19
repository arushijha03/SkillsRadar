"""Tests for SynthesisAgent SVS merge and report generation."""

from unittest.mock import MagicMock

from app.agents.synthesis_agent import SynthesisAgent


def _sample_inputs():
    jd_result = {
        "skills": [
            {
                "skill": "Python",
                "frequency": 8,
                "frequency_normalized": 0.8,
                "category": "programming_language",
            }
        ],
        "total_jds": 10,
        "error": None,
    }
    rag_result = {
        "topics": [
            {"topic": "Python", "momentum": 0.75, "community_score": 0.75},
            {"topic": "LangGraph", "momentum": 0.7, "community_score": 0.7},
        ],
        "chunks_retrieved": 8,
        "error": None,
    }
    trend_result = {
        "technologies": [
            {
                "technology": "LangGraph",
                "signal_strength": 0.85,
                "time_horizon": "6mo",
                "evidence": "Recent papers and HN stories",
            }
        ],
        "papers_found": 3,
        "stories_found": 5,
        "error": None,
    }
    return jd_result, rag_result, trend_result


def test_synthesis_agent_merges_signals_and_uses_llm_report():
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "# SkillRadar Report: ML Engineer\n\nReport body"
    mock_llm.invoke.return_value = mock_response

    jd_result, rag_result, trend_result = _sample_inputs()
    result = SynthesisAgent(llm=mock_llm).run(
        role="ML Engineer",
        horizon="6mo",
        jd_result=jd_result,
        rag_result=rag_result,
        trend_result=trend_result,
    )

    assert result["report_markdown"].startswith("# SkillRadar Report")
    assert len(result["signals"]) >= 2

    python = next(s for s in result["signals"] if s["skill"] == "Python")
    assert python["svs_score"] > 0.6
    assert "job_descriptions" in python["sources"]
    assert "community_rag" in python["sources"]

    langgraph = next(s for s in result["signals"] if s["skill"] == "LangGraph")
    assert "community_rag" in langgraph["sources"]
    assert "arxiv_hackernews" in langgraph["sources"]
    mock_llm.invoke.assert_called_once()


def test_synthesis_agent_falls_back_when_llm_fails():
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = RuntimeError("openai unavailable")

    jd_result, rag_result, trend_result = _sample_inputs()
    result = SynthesisAgent(llm=mock_llm).run(
        role="ML Engineer",
        horizon="6mo",
        jd_result=jd_result,
        rag_result=rag_result,
        trend_result=trend_result,
    )

    assert result["report_markdown"].startswith("# SkillRadar Report: ML Engineer")
    assert "Python" in result["report_markdown"]
    assert result["signals"]


def test_synthesis_agent_handles_empty_inputs():
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = ""
    mock_llm.invoke.return_value = mock_response

    result = SynthesisAgent(llm=mock_llm).run(
        role="Backend Engineer",
        horizon="12mo",
        jd_result={},
        rag_result={},
        trend_result={},
    )

    assert result["signals"] == []
    assert result["report_markdown"].startswith(
        "# SkillRadar Report: Backend Engineer"
    )
