"""Tests for LangGraph orchestrator wiring with mocked agents."""

from unittest.mock import MagicMock, patch

from app.agents.orchestrator import build_graph, run_analysis


def test_build_graph_compiles():
    graph = build_graph()
    compiled = graph.compile()
    assert compiled is not None


@patch("app.agents.orchestrator.SynthesisAgent")
@patch("app.agents.orchestrator.TrendAgent")
@patch("app.agents.orchestrator.RAGAgent")
@patch("app.agents.orchestrator.JDAgent")
def test_run_analysis_invokes_agents_in_order(
    mock_jd_class,
    mock_rag_class,
    mock_trend_class,
    mock_synthesis_class,
):
    jd_result = {
        "skills": [
            {"skill": "Python", "frequency_normalized": 0.9, "category": "language"}
        ],
        "total_jds": 10,
        "error": None,
    }
    rag_result = {
        "topics": [{"topic": "Python", "community_score": 0.7, "momentum": 0.7}],
        "chunks_retrieved": 8,
        "error": None,
    }
    trend_result = {
        "technologies": [
            {
                "technology": "Python",
                "signal_strength": 0.5,
                "time_horizon": "6mo",
                "evidence": "HN and arXiv",
            }
        ],
        "papers_found": 2,
        "stories_found": 3,
        "error": None,
    }
    synthesis_result = {
        "signals": [
            {
                "skill": "Python",
                "svs_score": 0.71,
                "category": "in_demand_now",
                "sources": ["job_descriptions", "community_rag", "arxiv_hackernews"],
            }
        ],
        "report_markdown": "# SkillRadar Report: ML Engineer",
    }

    mock_jd_class.return_value.run.return_value = jd_result
    mock_rag_class.return_value.run.return_value = rag_result
    mock_trend_class.return_value.run.return_value = trend_result
    mock_synthesis_class.return_value.run.return_value = synthesis_result

    response = run_analysis(
        role="ML Engineer",
        company="Acme",
        domain="NLP",
        horizon="6mo",
    )

    assert response.role == "ML Engineer"
    assert response.report_markdown == "# SkillRadar Report: ML Engineer"
    assert response.skills[0].skill == "Python"
    assert response.skills[0].svs_score == 0.71
    assert response.run_id

    mock_jd_class.return_value.run.assert_called_once_with(
        role="ML Engineer", company="Acme", domain="NLP"
    )
    mock_rag_class.return_value.run.assert_called_once_with(
        role="ML Engineer", domain="NLP"
    )
    mock_trend_class.return_value.run.assert_called_once_with(
        role="ML Engineer", domain="NLP", horizon="6mo"
    )
    mock_synthesis_class.return_value.run.assert_called_once_with(
        role="ML Engineer",
        horizon="6mo",
        jd_result=jd_result,
        rag_result=rag_result,
        trend_result=trend_result,
    )


@patch("app.agents.orchestrator.SynthesisAgent")
@patch("app.agents.orchestrator.TrendAgent")
@patch("app.agents.orchestrator.RAGAgent")
@patch("app.agents.orchestrator.JDAgent")
def test_run_analysis_continues_when_upstream_agent_fails(
    mock_jd_class,
    mock_rag_class,
    mock_trend_class,
    mock_synthesis_class,
):
    mock_jd_class.return_value.run.side_effect = RuntimeError("jsearch down")
    mock_rag_class.return_value.run.return_value = {
        "topics": [{"topic": "LangGraph", "community_score": 0.7, "momentum": 0.7}],
        "chunks_retrieved": 4,
        "error": None,
    }
    mock_trend_class.return_value.run.return_value = {
        "technologies": [],
        "papers_found": 0,
        "stories_found": 0,
        "error": None,
    }
    mock_synthesis_class.return_value.run.return_value = {
        "signals": [
            {
                "skill": "LangGraph",
                "svs_score": 0.41,
                "category": "rising_6mo",
                "sources": ["community_rag"],
            }
        ],
        "report_markdown": "# Partial Report",
    }

    response = run_analysis(role="ML Engineer")

    assert response.skills[0].skill == "LangGraph"
    assert response.report_markdown == "# Partial Report"

    synthesis_kwargs = mock_synthesis_class.return_value.run.call_args.kwargs
    assert synthesis_kwargs["jd_result"] == {"skills": []}
    assert synthesis_kwargs["rag_result"]["chunks_retrieved"] == 4


def test_run_analysis_returns_empty_report_when_synthesis_missing():
    with patch("app.agents.orchestrator.JDAgent") as mock_jd_class, patch(
        "app.agents.orchestrator.RAGAgent"
    ) as mock_rag_class, patch("app.agents.orchestrator.TrendAgent") as mock_trend_class, patch(
        "app.agents.orchestrator.SynthesisAgent"
    ) as mock_synthesis_class:
        mock_jd_class.return_value.run.return_value = {"skills": []}
        mock_rag_class.return_value.run.return_value = {"topics": []}
        mock_trend_class.return_value.run.return_value = {"technologies": []}
        mock_synthesis_class.return_value.run.side_effect = RuntimeError("llm down")

        response = run_analysis(role="ML Engineer")

    assert response.role == "ML Engineer"
    assert response.skills == []
    assert response.report_markdown == "Analysis could not be completed."
