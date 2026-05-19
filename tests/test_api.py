"""FastAPI endpoint tests."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.schemas import AnalysisResponse, SkillSignal

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "SkillRadar" in response.json()["message"]


@patch("app.api.routes.run_analysis")
def test_analyze_endpoint(mock_run):
    mock_run.return_value = AnalysisResponse(
        role="Data Scientist",
        skills=[
            SkillSignal(
                skill="Python",
                svs_score=0.85,
                category="in_demand_now",
                sources=["job_descriptions"],
            )
        ],
        report_markdown="# Test Report",
        run_id="test-run-id",
    )

    response = client.post(
        "/api/v1/analyze",
        json={"role": "Data Scientist", "horizon": "6mo"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["role"] == "Data Scientist"
    assert len(data["skills"]) == 1
    assert data["skills"][0]["skill"] == "Python"


@patch("app.api.routes.RAGAgent")
def test_rag_only_endpoint(mock_rag_class):
    mock_agent = mock_rag_class.return_value
    mock_agent.run.return_value = {
        "topics": [
            {"topic": "LangGraph", "momentum": 0.82, "community_score": 0.82}
        ],
        "chunks_retrieved": 8,
        "error": None,
    }

    response = client.post("/api/v1/rag-only?role=ML%20Engineer&domain=NLP")
    assert response.status_code == 200

    data = response.json()
    assert data["chunks_retrieved"] == 8
    assert data["topics"][0]["topic"] == "LangGraph"
    mock_agent.run.assert_called_once_with(role="ML Engineer", domain="NLP")


@patch("app.api.routes.RAGAgent")
def test_rag_only_endpoint_propagates_failure(mock_rag_class):
    mock_rag_class.return_value.run.side_effect = RuntimeError("pinecone down")

    response = client.post("/api/v1/rag-only?role=Backend%20Engineer")
    assert response.status_code == 500
    assert "pinecone down" in response.json()["detail"]


@patch("app.api.routes.TrendAgent")
def test_trend_only_endpoint(mock_trend_class):
    mock_agent = mock_trend_class.return_value
    mock_agent.run.return_value = {
        "technologies": [
            {
                "technology": "Agentic RAG",
                "signal_strength": 0.84,
                "time_horizon": "6mo",
                "evidence": "Recent papers and HN discussions",
            }
        ],
        "papers_found": 4,
        "stories_found": 6,
        "error": None,
    }

    response = client.post(
        "/api/v1/trend-only?role=ML%20Engineer&domain=NLP&horizon=12mo"
    )
    assert response.status_code == 200

    data = response.json()
    assert data["papers_found"] == 4
    assert data["stories_found"] == 6
    assert data["technologies"][0]["technology"] == "Agentic RAG"
    mock_agent.run.assert_called_once_with(
        role="ML Engineer", domain="NLP", horizon="12mo"
    )


@patch("app.api.routes.TrendAgent")
def test_trend_only_endpoint_propagates_failure(mock_trend_class):
    mock_trend_class.return_value.run.side_effect = RuntimeError("trend down")

    response = client.post("/api/v1/trend-only?role=Backend%20Engineer")
    assert response.status_code == 500
    assert "trend down" in response.json()["detail"]
