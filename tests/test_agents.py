"""Agent unit tests with mocked external services."""

from unittest.mock import MagicMock, patch

import pytest

from app.agents.jd_agent import JDAgent
from app.utils.scoring import merge_skill_signals


@pytest.fixture
def mock_jd_descriptions():
    return [
        "We need Python, TensorFlow, and AWS experience.",
        "Required: Python, PyTorch, Docker, Kubernetes.",
    ]


@patch("app.agents.jd_agent.JSearchService")
def test_jd_agent_parses_skills(mock_jsearch_class, mock_jd_descriptions):
    mock_service = MagicMock()
    mock_service.fetch_job_descriptions.return_value = mock_jd_descriptions
    mock_jsearch_class.return_value = mock_service

    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = """[
        {"skill": "Python", "frequency": 2, "frequency_normalized": 1.0, "category": "language"},
        {"skill": "TensorFlow", "frequency": 1, "frequency_normalized": 0.5, "category": "ml"}
    ]"""
    mock_llm.invoke.return_value = mock_response

    agent = JDAgent(llm=mock_llm)
    agent.jsearch = mock_service
    result = agent.run(role="ML Engineer")

    assert result["total_jds"] == 2
    assert len(result["skills"]) == 2
    assert result["skills"][0]["skill"] == "Python"


def test_merge_empty_inputs():
    merged = merge_skill_signals([], [], [])
    assert merged == []
