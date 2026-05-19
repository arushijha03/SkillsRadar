"""Tests for SVS scoring utilities."""

import pytest

from app.utils.scoring import categorize_svs, compute_svs, merge_skill_signals


def test_compute_svs():
    svs = compute_svs(0.8, 0.6, 0.4)
    assert svs == pytest.approx(0.5 * 0.8 + 0.3 * 0.6 + 0.2 * 0.4)


def test_categorize_svs_in_demand():
    assert categorize_svs(0.75) == "in_demand_now"
    assert categorize_svs(0.7) == "in_demand_now"


def test_categorize_svs_rising():
    assert categorize_svs(0.55) == "rising_6mo"
    assert categorize_svs(0.4) == "rising_6mo"


def test_categorize_svs_emerging():
    assert categorize_svs(0.39) == "emerging_12mo"
    assert categorize_svs(0.0) == "emerging_12mo"


def test_merge_skill_signals():
    jd = [
        {"skill": "Python", "frequency_normalized": 0.9, "category": "lang"},
    ]
    rag = [
        {"topic": "Python", "community_score": 0.7, "momentum": 0.7},
        {"topic": "LangGraph", "community_score": 0.6, "momentum": 0.6},
    ]
    trend = [
        {
            "technology": "LangGraph",
            "signal_strength": 0.8,
            "time_horizon": "6mo",
            "evidence": "HN trending",
        },
    ]

    merged = merge_skill_signals(jd, rag, trend)
    assert len(merged) >= 2

    python = next(m for m in merged if m["skill"] == "Python")
    assert python["svs_score"] > 0.5
    assert "job_descriptions" in python["sources"]
    assert "community_rag" in python["sources"]

    langgraph = next(m for m in merged if m["skill"] == "LangGraph")
    assert "community_rag" in langgraph["sources"]
    assert "arxiv_hackernews" in langgraph["sources"]
