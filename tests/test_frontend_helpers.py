"""Tests for Streamlit UI helper functions."""

from frontend.ui_helpers import (
    CATEGORY_BLURBS,
    CATEGORY_LABELS,
    KIND_BLURBS,
    KIND_LABELS,
    build_analysis_payload,
    classify_skill_kind,
    group_skills_by_category,
    group_skills_by_kind,
    normalize_api_url,
    pick_differentiator_skill,
    skills_to_rows,
    to_demand_score,
)


def test_build_analysis_payload_strips_and_normalizes_optional_fields():
    payload = build_analysis_payload(
        role="  Machine Learning Engineer  ",
        company="   ",
        domain=" NLP ",
        horizon="6mo",
    )

    assert payload == {
        "role": "Machine Learning Engineer",
        "company": None,
        "domain": "NLP",
        "horizon": "6mo",
    }


def test_normalize_api_url_strips_trailing_slash_and_space():
    assert normalize_api_url(" https://api.example.com/ ") == "https://api.example.com"


def test_to_demand_score_scales_and_clamps():
    assert to_demand_score(0) == 0
    assert to_demand_score(0.5) == 50
    assert to_demand_score(0.723) == 72
    assert to_demand_score(1.0) == 100
    assert to_demand_score(1.7) == 100
    assert to_demand_score(-0.3) == 0


def test_category_labels_and_blurbs_share_keys():
    assert set(CATEGORY_LABELS.keys()) == set(CATEGORY_BLURBS.keys())
    assert "Hot" in CATEGORY_LABELS["in_demand_now"]
    assert "Heating" in CATEGORY_LABELS["rising_6mo"]
    assert "Horizon" in CATEGORY_LABELS["emerging_12mo"]


def test_skills_to_rows_uses_0_to_100_demand_score_and_type():
    rows = skills_to_rows(
        [
            {
                "skill": "LangGraph",
                "svs_score": 0.72345,
                "category": "rising_6mo",
                "sources": ["community_rag", "arxiv_hackernews"],
            },
            {
                "skill": "Stakeholder Communication",
                "svs_score": 0.55,
                "category": "in_demand_now",
                "sources": ["job_descriptions"],
            },
        ]
    )

    assert rows == [
        {"Skill": "LangGraph", "Demand Score": 72, "Type": "Technical"},
        {
            "Skill": "Stakeholder Communication",
            "Demand Score": 55,
            "Type": "Non-technical",
        },
    ]


def test_classify_skill_kind_basics():
    assert classify_skill_kind("Python") == "technical"
    assert classify_skill_kind("Docker") == "technical"
    assert classify_skill_kind("Communication") == "non_technical"
    assert classify_skill_kind("Stakeholder management") == "non_technical"
    assert classify_skill_kind("Cross-functional collaboration") == "non_technical"
    assert classify_skill_kind("Project Management") == "non_technical"
    assert classify_skill_kind("Agile") == "non_technical"
    assert classify_skill_kind("") == "technical"


def test_group_skills_by_kind_sorts_within_each_bucket():
    skills = [
        {"skill": "Python", "svs_score": 0.95},
        {"skill": "Communication", "svs_score": 0.40},
        {"skill": "Docker", "svs_score": 0.70},
        {"skill": "Leadership", "svs_score": 0.55},
    ]
    grouped = group_skills_by_kind(skills)

    assert list(grouped.keys()) == list(KIND_LABELS.keys())
    assert [s["skill"] for s in grouped["technical"]] == ["Python", "Docker"]
    assert [s["skill"] for s in grouped["non_technical"]] == [
        "Leadership",
        "Communication",
    ]


def test_kind_labels_and_blurbs_share_keys():
    assert set(KIND_LABELS.keys()) == set(KIND_BLURBS.keys())
    assert set(KIND_LABELS.keys()) == {"technical", "non_technical"}


def test_pick_differentiator_prefers_rising_over_in_demand():
    skills = [
        {"skill": "Python", "svs_score": 0.95, "category": "in_demand_now"},
        {"skill": "LangGraph", "svs_score": 0.62, "category": "rising_6mo"},
        {"skill": "WebGPU", "svs_score": 0.31, "category": "emerging_12mo"},
    ]
    assert pick_differentiator_skill(skills)["skill"] == "LangGraph"


def test_pick_differentiator_picks_highest_within_tier():
    skills = [
        {"skill": "LangGraph", "svs_score": 0.55, "category": "rising_6mo"},
        {"skill": "DSPy", "svs_score": 0.68, "category": "rising_6mo"},
        {"skill": "Python", "svs_score": 0.95, "category": "in_demand_now"},
    ]
    assert pick_differentiator_skill(skills)["skill"] == "DSPy"


def test_pick_differentiator_falls_back_to_emerging_then_mainstream():
    only_emerging = [
        {"skill": "WebGPU", "svs_score": 0.31, "category": "emerging_12mo"},
        {"skill": "Python", "svs_score": 0.95, "category": "in_demand_now"},
    ]
    assert pick_differentiator_skill(only_emerging)["skill"] == "WebGPU"

    only_mainstream = [
        {"skill": "Python", "svs_score": 0.95, "category": "in_demand_now"},
        {"skill": "SQL", "svs_score": 0.80, "category": "in_demand_now"},
    ]
    assert pick_differentiator_skill(only_mainstream)["skill"] == "Python"


def test_pick_differentiator_returns_none_for_empty_list():
    assert pick_differentiator_skill([]) is None


def test_group_skills_by_category_preserves_display_order():
    skills = [
        {"skill": "A", "category": "emerging_12mo"},
        {"skill": "B", "category": "in_demand_now"},
        {"skill": "C", "category": "rising_6mo"},
    ]

    grouped = group_skills_by_category(skills)

    assert list(grouped.keys()) == list(CATEGORY_LABELS.keys())
    assert grouped["in_demand_now"] == [{"skill": "B", "category": "in_demand_now"}]
    assert grouped["rising_6mo"] == [{"skill": "C", "category": "rising_6mo"}]
    assert grouped["emerging_12mo"] == [{"skill": "A", "category": "emerging_12mo"}]
