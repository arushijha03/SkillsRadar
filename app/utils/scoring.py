"""Skill Velocity Score (SVS) computation and skill merging."""

from typing import Literal

Category = Literal["in_demand_now", "rising_6mo", "emerging_12mo"]

SVS_WEIGHTS = {
    "jd": 0.5,
    "community": 0.3,
    "trend": 0.2,
}


def compute_svs(
    jd_frequency_normalized: float,
    community_score: float,
    trend_signal_strength: float,
) -> float:
    """Compute Skill Velocity Score from normalized component scores."""
    return (
        SVS_WEIGHTS["jd"] * jd_frequency_normalized
        + SVS_WEIGHTS["community"] * community_score
        + SVS_WEIGHTS["trend"] * trend_signal_strength
    )


def categorize_svs(svs_score: float) -> Category:
    """Map SVS to time-horizon category."""
    if svs_score >= 0.7:
        return "in_demand_now"
    if svs_score >= 0.4:
        return "rising_6mo"
    return "emerging_12mo"


def merge_skill_signals(
    jd_skills: list[dict],
    rag_topics: list[dict],
    trend_technologies: list[dict],
) -> list[dict]:
    """
    Merge outputs from JD, RAG, and Trend agents into unified skill records.

    Returns list of dicts with keys: skill, svs_score, category, sources,
    jd_frequency_normalized, community_score, trend_signal_strength.
    """
    skill_map: dict[str, dict] = {}

    for item in jd_skills:
        name = item.get("skill", "").strip()
        if not name:
            continue
        key = name.lower()
        skill_map[key] = {
            "skill": name,
            "jd_frequency_normalized": float(item.get("frequency_normalized", 0)),
            "community_score": 0.0,
            "trend_signal_strength": 0.0,
            "sources": ["job_descriptions"],
            "category_hint": item.get("category", ""),
        }

    for item in rag_topics:
        topic = item.get("topic", "").strip()
        if not topic:
            continue
        key = topic.lower()
        momentum = float(item.get("momentum", item.get("community_score", 0)))
        community = float(item.get("community_score", momentum))
        if key in skill_map:
            skill_map[key]["community_score"] = max(
                skill_map[key]["community_score"], community
            )
            if "community_rag" not in skill_map[key]["sources"]:
                skill_map[key]["sources"].append("community_rag")
        else:
            skill_map[key] = {
                "skill": topic,
                "jd_frequency_normalized": 0.0,
                "community_score": community,
                "trend_signal_strength": 0.0,
                "sources": ["community_rag"],
                "category_hint": "",
            }

    for item in trend_technologies:
        tech = item.get("technology", "").strip()
        if not tech:
            continue
        key = tech.lower()
        strength = float(item.get("signal_strength", 0))
        if key in skill_map:
            skill_map[key]["trend_signal_strength"] = max(
                skill_map[key]["trend_signal_strength"], strength
            )
            if "arxiv_hackernews" not in skill_map[key]["sources"]:
                skill_map[key]["sources"].append("arxiv_hackernews")
        else:
            skill_map[key] = {
                "skill": tech,
                "jd_frequency_normalized": 0.0,
                "community_score": 0.0,
                "trend_signal_strength": strength,
                "sources": ["arxiv_hackernews"],
                "category_hint": item.get("time_horizon", ""),
            }

    results = []
    for record in skill_map.values():
        svs = compute_svs(
            record["jd_frequency_normalized"],
            record["community_score"],
            record["trend_signal_strength"],
        )
        results.append(
            {
                "skill": record["skill"],
                "svs_score": round(svs, 4),
                "category": categorize_svs(svs),
                "sources": record["sources"],
            }
        )

    results.sort(key=lambda x: x["svs_score"], reverse=True)
    return results
