"""Pure helpers for the Streamlit UI.

Internally the API still returns 0.0–1.0 SVS scores and machine-readable
category keys. The UI translates those to a plain-English vocabulary:
- "Demand Score" on a 0–100 scale
- Friendly status labels with intuitive emoji
"""

from __future__ import annotations

from typing import Any

CATEGORY_LABELS = {
    "in_demand_now": "🔥 Hot Right Now",
    "rising_6mo": "📈 Heating Up",
    "emerging_12mo": "🌱 On the Horizon",
}

CATEGORY_BLURBS = {
    "in_demand_now": "Employers are hiring for this today.",
    "rising_6mo": "Growing fast — invest in it for the next 6 months.",
    "emerging_12mo": "Early signals — bet on it for the next year.",
}


def build_analysis_payload(
    role: str,
    company: str | None,
    domain: str | None,
    horizon: str,
) -> dict[str, str | None]:
    """Build the FastAPI request payload from form values."""
    return {
        "role": role.strip(),
        "company": company.strip() if company and company.strip() else None,
        "domain": domain.strip() if domain and domain.strip() else None,
        "horizon": horizon,
    }


def normalize_api_url(api_url: str) -> str:
    """Normalize a user-provided API URL."""
    return api_url.strip().rstrip("/")


def to_demand_score(svs_score: float | int) -> int:
    """Convert an internal 0.0–1.0 SVS to a 0–100 demand score."""
    return max(0, min(100, round(float(svs_score) * 100)))


NON_TECHNICAL_KEYWORDS: tuple[str, ...] = (
    "communication",
    "collaboration",
    "teamwork",
    "leadership",
    "mentor",
    "coaching",
    "presentation",
    "public speaking",
    "negotiation",
    "stakeholder",
    "cross-functional",
    "interpersonal",
    "people management",
    "people skills",
    "emotional intelligence",
    "empathy",
    "active listening",
    "storytelling",
    "writing",
    "documentation",
    "facilitation",
    "influencing",
    "feedback",
    "conflict resolution",
    "decision making",
    "decision-making",
    "critical thinking",
    "analytical thinking",
    "problem solving",
    "problem-solving",
    "creativity",
    "adaptability",
    "growth mindset",
    "self-motivation",
    "ownership",
    "accountability",
    "time management",
    "prioritization",
    "planning",
    "organization",
    "organizational",
    "project management",
    "product thinking",
    "product management",
    "program management",
    "business acumen",
    "strategy",
    "strategic thinking",
    "customer focus",
    "customer-centric",
    "scrum",
    "agile",
    "kanban",
    "attention to detail",
)

KIND_LABELS: dict[str, str] = {
    "technical": "💻 Technical Skills",
    "non_technical": "🤝 Non-Technical Skills",
}

KIND_BLURBS: dict[str, str] = {
    "technical": "Tools, languages, and platforms employers want you to know.",
    "non_technical": "Ways of working and thinking that make you stand out.",
}


def classify_skill_kind(skill_name: str) -> str:
    """Classify a skill name as 'technical' or 'non_technical'.

    Uses a keyword allowlist for soft / process skills. Anything else is
    treated as technical, which is the safer default for a tech-skills app.
    """
    name = (skill_name or "").lower()
    for keyword in NON_TECHNICAL_KEYWORDS:
        if keyword in name:
            return "non_technical"
    return "technical"


def skills_to_rows(skills: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Format skill signals for the simplified results table."""
    rows = []
    for skill in skills:
        kind = classify_skill_kind(skill.get("skill", ""))
        rows.append(
            {
                "Skill": skill.get("skill", ""),
                "Demand Score": to_demand_score(skill.get("svs_score", 0)),
                "Type": "Technical" if kind == "technical" else "Non-technical",
            }
        )
    return rows


def group_skills_by_category(
    skills: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group skill signals by internal category (used for differentiator logic)."""
    return {
        category: [skill for skill in skills if skill.get("category") == category]
        for category in CATEGORY_LABELS
    }


def group_skills_by_kind(
    skills: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group skills into technical vs non-technical, sorted by demand score."""
    buckets: dict[str, list[dict[str, Any]]] = {kind: [] for kind in KIND_LABELS}
    for skill in skills:
        buckets[classify_skill_kind(skill.get("skill", ""))].append(skill)
    for bucket in buckets.values():
        bucket.sort(key=lambda s: float(s.get("svs_score", 0)), reverse=True)
    return buckets


# Categories ordered from most differentiating to least.
# A "Heating Up" skill is rising fast but not yet mainstream — high upside.
# An "On the Horizon" skill is early — biggest bet, biggest payoff if it lands.
# A "Hot Right Now" skill is the fallback when nothing else stands out.
DIFFERENTIATOR_PREFERENCE = ("rising_6mo", "emerging_12mo", "in_demand_now")


def pick_differentiator_skill(
    skills: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Pick the skill most likely to set the user apart from competitors.

    Prefers fast-rising and emerging skills over already-saturated ones, since
    those carry the strongest early-mover advantage. Within each tier, returns
    the highest demand-score skill.
    """
    if not skills:
        return None

    grouped = group_skills_by_category(skills)
    for category in DIFFERENTIATOR_PREFERENCE:
        candidates = grouped.get(category) or []
        if candidates:
            return max(candidates, key=lambda s: float(s.get("svs_score", 0)))

    return max(skills, key=lambda s: float(s.get("svs_score", 0)))
