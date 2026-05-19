"""SkillRadar Streamlit UI.

The interface is intentionally written in plain English. Internal terms like
"SVS" or "agent" are translated to outcomes the user cares about:
- "Demand Score" (0–100) instead of "SVS"
- Friendly status pills ("Hot Right Now", "Heating Up", "On the Horizon")
- A live progress strip that narrates what the system is doing right now.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

import requests
import streamlit as st

from frontend.ui_helpers import (
    KIND_BLURBS,
    KIND_LABELS,
    build_analysis_payload,
    group_skills_by_kind,
    normalize_api_url,
    pick_differentiator_skill,
    skills_to_rows,
    to_demand_score,
)

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="SkillRadar",
    page_icon="📡",
    layout="wide",
)

st.markdown(
    """
    <style>
    .skill-card {
        background: #111827;
        border: 1px solid #1F2937;
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }
    .skill-card h4 {
        margin: 0 0 4px 0;
        color: #F8FAFC;
        font-size: 1.05rem;
    }
    .skill-card .score {
        font-size: 1.6rem;
        font-weight: 700;
        color: #38BDF8;
    }
    .skill-card .blurb {
        color: #94A3B8;
        font-size: 0.85rem;
        margin-top: 4px;
    }
    .pill {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        background: #1E293B;
        color: #E2E8F0;
        font-size: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📡 SkillRadar")
st.caption("Know which skills to learn next — before the market catches up.")

with st.sidebar:
    st.subheader("Settings")
    api_url = normalize_api_url(
        st.text_input("Server URL", value=API_BASE_URL, label_visibility="collapsed")
    )
    if st.button("Test connection", use_container_width=True):
        try:
            health = requests.get(f"{api_url}/health", timeout=10)
            health.raise_for_status()
            st.success("Connected.")
        except Exception as exc:
            st.error(f"Couldn't connect: {exc}")
    st.markdown("---")
    st.caption(
        "SkillRadar reads live job postings, expert tech blogs, "
        "research papers, and developer community signals to "
        "find skills that are growing in demand."
    )

st.markdown("### Tell us about your goal")
form_col1, form_col2 = st.columns([2, 1])
with form_col1:
    role = st.text_input(
        "What role are you targeting?",
        placeholder="e.g. Machine Learning Engineer, Product Manager, Backend Developer",
    )
    sub_col1, sub_col2 = st.columns(2)
    with sub_col1:
        company = st.text_input(
            "Any specific company? (optional)",
            placeholder="e.g. Google",
        )
    with sub_col2:
        domain = st.text_input(
            "Any specific area? (optional)",
            placeholder="e.g. NLP, fintech",
        )

with form_col2:
    horizon_display = st.radio(
        "How far ahead should we look?",
        ["6 months", "12 months"],
        index=0,
        horizontal=False,
    )
    horizon = "6mo" if horizon_display == "6 months" else "12mo"

analyze_btn = st.button(
    "Find my skills",
    type="primary",
    use_container_width=True,
    disabled=not role.strip(),
)


def _run_analysis(
    api_url: str,
    payload: dict[str, Any],
    result_ref: dict[str, Any],
) -> None:
    """Worker thread: call the FastAPI analyze endpoint."""
    try:
        response = requests.post(
            f"{api_url}/api/v1/analyze",
            json=payload,
            timeout=300,
        )
        response.raise_for_status()
        result_ref["data"] = response.json()
    except requests.exceptions.ConnectionError:
        result_ref["error"] = (
            f"Couldn't reach the server at {api_url}. "
            "Make sure the backend is running."
        )
    except requests.exceptions.HTTPError as exc:
        result_ref["error"] = f"Server error: {exc.response.text}"
    except Exception as exc:
        result_ref["error"] = f"Something went wrong: {exc}"


PROGRESS_STEPS = [
    ("📋", "Reading live job postings"),
    ("📚", "Scanning expert tech blogs"),
    ("🔬", "Checking research papers and developer communities"),
    ("✍️", "Writing your personalized report"),
]


def _animate_progress(thread: threading.Thread) -> None:
    """Show step-by-step progress while the worker thread runs."""
    progress_bar = st.progress(0.0)
    placeholders = [st.empty() for _ in PROGRESS_STEPS]
    for emoji, label in PROGRESS_STEPS:
        # Initial pending state for each row.
        idx = PROGRESS_STEPS.index((emoji, label))
        placeholders[idx].markdown(f"<span style='color:#475569'>{emoji} &nbsp; {label}</span>", unsafe_allow_html=True)

    seconds_per_step = 18  # Approximate full pipeline ~70s end-to-end.
    start = time.time()
    current = 0

    while thread.is_alive():
        elapsed = time.time() - start
        target = min(int(elapsed // seconds_per_step), len(PROGRESS_STEPS) - 1)

        while current <= target:
            emoji, label = PROGRESS_STEPS[current]
            placeholders[current].markdown(
                f"⏳ &nbsp; {emoji} &nbsp; **{label}…**",
                unsafe_allow_html=True,
            )
            current += 1

        progress = min(0.95, elapsed / (seconds_per_step * len(PROGRESS_STEPS)))
        progress_bar.progress(progress)
        time.sleep(0.4)

    for idx, (emoji, label) in enumerate(PROGRESS_STEPS):
        placeholders[idx].markdown(
            f"✅ &nbsp; {emoji} &nbsp; {label}",
            unsafe_allow_html=True,
        )
    progress_bar.progress(1.0)


def _render_kind_columns(grouped: dict[str, list[dict[str, Any]]]) -> None:
    """Two side-by-side cards: Technical / Non-Technical."""
    cols = st.columns(2)
    for col, (kind_key, kind_label) in zip(cols, KIND_LABELS.items()):
        with col:
            st.markdown(f"#### {kind_label}")
            st.caption(KIND_BLURBS[kind_key])
            kind_skills = grouped[kind_key][:8]
            if not kind_skills:
                st.markdown(
                    "<span style='color:#64748B'>Nothing notable here yet.</span>",
                    unsafe_allow_html=True,
                )
                continue
            for skill in kind_skills:
                score = to_demand_score(skill.get("svs_score", 0))
                st.markdown(
                    f"""
                    <div class="skill-card">
                        <h4>{skill.get("skill", "")}</h4>
                        <div class="score">{score}<span style='font-size:0.9rem;color:#64748B'> / 100</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


if analyze_btn and role.strip():
    payload = build_analysis_payload(role, company, domain, horizon)
    result: dict[str, Any] = {"data": None, "error": None}

    st.markdown("---")
    st.subheader("Working on it…")
    st.caption(
        "This takes roughly 1–2 minutes. We're checking several data sources "
        "in parallel — feel free to grab a coffee."
    )

    worker = threading.Thread(
        target=_run_analysis,
        args=(api_url, payload, result),
        daemon=True,
    )
    worker.start()
    _animate_progress(worker)
    worker.join()

    if result["error"]:
        st.error(result["error"])
        st.stop()

    data = result["data"] or {}
    skills = data.get("skills", [])

    st.markdown("---")
    if not skills:
        st.warning(
            "We couldn't find clear skill signals for this role. "
            "Try a broader role name or a different area."
        )
        st.stop()

    differentiator = pick_differentiator_skill(skills) or {}
    headline_cols = st.columns(3)
    headline_cols[0].metric("Skills found", len(skills))
    headline_cols[1].metric(
        "Skill that sets you apart",
        differentiator.get("skill", "—"),
        help=(
            "The fastest-growing skill that hasn't gone mainstream yet — "
            "learning it now gives you an early-mover advantage."
        ),
    )
    headline_cols[2].metric(
        "Its demand score",
        f"{to_demand_score(differentiator.get('svs_score', 0))} / 100",
    )

    st.markdown("### Skills breakdown")
    grouped = group_skills_by_kind(skills)
    _render_kind_columns(grouped)

    with st.expander("See the full skill list"):
        st.dataframe(
            skills_to_rows(skills),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Demand Score": st.column_config.ProgressColumn(
                    "Demand Score",
                    help="0 = niche, 100 = white-hot demand",
                    format="%d",
                    min_value=0,
                    max_value=100,
                ),
            },
        )

    with st.expander("Read the full personalized report"):
        st.markdown(data.get("report_markdown", "_No report generated._"))
