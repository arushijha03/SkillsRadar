# SkillsRadar

**A Multi-Agent Intelligence System for Proactive Skill Development**

Enter a target role → get a data-driven breakdown of in-demand skills today + predictions for what will matter in 6–12 months, powered by live job descriptions, tech blogs, arXiv papers, and HackerNews signals.

## Live Demo

| | Link |
|---|---|
| **Watch Demo** | [https://www.loom.com/share/9ed0533b7ebc4dbbbfc830597f12d9dc](https://www.loom.com/share/9ed0533b7ebc4dbbbfc830597f12d9dc) |
| **Try the app** (Streamlit UI) | [https://skillsradar-hfjwf7ewehabzwyj2sucd3.streamlit.app/](https://skillsradar-hfjwf7ewehabzwyj2sucd3.streamlit.app/) |
| **API health check** (Railway backend) | [https://web-production-25187.up.railway.app/health](https://web-production-25187.up.railway.app/health) |
| **API docs** | [https://web-production-25187.up.railway.app/docs](https://web-production-25187.up.railway.app/docs) |

**Quick try:** open the Streamlit app → enter a target role (e.g. *Forward Deployed Engineer*, area *AI*) → click **Find my skills**.

## Problem Statement

The tech skill landscape moves faster than the systems people use to track it:

- **Job descriptions lag the market by 6–12 months.** By the time a skill is widely listed, the early-mover advantage is gone.
- **Community signals are fragmented.** Engineering blogs, arXiv, Hacker News, and conference talks each carry partial signal, but no single source synthesizes them for a target role.
- **Self-directed learners over-index on what is already mainstream**, leading to crowded skill markets and slow ROI on learning time.
- **Career platforms surface descriptive analytics** (what was hot last quarter) instead of predictive guidance (what will matter in 6–12 months).

SkillRadar addresses this by combining lagging demand signals (job descriptions) with leading indicators (research papers, community discussion, technical blogs) into a single Skill Velocity Score per skill, per role.

## Business Impact

| Stakeholder | Value Delivered |
|---|---|
| **Engineers and job seekers** | Pick learning targets 6–12 months ahead of competing candidates, shifting from reactive upskilling to proactive positioning. |
| **Engineering managers** | Identify emerging skills to hire and train for before they become expensive and scarce in the market. |
| **Bootcamps and educators** | Justify curriculum updates with multi-source evidence rather than anecdote, reducing the risk of teaching outdated stacks. |
| **Career platforms and recruiters** | Differentiate with forward-looking skill intelligence instead of historical job posting analytics. |
| **Internal L&D teams** | Quantify "skills the org should be investing in" using auditable signals across JDs, research, and community discourse. |

The core differentiator is the **Skill Velocity Score (SVS)**, which blends three independent signal classes (demand, community, research) and categorizes every skill into one of three actionable horizons: in-demand-now, rising-6mo, and emerging-12mo.

## Architecture

```
User Input → FastAPI (/api/v1/analyze)
                    ┌─────────────────────────────┐
                    │   Orchestrator (LangGraph)   │
                    └──────┬──────┬──────┬─────────┘
                           │      │      │
                    ┌──────▼─┐ ┌──▼───┐ ┌▼──────────┐
                    │JD Agent│ │ RAG  │ │   Trend   │
                    └──────┬─┘ └──┬───┘ └────┬──────┘
                           └──────┴──────────┘
                                  │
                        ┌─────────▼──────────┐
                        │  Synthesis Agent   │
                        └─────────┬──────────┘
                        FastAPI + Streamlit UI
```

## Tech Stack

| Layer | Technology |
|---|---|
| Agent Orchestration | LangGraph |
| LLM Framework | LangChain |
| LLMs | OpenAI GPT-4o-mini + GPT-4o |
| Vector Database | Pinecone |
| Backend | FastAPI + Pydantic v2 |
| Frontend | Streamlit |
| Observability | LangSmith |
| Evaluation | RAGAs |
| Deployment | Railway + Streamlit Cloud |

## Results

End-to-end system is operational, deployable, and continuously evaluated.

### System Metrics

| Metric | Value |
|---|---|
| Agents orchestrated | 4 (JD, RAG, Trend, Synthesis) via LangGraph |
| Data sources integrated | 4 (JSearch JDs, Pinecone-backed tech blogs, arXiv, Hacker News) |
| RSS feeds ingested | 10 (LangChain, OpenAI, Hugging Face, Pinecone, AWS ML, Google AI, etc.) |
| Pinecone vectors (post-ingestion) | ~450 chunks across feeds, 1536-dim cosine index |
| Skill categories produced | 3 (`in_demand_now`, `rising_6mo`, `emerging_12mo`) |
| Test suite | **80 tests** passing across agents, API, RAG, orchestration, evaluation, frontend, and Docker |
| API isolation endpoints | `/jd-only`, `/rag-only`, `/trend-only` + full `/analyze` |
| Observability | LangSmith traces auto-enabled when API key present, gracefully disabled otherwise |
| Container health | Docker + Compose with `/health` health checks for API and Streamlit |

### Sample RAGAs Evaluation Scores

Run via `python evaluation/ragas_eval.py --output-json evaluation/ragas_scores.json`:

| Metric | Score | Interpretation |
|---|---|---|
| `faithfulness` | 0.33 | Topic list is partially grounded in retrieved chunks (baseline; improves with richer evidence prompts). |
| `answer_relevancy` | 0.55 | Generated answers are reasonably aligned with the asked question. |

These scores establish a measurable evaluation baseline; the harness is wired and reproducible per release.

### Example Output (Machine Learning Engineer, NLP, 6-month horizon)

The synthesis agent returns a ranked skill table, category groupings, and a markdown report. Representative output shape:

```json
{
  "role": "Machine Learning Engineer",
  "skills": [
    { "skill": "Python", "svs_score": 0.84, "category": "in_demand_now",
      "sources": ["job_descriptions", "community_rag"] },
    { "skill": "LangGraph", "svs_score": 0.61, "category": "rising_6mo",
      "sources": ["community_rag", "arxiv_hackernews"] }
  ],
  "report_markdown": "# SkillRadar Report: Machine Learning Engineer ..."
}
```

## Setup Instructions

### 1. Clone and set up environment

```powershell
cd SkillsRadar
python -m venv venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API keys

Copy `.env.example` to `.env` and fill in your keys:

```powershell
copy .env.example .env
```

| Service | URL |
|---|---|
| OpenAI | https://platform.openai.com |
| Pinecone | https://pinecone.io |
| LangSmith | https://smith.langchain.com |
| RapidAPI / JSearch | https://rapidapi.com (search "JSearch") |

**Pinecone index:** Name `skillradar` | Dimensions `1536` | Metric `cosine` | AWS us-east-1

**LangSmith tracing:** Keep `LANGCHAIN_TRACING_V2=false` for local development unless
`LANGCHAIN_API_KEY` is set. The app automatically disables hosted tracing when the
flag is true but no API key is present, avoiding noisy 401 warnings.

### 3. Index blog content (one-time)

```powershell
python scripts/ingest_blogs.py
```

Verify the index was populated:

```powershell
python scripts/check_pinecone.py --query "LangGraph agents"
```

Exit codes: `0` healthy · `1` no API key · `3` index missing · `4` index empty.

### 4. Run the API

```powershell
uvicorn app.main:app --reload
```

Open http://localhost:8000/docs

### 5. Run Streamlit UI (separate terminal)

```powershell
streamlit run frontend/streamlit_app.py
```

## API Usage

**POST** `/api/v1/analyze`

```json
{
  "role": "Machine Learning Engineer",
  "company": "optional",
  "domain": "NLP",
  "horizon": "6mo"
}
```

Per-agent isolation endpoints (useful for debugging and demos):

- `POST /api/v1/jd-only?role=Data%20Scientist`
- `POST /api/v1/rag-only?role=Machine%20Learning%20Engineer&domain=NLP`
- `POST /api/v1/trend-only?role=Machine%20Learning%20Engineer&domain=NLP&horizon=6mo`

End-to-end pipeline:

```powershell
curl -X POST "http://localhost:8000/api/v1/analyze" `
  -H "Content-Type: application/json" `
  -d "{\"role\":\"Machine Learning Engineer\",\"domain\":\"NLP\",\"horizon\":\"6mo\"}"
```

## Skill Velocity Score (SVS)

```
SVS = (0.5 × JD_frequency) + (0.3 × community_score) + (0.2 × trend_signal)

SVS ≥ 0.7  → in_demand_now
SVS 0.4–0.7 → rising_6mo
SVS < 0.4   → emerging_12mo
```

## Docker

```powershell
docker compose up --build
```

API: http://localhost:8000 | UI: http://localhost:8501

Health checks:

```powershell
docker compose ps
curl http://localhost:8000/health
```

## Testing

```powershell
python -m pytest tests/ -v
```

## RAGAs Evaluation

```powershell
python evaluation/ragas_eval.py --output-json evaluation/ragas_scores.json
```

Custom questions:

```powershell
python evaluation/ragas_eval.py `
  --question "What skills are trending for machine learning engineers?" `
  --question "What tools should backend engineers learn next?"
```

## Deployment

**Production (live):**

- **Frontend:** [Streamlit Cloud](https://skillsradar-hfjwf7ewehabzwyj2sucd3.streamlit.app/) — `frontend/streamlit_app.py`
- **Backend:** [Railway](https://web-production-25187.up.railway.app/health) — `railway.json` + `Dockerfile`

Streamlit Cloud secret:

```toml
API_BASE_URL = "https://web-production-25187.up.railway.app"
```

Full setup guide: [`DEPLOYMENT.md`](DEPLOYMENT.md)

Required production environment variables:

```env
OPENAI_API_KEY=
PINECONE_API_KEY=
PINECONE_INDEX_NAME=skillradar
RAPIDAPI_KEY=
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT=skillradar
LANGCHAIN_TRACING_V2=true
```

## Demo Checklist

1. Run `python scripts/check_pinecone.py --query "LangGraph agents"` and confirm the vector store is healthy.
2. Start the API and Streamlit UI.
3. Use role `Machine Learning Engineer`, domain `NLP`, horizon `6mo`.
4. Confirm the report includes SVS scores, categorized skills, and source labels.
5. Open LangSmith project `skillradar` and show the multi-agent trace.
6. Run `python evaluation/ragas_eval.py --output-json evaluation/ragas_scores.json` to show evaluation plumbing.

## Resume Bullets

- Built **SkillRadar**, a multi-agent FastAPI + Streamlit system that analyzes live job descriptions, tech blog vectors, arXiv papers, and Hacker News signals to forecast role-specific skill demand.
- Implemented a LangGraph orchestration pipeline with JD, RAG, Trend, and Synthesis agents, producing Skill Velocity Scores across `in_demand_now`, `rising_6mo`, and `emerging_12mo` categories.
- Integrated Pinecone-backed RAG ingestion from RSS feeds, LangSmith observability, RAGAs evaluation, Docker health checks, and Railway/Streamlit Cloud deployment configuration.
- Added a 80+ test suite covering agents, API routes, RAG ingestion/retrieval, LangGraph orchestration, RAGAs helpers, Docker config, and frontend formatting.

## 6-Day Build Plan

| Day | Focus |
|---|---|
| 1 | FastAPI + JD Agent |
| 2 | RAG pipeline + Pinecone |
| 3 | Trend Agent + LangGraph + SVS |
| 4 | LangSmith + RAGAs + Docker |
| 5 | Streamlit + Railway + Streamlit Cloud |
| 6 | README polish + resume bullets |

## Project Structure

```
SkillsRadar/
├── app/
│   ├── main.py              # FastAPI entry
│   ├── api/routes.py
│   ├── agents/              # JD, RAG, Trend, Synthesis, Orchestrator
│   ├── rag/                 # Indexer, retriever, ingestion
│   ├── models/schemas.py
│   ├── services/            # JSearch, arXiv, HN
│   └── utils/scoring.py     # SVS formula
├── frontend/streamlit_app.py
├── scripts/ingest_blogs.py
├── tests/
└── evaluation/ragas_eval.py
```

## License

MIT
