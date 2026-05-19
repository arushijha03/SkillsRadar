# SkillRadar Deployment Guide

## Backend: Railway

1. Push the repository to GitHub.
2. Create a new Railway project from the GitHub repo.
3. Railway will use `railway.json` and `Dockerfile`.
4. Add these environment variables in Railway:

```env
OPENAI_API_KEY=
PINECONE_API_KEY=
PINECONE_INDEX_NAME=skillradar
RAPIDAPI_KEY=
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT=skillradar
LANGCHAIN_TRACING_V2=true
```

5. Deploy and confirm `https://<railway-app>/health` returns `healthy`.

## Frontend: Streamlit Cloud

1. Create a Streamlit Cloud app from the same GitHub repo.
2. Set the app entrypoint to `frontend/streamlit_app.py`.
3. Add this secret:

```toml
API_BASE_URL = "https://<railway-app>"
```

4. Deploy and use the sidebar health check to confirm the API is reachable.

## Local Docker Smoke Test

```powershell
docker compose up --build
docker compose ps
curl http://localhost:8000/health
```

API: http://localhost:8000

Streamlit: http://localhost:8501

## Pre-Deploy Checklist

- Pinecone index exists and `scripts/check_pinecone.py` passes.
- LangSmith traces appear under the `skillradar` project.
- `python -m pytest tests/ -q` passes.
- `docker compose config --quiet` passes.
- `.env` is not committed.
