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

**If deploy fails at the Network step:** the app must listen on Railway's `$PORT`
(not a hardcoded `8000`). The `Dockerfile` starts uvicorn via `sh -c` so `${PORT}`
is expanded correctly.

**If logs show `'$PORT' is not a valid integer`:** remove any custom **Start Command**
in the Railway dashboard (it overrides the Dockerfile and may pass `$PORT` literally).
Leave Start Command empty and redeploy.

Also check **Deploy Logs** for crash-on-start (missing env vars, import errors).
In Railway → **Settings** → **Networking**, generate a public domain after the
service is healthy.

## Frontend: Streamlit Cloud

1. Create a Streamlit Cloud app from the same GitHub repo.
2. Set the app entrypoint to `frontend/streamlit_app.py`.
3. Add this secret:

```toml
API_BASE_URL = "https://web-production-25187.up.railway.app"
```

(Production Streamlit app: https://skillsradar-hfjwf7ewehabzwyj2sucd3.streamlit.app/)

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
