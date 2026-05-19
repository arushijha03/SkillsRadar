"""SkillRadar FastAPI application entry point."""

import logging
import os

from dotenv import load_dotenv

# Must run before any LangChain import for LangSmith tracing
load_dotenv()

from app.utils.langsmith import configure_langsmith_tracing

configure_langsmith_tracing()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(
    title="SkillRadar",
    description=(
        "Multi-Agent Intelligence System for Proactive Skill Development. "
        "Enter a target role to get in-demand skills today plus 6–12 month predictions."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health")
async def health():
    """Health check for Railway and Docker."""
    return {
        "status": "healthy",
        "service": "skillradar",
        "langsmith_tracing": os.getenv("LANGCHAIN_TRACING_V2", "false"),
    }


@app.get("/")
async def root():
    return {
        "message": "SkillRadar API",
        "docs": "/docs",
        "analyze": "POST /api/v1/analyze",
    }
