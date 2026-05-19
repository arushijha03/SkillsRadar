"""Tests for Docker and compose configuration."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_has_api_healthcheck():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "HEALTHCHECK" in dockerfile
    assert "http://127.0.0.1:8000/health" in dockerfile
    assert 'CMD ["uvicorn", "app.main:app"' in dockerfile


def test_dockerignore_excludes_local_secrets_and_virtualenv():
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()

    assert ".env" in dockerignore
    assert "venv" in dockerignore
    assert ".git" in dockerignore
    assert "!/.env.example" in dockerignore


def test_compose_defines_api_and_frontend_healthchecks():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))

    api = compose["services"]["api"]
    frontend = compose["services"]["frontend"]

    assert "healthcheck" in api
    assert "8000/health" in " ".join(api["healthcheck"]["test"])

    assert frontend["depends_on"]["api"]["condition"] == "service_healthy"
    assert "healthcheck" in frontend
    assert "8501/_stcore/health" in " ".join(frontend["healthcheck"]["test"])
