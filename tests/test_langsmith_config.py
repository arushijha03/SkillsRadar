"""Tests for LangSmith tracing environment configuration."""

from app.utils.langsmith import configure_langsmith_tracing


def test_configure_langsmith_disables_tracing_without_api_key(monkeypatch):
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)

    enabled = configure_langsmith_tracing()

    assert enabled is False
    assert "LANGCHAIN_TRACING_V2" in __import__("os").environ
    assert __import__("os").environ["LANGCHAIN_TRACING_V2"] == "false"


def test_configure_langsmith_keeps_tracing_with_api_key(monkeypatch):
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "lsv2_test")

    assert configure_langsmith_tracing() is True


def test_configure_langsmith_stays_disabled_when_false(monkeypatch):
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "lsv2_test")

    assert configure_langsmith_tracing() is False
