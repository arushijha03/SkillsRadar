"""LangSmith tracing configuration helpers."""

import os


def configure_langsmith_tracing() -> bool:
    """Enable hosted LangSmith tracing only when a usable API key is present.

    LangChain reads ``LANGCHAIN_TRACING_V2`` during client construction. If tracing
    is set to true without ``LANGCHAIN_API_KEY``, local runs still work but logs get
    noisy 401 warnings. This keeps local development quiet while preserving opt-in
    tracing for Day 4 observability.
    """
    tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    has_api_key = bool(os.getenv("LANGCHAIN_API_KEY", "").strip())

    if tracing_enabled and not has_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        return False

    return tracing_enabled and has_api_key
