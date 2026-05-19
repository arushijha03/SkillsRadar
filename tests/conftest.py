"""Shared pytest fixtures and test-time environment setup.

Sets dummy API keys so eager constructors (e.g. ``OpenAIEmbeddings``,
``ChatOpenAI``) don't raise during test collection. Tests should still mock
all real network calls.
"""

import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy")
os.environ.setdefault("PINECONE_API_KEY", "pc-test-dummy")
os.environ.setdefault("PINECONE_INDEX_NAME", "skillradar-test")
os.environ.setdefault("RAPIDAPI_KEY", "rapid-test-dummy")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
