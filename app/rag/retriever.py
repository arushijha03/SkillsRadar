"""Pinecone retriever for RAG agent queries."""

import os
from typing import Any

from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone


class PineconeRetriever:
    """Query Pinecone for relevant blog chunks."""

    def __init__(
        self,
        api_key: str | None = None,
        index_name: str | None = None,
        top_k: int = 8,
    ):
        self.api_key = api_key or os.getenv("PINECONE_API_KEY", "")
        self.index_name = index_name or os.getenv(
            "PINECONE_INDEX_NAME", "skillradar"
        )
        self.top_k = top_k
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self._client: Pinecone | None = None

    @property
    def client(self) -> Pinecone:
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY is not set.")
        if self._client is None:
            self._client = Pinecone(api_key=self.api_key)
        return self._client

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """Return top-k matching chunks with scores."""
        k = top_k or self.top_k
        index = self.client.Index(self.index_name)
        query_vector = self.embeddings.embed_query(query)

        results = index.query(
            vector=query_vector,
            top_k=k,
            include_metadata=True,
        )

        chunks: list[dict[str, Any]] = []
        for match in results.get("matches") or []:
            meta = match.get("metadata") or {}
            chunks.append(
                {
                    "text": meta.get("text", ""),
                    "title": meta.get("title", ""),
                    "url": meta.get("url", ""),
                    "source": meta.get("source", ""),
                    "score": match.get("score", 0.0),
                }
            )
        return chunks

    def format_for_llm(self, chunks: list[dict[str, Any]]) -> str:
        """Format retrieved chunks as context string."""
        if not chunks:
            return "No relevant community content found in the vector store."

        lines = []
        for i, c in enumerate(chunks, 1):
            lines.append(
                f"[{i}] (score={c['score']:.3f}) {c['title']}\n"
                f"Source: {c['url']}\n"
                f"{c['text'][:600]}"
            )
        return "\n\n---\n\n".join(lines)
