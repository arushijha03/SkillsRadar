"""Pinecone vector indexing for blog content."""

import logging
import os
import time
import uuid
from typing import Any

from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

from app.rag.ingestion.preprocessor import iter_chunks

logger = logging.getLogger(__name__)

EMBEDDING_DIMENSION = 1536
BATCH_SIZE = 50
SLEEP_BETWEEN_BATCHES = 1.0


class PineconeIndexer:
    """Embed documents and upsert into Pinecone."""

    def __init__(
        self,
        api_key: str | None = None,
        index_name: str | None = None,
    ):
        self.api_key = api_key or os.getenv("PINECONE_API_KEY", "")
        self.index_name = index_name or os.getenv(
            "PINECONE_INDEX_NAME", "skillradar"
        )
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self._client: Pinecone | None = None
        self._index = None

    @property
    def client(self) -> Pinecone:
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY is not set.")
        if self._client is None:
            self._client = Pinecone(api_key=self.api_key)
        return self._client

    @property
    def index(self):
        if self._index is None:
            self._index = self.client.Index(self.index_name)
        return self._index

    def ensure_index_exists(self) -> None:
        """Create Pinecone index if it does not exist."""
        existing = [idx.name for idx in self.client.list_indexes()]
        if self.index_name in existing:
            return

        logger.info("Creating Pinecone index: %s", self.index_name)
        self.client.create_index(
            name=self.index_name,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not self.client.describe_index(self.index_name).status.get("ready"):
            time.sleep(1)

    def index_documents(self, documents: list[dict[str, Any]]) -> int:
        """Chunk, embed, and upsert documents. Returns vector count."""
        self.ensure_index_exists()

        vectors_upserted = 0
        batch_texts: list[str] = []
        batch_metas: list[dict] = []
        batch_ids: list[str] = []

        for chunk_doc in iter_chunks(documents):
            batch_texts.append(chunk_doc["text"])
            batch_metas.append(chunk_doc["metadata"])
            batch_ids.append(str(uuid.uuid4()))

            if len(batch_texts) >= BATCH_SIZE:
                vectors_upserted += self._upsert_batch(
                    batch_ids, batch_texts, batch_metas
                )
                batch_texts, batch_metas, batch_ids = [], [], []
                time.sleep(SLEEP_BETWEEN_BATCHES)

        if batch_texts:
            vectors_upserted += self._upsert_batch(
                batch_ids, batch_texts, batch_metas
            )

        return vectors_upserted

    def _upsert_batch(
        self,
        ids: list[str],
        texts: list[str],
        metadatas: list[dict],
    ) -> int:
        embeddings = self.embeddings.embed_documents(texts)
        vectors = []
        for vid, emb, meta, text in zip(ids, embeddings, metadatas, texts):
            meta_with_text = {**meta, "text": text[:1000]}
            vectors.append({"id": vid, "values": emb, "metadata": meta_with_text})

        self.index.upsert(vectors=vectors)
        return len(vectors)
