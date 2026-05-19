"""Verify Pinecone setup: connection, index existence, vector count, sample query.

Usage:
    python scripts/check_pinecone.py
    python scripts/check_pinecone.py --query "LangGraph agents"

Exits non-zero if Pinecone is unreachable or the index is empty.
"""

import argparse
import logging
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(ROOT, ".env"))

from app.rag.retriever import PineconeRetriever  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Pinecone health check for SkillRadar.")
    parser.add_argument(
        "--query",
        default="machine learning engineer skills",
        help="Sample query to run against the index.",
    )
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    api_key = os.getenv("PINECONE_API_KEY", "")
    index_name = os.getenv("PINECONE_INDEX_NAME", "skillradar")

    if not api_key:
        logger.error("PINECONE_API_KEY not set in .env. Aborting.")
        return 1

    logger.info("Connecting to Pinecone (index=%s)...", index_name)
    retriever = PineconeRetriever(api_key=api_key, index_name=index_name)

    try:
        existing = [idx.name for idx in retriever.client.list_indexes()]
    except Exception as exc:
        logger.error("Failed to list indexes: %s", exc)
        return 2

    if index_name not in existing:
        logger.error(
            "Index %r not found. Existing indexes: %s. "
            "Run `python scripts/ingest_blogs.py` to create and populate it.",
            index_name,
            existing,
        )
        return 3

    index = retriever.client.Index(index_name)
    stats = index.describe_index_stats()
    total = stats.get("total_vector_count", 0)
    logger.info("Index %r has %s vectors.", index_name, total)

    if total == 0:
        logger.warning("Index is empty. Run `python scripts/ingest_blogs.py` to populate it.")
        return 4

    logger.info("Running sample query: %r (top_k=%d)", args.query, args.top_k)
    results = retriever.retrieve(args.query, top_k=args.top_k)

    if not results:
        logger.warning("Query returned no matches.")
        return 5

    for i, chunk in enumerate(results, 1):
        logger.info(
            "  [%d] score=%.3f  %s  (%s)",
            i,
            chunk["score"],
            chunk["title"][:80] or "<no title>",
            chunk["url"][:80],
        )

    logger.info("Pinecone health check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
