"""One-time script to populate Pinecone with tech blog content."""

import logging
import os
import sys

# Windows-safe path setup for scripts/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(ROOT, ".env"))

from app.rag.indexer import PineconeIndexer
from app.rag.ingestion.blog_ingester import BlogIngester

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("Fetching blog articles from RSS feeds...")
    ingester = BlogIngester()
    articles = ingester.fetch_all(max_entries_per_feed=50)
    logger.info("Fetched %d articles total", len(articles))

    if not articles:
        logger.error("No articles fetched. Check network and feed URLs.")
        sys.exit(1)

    logger.info("Indexing into Pinecone (this may take several minutes)...")
    indexer = PineconeIndexer()
    count = indexer.index_documents(articles)
    logger.info("Successfully indexed %d vectors.", count)


if __name__ == "__main__":
    main()
