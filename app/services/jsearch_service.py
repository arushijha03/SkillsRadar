"""JSearch API client via RapidAPI."""

import os
from typing import Any

import requests


class JSearchService:
    """Fetch live job descriptions from JSearch (RapidAPI)."""

    BASE_URL = "https://jsearch.p.rapidapi.com/search"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("RAPIDAPI_KEY", "")
        self.headers = {
            "X-RapidAPI-Key": self.api_key,
            "X-RapidAPI-Host": "jsearch.p.rapidapi.com",
        }

    def search_jobs(
        self,
        query: str,
        num_pages: int = 1,
        page: int = 1,
    ) -> list[dict[str, Any]]:
        """
        Search job listings. Returns up to ~10 jobs per page.

        Raises ValueError if API key missing or response invalid.
        """
        if not self.api_key:
            raise ValueError(
                "RAPIDAPI_KEY is not set. Add it to your .env file."
            )

        params = {
            "query": query,
            "page": str(page),
            "num_pages": str(num_pages),
            "date_posted": "month",
        }

        response = requests.get(
            self.BASE_URL,
            headers=self.headers,
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()

        data = payload.get("data") or []
        if not data:
            return []

        return data

    def fetch_job_descriptions(
        self,
        role: str,
        company: str | None = None,
        limit: int = 20,
    ) -> list[str]:
        """Fetch job description text for skill extraction."""
        query_parts = [role]
        if company:
            query_parts.append(company)
        query = " ".join(query_parts)

        jobs = self.search_jobs(query=query, num_pages=2)
        descriptions: list[str] = []

        for job in jobs:
            desc = job.get("job_description") or job.get("description") or ""
            if desc and len(desc.strip()) > 50:
                descriptions.append(desc.strip())
            if len(descriptions) >= limit:
                break

        return descriptions
