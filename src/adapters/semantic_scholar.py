"""Adapter for the Semantic Scholar Academic Graph API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import requests


SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"

_DEFAULT_FIELDS = "paperId,title,abstract,authors,year,citationCount,externalIds,url"


@dataclass
class S2Paper:
    """Lightweight representation of a Semantic Scholar paper."""

    paper_id: str
    title: str
    abstract: str
    authors: list[str]
    year: Optional[int]
    citation_count: int
    url: str
    external_ids: dict[str, str] = field(default_factory=dict)


class SemanticScholarAdapter:
    """Thin wrapper around the Semantic Scholar REST API.

    An API key is optional but strongly recommended to avoid rate-limiting.
    Set the ``SEMANTIC_SCHOLAR_API_KEY`` environment variable or pass it
    directly via *api_key*.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._session = requests.Session()
        if api_key:
            self._session.headers["x-api-key"] = api_key

    def search(self, query: str, limit: int = 10) -> list[S2Paper]:
        """Keyword search across the Semantic Scholar corpus."""
        resp = self._session.get(
            f"{SEMANTIC_SCHOLAR_BASE}/paper/search",
            params={"query": query, "limit": limit, "fields": _DEFAULT_FIELDS},
            timeout=15,
        )
        resp.raise_for_status()
        data: list[dict[str, Any]] = resp.json().get("data", [])
        return [self._to_paper(d) for d in data]

    def fetch_by_id(self, paper_id: str) -> S2Paper:
        """Fetch a single paper by Semantic Scholar paper ID or DOI."""
        resp = self._session.get(
            f"{SEMANTIC_SCHOLAR_BASE}/paper/{paper_id}",
            params={"fields": _DEFAULT_FIELDS},
            timeout=15,
        )
        resp.raise_for_status()
        return self._to_paper(resp.json())

    @staticmethod
    def _to_paper(data: dict[str, Any]) -> S2Paper:
        return S2Paper(
            paper_id=data.get("paperId", ""),
            title=data.get("title", ""),
            abstract=data.get("abstract", ""),
            authors=[a["name"] for a in data.get("authors", [])],
            year=data.get("year"),
            citation_count=data.get("citationCount", 0),
            url=data.get("url", ""),
            external_ids=data.get("externalIds", {}),
        )
