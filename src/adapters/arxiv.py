"""Adapter for the arXiv public API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import arxiv


@dataclass
class ArxivPaper:
    """Lightweight representation of an arXiv paper."""

    paper_id: str
    title: str
    abstract: str
    authors: list[str]
    url: str
    pdf_url: str
    categories: list[str] = field(default_factory=list)


class ArxivAdapter:
    """Wraps the ``arxiv`` Python client to fetch paper metadata and PDFs."""

    def __init__(self, max_results: int = 10) -> None:
        self.max_results = max_results
        self._client = arxiv.Client()

    def search(self, query: str, max_results: Optional[int] = None) -> list[ArxivPaper]:
        """Return a list of papers matching *query*."""
        limit = max_results or self.max_results
        search = arxiv.Search(query=query, max_results=limit)
        results = self._client.results(search)
        return [self._to_paper(r) for r in results]

    def fetch_by_id(self, arxiv_id: str) -> ArxivPaper:
        """Fetch a single paper by its arXiv ID (e.g. ``2301.00001``)."""
        search = arxiv.Search(id_list=[arxiv_id])
        result = next(self._client.results(search))
        return self._to_paper(result)

    @staticmethod
    def _to_paper(result: arxiv.Result) -> ArxivPaper:
        return ArxivPaper(
            paper_id=result.entry_id,
            title=result.title,
            abstract=result.summary,
            authors=[str(a) for a in result.authors],
            url=result.entry_id,
            pdf_url=result.pdf_url,
            categories=result.categories,
        )
