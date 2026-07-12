"""LangChain tool for searching papers via the ArXiv and Semantic Scholar adapters."""

from __future__ import annotations

import json
from typing import Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from src.adapters.arxiv import ArxivAdapter
from src.adapters.semantic_scholar import SemanticScholarAdapter


class PaperSearchInput(BaseModel):
    query: str = Field(description="Natural-language search query for academic papers.")
    source: str = Field(
        default="arxiv",
        description="Data source to search: 'arxiv' or 'semantic_scholar'.",
    )
    max_results: int = Field(default=5, description="Maximum number of results to return.")


class PaperSearchTool(BaseTool):
    """Search for academic papers by keyword or topic."""

    name: str = "paper_search"
    description: str = (
        "Search for academic papers on arXiv or Semantic Scholar. "
        "Returns titles, authors, abstracts, and URLs."
    )
    args_schema: Type[BaseModel] = PaperSearchInput

    _arxiv: ArxivAdapter = ArxivAdapter()
    _s2: SemanticScholarAdapter = SemanticScholarAdapter()

    def _run(self, query: str, source: str = "arxiv", max_results: int = 5) -> str:
        if source == "semantic_scholar":
            papers = self._s2.search(query, limit=max_results)
            results = [
                {"title": p.title, "authors": p.authors, "abstract": p.abstract, "url": p.url}
                for p in papers
            ]
        else:
            papers = self._arxiv.search(query, max_results=max_results)
            results = [
                {"title": p.title, "authors": p.authors, "abstract": p.abstract, "url": p.url}
                for p in papers
            ]
        return json.dumps(results, indent=2)
