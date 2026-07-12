"""Adapters for connecting to external data sources and services."""

from .arxiv import ArxivAdapter
from .semantic_scholar import SemanticScholarAdapter
from .pdf import PDFAdapter

__all__ = ["ArxivAdapter", "SemanticScholarAdapter", "PDFAdapter"]
