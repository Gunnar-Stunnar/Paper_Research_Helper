"""Adapters for connecting to external data sources and services."""

from .arxiv import ArxivAdapter
from .semantic_scholar import SemanticScholarAdapter
from .pdf import PDFAdapter
from .postgres import LocalPostgresAdapter
from .aurora import AuroraAdapter
from .claude import ClaudeAdapter
from .bedrock import BedrockAdapter

__all__ = [
    # Data sources
    "ArxivAdapter",
    "SemanticScholarAdapter",
    "PDFAdapter",
    # Vector stores
    "LocalPostgresAdapter",
    "AuroraAdapter",
    # LLMs
    "ClaudeAdapter",
    "BedrockAdapter",
]
