"""Paper ingestion pipeline.

Responsibilities
----------------
1. Fetch paper PDFs via the PDF adapter (or accept raw text).
2. Split text into chunks with a LangChain text splitter.
3. Embed chunks and upsert them into the vector store.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.adapters.arxiv import ArxivAdapter, ArxivPaper
from src.adapters.pdf import PDFAdapter


@dataclass
class IngestionResult:
    paper_id: str
    title: str
    chunks_added: int


class PaperIngestionPipeline:
    """Fetch, chunk, embed, and index papers into a vector store."""

    def __init__(
        self,
        vectorstore: VectorStore,
        embeddings: Embeddings,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        self.vectorstore = vectorstore
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self._arxiv = ArxivAdapter()
        self._pdf = PDFAdapter()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_arxiv_id(self, arxiv_id: str) -> IngestionResult:
        """Fetch an arXiv paper by ID, extract its PDF text, and index it."""
        paper = self._arxiv.fetch_by_id(arxiv_id)
        text = self._pdf.from_url(paper.pdf_url)
        return self._index(text, paper)

    def ingest_text(
        self,
        text: str,
        paper_id: str,
        title: str,
        source_url: Optional[str] = None,
    ) -> IngestionResult:
        """Index arbitrary pre-extracted text."""

        class _MockPaper:
            paper_id = paper_id
            title = title
            authors: list = []
            url = source_url or ""

        return self._index(text, _MockPaper())

    def ingest_local_pdf(self, path: str, paper_id: str, title: str) -> IngestionResult:
        """Extract text from a local PDF file and index it."""

        class _MockPaper:
            paper_id = paper_id
            title = title
            authors: list = []
            url = path

        text = self._pdf.from_path(path)
        return self._index(text, _MockPaper())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _index(self, text: str, paper) -> IngestionResult:
        chunks = self._splitter.split_text(text)
        docs = [
            Document(
                page_content=chunk,
                metadata={
                    "paper_id": paper.paper_id,
                    "title": paper.title,
                    "source": paper.url,
                    "page": i,
                },
            )
            for i, chunk in enumerate(chunks)
        ]
        self.vectorstore.add_documents(docs)
        return IngestionResult(
            paper_id=paper.paper_id,
            title=paper.title,
            chunks_added=len(docs),
        )
