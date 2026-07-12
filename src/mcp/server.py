"""MCP server that exposes the research assistant over the Model Context Protocol.

Exposed tools
-------------
* ``search_papers``   — search arXiv / Semantic Scholar for papers.
* ``ask_question``    — ask a question answered by the QA graph.
* ``ingest_paper``    — ingest an arXiv paper by ID into the vector store.

Run with:
    python -m src.mcp.server
"""

from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

from src.adapters.arxiv import ArxivAdapter
from src.graphs.research_graph import build_research_graph
from src.pipeline.ingestion import PaperIngestionPipeline


# ---------------------------------------------------------------------------
# Lazy singletons — initialised on first use to keep import time fast
# ---------------------------------------------------------------------------

_graph = None
_pipeline = None


def _get_graph():
    global _graph
    if _graph is None:
        from langchain_openai import ChatOpenAI
        from langchain_openai import OpenAIEmbeddings
        from langchain_community.vectorstores import FAISS

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = FAISS.load_local(
            os.getenv("VECTORSTORE_PATH", "data/vectorstore"),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
        _graph = build_research_graph(llm=llm, vectorstore=vectorstore)
    return _graph


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from langchain_community.vectorstores import FAISS

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore_path = os.getenv("VECTORSTORE_PATH", "data/vectorstore")

        try:
            vectorstore = FAISS.load_local(
                vectorstore_path,
                embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception:
            # Create a fresh store if none exists yet
            vectorstore = FAISS.from_texts(["init"], embeddings)
            vectorstore.save_local(vectorstore_path)

        _pipeline = PaperIngestionPipeline(vectorstore=vectorstore, embeddings=embeddings)
    return _pipeline


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

def create_mcp_server() -> FastMCP:
    mcp = FastMCP("paper-research-helper")

    @mcp.tool()
    def search_papers(query: str, source: str = "arxiv", max_results: int = 5) -> str:
        """Search for academic papers on arXiv or Semantic Scholar.

        Args:
            query: Natural-language search query.
            source: 'arxiv' or 'semantic_scholar'.
            max_results: Maximum number of papers to return.
        """
        adapter = ArxivAdapter(max_results=max_results)
        papers = adapter.search(query)
        lines = [
            f"- **{p.title}** ({', '.join(p.authors[:2])}{'et al.' if len(p.authors) > 2 else ''})\n"
            f"  {p.abstract[:300]}...\n  URL: {p.url}"
            for p in papers
        ]
        return "\n\n".join(lines) if lines else "No results found."

    @mcp.tool()
    def ask_question(question: str) -> str:
        """Ask a question answered by the research QA graph.

        The graph retrieves relevant passages from the indexed paper corpus
        and synthesises a grounded answer.

        Args:
            question: The research question to answer.
        """
        graph = _get_graph()
        result = graph.invoke({"query": question, "messages": []})
        return result.get("final_answer", "No answer generated.")

    @mcp.tool()
    def ingest_paper(arxiv_id: str) -> str:
        """Ingest an arXiv paper into the local vector store by its arXiv ID.

        Args:
            arxiv_id: arXiv paper identifier, e.g. '2301.00001'.
        """
        pipeline = _get_pipeline()
        result = pipeline.ingest_arxiv_id(arxiv_id)
        return (
            f"Ingested '{result.title}' ({result.paper_id}). "
            f"Added {result.chunks_added} chunks to the vector store."
        )

    return mcp


if __name__ == "__main__":
    server = create_mcp_server()
    server.run(transport="stdio")
