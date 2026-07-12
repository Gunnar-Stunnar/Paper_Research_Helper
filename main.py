"""Entry points for the Paper Research Helper.

Usage
-----
Start the MCP server (stdio transport, compatible with Claude Desktop / Cursor):
    python main.py serve

Ingest a paper from arXiv:
    python main.py ingest --arxiv-id 2301.00001

Ask a one-shot question (no MCP):
    python main.py ask "What are the key contributions of attention mechanisms?"
"""

from __future__ import annotations

import argparse
import sys


def cmd_serve(_args: argparse.Namespace) -> None:
    """Start the MCP server."""
    from src.mcp.server import create_mcp_server

    server = create_mcp_server()
    server.run(transport="stdio")


def cmd_ingest(args: argparse.Namespace) -> None:
    """Ingest a paper into the vector store."""
    import os
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    from src.pipeline.ingestion import PaperIngestionPipeline

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore_path = os.getenv("VECTORSTORE_PATH", "data/vectorstore")

    try:
        vs = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
    except Exception:
        vs = FAISS.from_texts(["init"], embeddings)

    pipeline = PaperIngestionPipeline(vectorstore=vs, embeddings=embeddings)
    result = pipeline.ingest_arxiv_id(args.arxiv_id)
    vs.save_local(vectorstore_path)
    print(f"Ingested: {result.title} ({result.chunks_added} chunks)")


def cmd_ask(args: argparse.Namespace) -> None:
    """Ask a one-shot question using the QA graph."""
    import os
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    from src.graphs.research_graph import build_research_graph

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore_path = os.getenv("VECTORSTORE_PATH", "data/vectorstore")

    try:
        vs = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
    except Exception:
        print("No vector store found. Ingest papers first with: python main.py ingest --arxiv-id <id>")
        sys.exit(1)

    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
    graph = build_research_graph(llm=llm, vectorstore=vs)
    result = graph.invoke({"query": args.question, "messages": []})
    print(result.get("final_answer", "No answer generated."))


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper Research Helper")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("serve", help="Start the MCP server (stdio)")

    ingest_p = sub.add_parser("ingest", help="Ingest a paper into the vector store")
    ingest_p.add_argument("--arxiv-id", required=True, help="arXiv paper ID, e.g. 2301.00001")

    ask_p = sub.add_parser("ask", help="Ask a one-shot question")
    ask_p.add_argument("question", help="The research question")

    args = parser.parse_args()
    {"serve": cmd_serve, "ingest": cmd_ingest, "ask": cmd_ask}[args.command](args)


if __name__ == "__main__":
    main()
