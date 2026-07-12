"""Entry points for the Paper Research Helper.

Usage
-----
Start the MCP server (stdio transport, compatible with Claude Desktop / Cursor):
    python main.py serve

Ingest a paper from arXiv:
    python main.py ingest --arxiv-id 2301.00001

Ask a one-shot question (no MCP):
    python main.py ask "What are the key contributions of attention mechanisms?"

Add --verbose / -v to any command to see every agent step, tool call, and LLM
response printed to the console in real time.
"""

from __future__ import annotations

import argparse
import logging
import sys

from dotenv import load_dotenv

load_dotenv()  # loads .env into os.environ before any command runs


def _setup_logging(verbose: bool) -> None:
    """Configure Python logging and return whether verbose mode is on."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    # Silence noisy third-party loggers even in verbose mode
    for noisy in ("httpx", "httpcore", "urllib3", "botocore", "boto3",
                  "sentence_transformers", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _callbacks(verbose: bool) -> list:
    if not verbose:
        return []
    from src.callbacks import ConsoleCallbackHandler
    return [ConsoleCallbackHandler()]


def cmd_serve(args: argparse.Namespace) -> None:
    """Start the MCP server."""
    _setup_logging(args.verbose)
    from src.mcp.server import create_mcp_server

    server = create_mcp_server()
    server.run(transport="stdio")


def cmd_ingest(args: argparse.Namespace) -> None:
    """Ingest a paper into the vector store."""
    import os
    _setup_logging(args.verbose)
    from langchain_community.vectorstores import FAISS
    from src.pipeline.ingestion import PaperIngestionPipeline
    from src.llm import get_embeddings

    embeddings = get_embeddings()
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
    _setup_logging(args.verbose)
    from langchain_community.vectorstores import FAISS
    from src.graphs.research_graph import build_research_graph
    from src.llm import get_llm, get_embeddings

    embeddings = get_embeddings()
    vectorstore_path = os.getenv("VECTORSTORE_PATH", "data/vectorstore")

    try:
        vs = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
    except Exception:
        print("No vector store found. Ingest papers first with: python main.py ingest --arxiv-id <id>")
        sys.exit(1)

    llm = get_llm()
    graph = build_research_graph(llm=llm, vectorstore=vs)
    result = graph.invoke(
        {"query": args.question, "messages": []},
        config={"callbacks": _callbacks(args.verbose)},
    )
    print(result.get("final_answer", "No answer generated."))


def _add_verbose(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Print every agent step, tool call, and LLM response to the console.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper Research Helper")
    sub = parser.add_subparsers(dest="command", required=True)

    serve_p = sub.add_parser("serve", help="Start the MCP server (stdio)")
    _add_verbose(serve_p)

    ingest_p = sub.add_parser("ingest", help="Ingest a paper into the vector store")
    ingest_p.add_argument("--arxiv-id", required=True, help="arXiv paper ID, e.g. 2301.00001")
    _add_verbose(ingest_p)

    ask_p = sub.add_parser("ask", help="Ask a one-shot question")
    ask_p.add_argument("question", help="The research question")
    _add_verbose(ask_p)

    args = parser.parse_args()
    {"serve": cmd_serve, "ingest": cmd_ingest, "ask": cmd_ask}[args.command](args)


if __name__ == "__main__":
    main()
