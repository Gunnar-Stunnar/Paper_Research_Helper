# Paper Research Helper

A LangChain + LangGraph research assistant that ingests academic papers and answers questions about them, exposed as an MCP server compatible with Cursor and Claude Desktop.

## Features

- **Paper ingestion** — fetch PDFs from arXiv by ID, extract text, chunk and embed into a local FAISS vector store.
- **Semantic search** — search arXiv and Semantic Scholar for papers by keyword.
- **QA over papers** — ask natural-language questions answered with retrieved passages from ingested papers.
- **MCP server** — expose all capabilities as tools consumable by any MCP-compatible client.

## Project Structure

```
paper_research_helper/
├── main.py                    # CLI entry point (serve / ingest / ask)
├── requirements.txt
├── .env.example
├── docs/
│   ├── architecture.md        # System design & data flow
│   └── getting_started.md    # Step-by-step setup guide
└── src/
    ├── adapters/              # External service connectors
    │   ├── arxiv.py           #   arXiv API
    │   ├── semantic_scholar.py #  Semantic Scholar API
    │   └── pdf.py             #   PDF text extraction
    ├── tools/                 # LangChain tools used by agents
    │   ├── search.py          #   Paper search tool
    │   ├── retrieval.py       #   Vector store retrieval tool
    │   └── summarize.py       #   LLM summarization tool
    ├── agents/                # LangGraph agent nodes
    │   ├── research_agent.py  #   Discovers & summarises papers
    │   └── qa_agent.py        #   RAG question-answering agent
    ├── graphs/                # LangGraph state graphs
    │   └── research_graph.py  #   Full research pipeline graph
    ├── pipeline/              # Ingestion orchestration
    │   └── ingestion.py       #   Fetch → chunk → embed → index
    └── mcp/                   # MCP server
        └── server.py          #   FastMCP server with 3 tools
```

## Quick Start

### 1. Install dependencies

```bash
uv sync           # creates .venv and installs all dependencies
```

### 2. Configure environment

```bash
cp .env.example .env
# edit .env and set OPENAI_API_KEY at minimum
```

### 3. Ingest a paper

```bash
python main.py ingest --arxiv-id 2301.07041   # Attention Is All You Need (example)
```

### 4. Ask a question

```bash
python main.py ask "What problem does the transformer architecture solve?"
```

### 5. Start the MCP server

```bash
python main.py serve
```

Then add the server to your Cursor or Claude Desktop MCP config:

```json
{
  "mcpServers": {
    "paper-research-helper": {
      "command": "python",
      "args": ["main.py", "serve"],
      "cwd": "/path/to/paper_research_helper"
    }
  }
}
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `search_papers` | Search arXiv or Semantic Scholar by keyword |
| `ingest_paper` | Ingest an arXiv paper into the local vector store |
| `ask_question` | Answer a research question using the QA graph |

## License

MIT
