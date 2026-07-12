# Getting Started

## Prerequisites

- Python 3.11+
- An OpenAI API key (for embeddings and the LLM)
- (Optional) A Semantic Scholar API key for higher rate limits

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/your-org/paper_research_helper.git
cd paper_research_helper

# 2. Install dependencies (uv creates .venv automatically)
uv sync

# 3. Add a new dependency any time with:
# uv add <package>

# 4. Configure environment variables
cp .env.example .env
# Open .env and set OPENAI_API_KEY (required) and optionally other vars
```

---

## Ingesting Papers

Ingest a paper from arXiv by its ID:

```bash
python main.py ingest --arxiv-id 1706.03762   # "Attention Is All You Need"
python main.py ingest --arxiv-id 2005.11401   # RAG paper
```

The pipeline downloads the PDF, splits it into chunks, embeds them, and saves the FAISS index to `data/vectorstore/` (configurable via `VECTORSTORE_PATH`).

---

## Asking Questions

One-shot question answering from the command line:

```bash
python main.py ask "How does self-attention differ from cross-attention?"
python main.py ask "What datasets were used to evaluate RAG models?"
```

---

## Running the MCP Server

Start the server in stdio mode (required for Cursor/Claude Desktop integration):

```bash
python main.py serve
```

### Connecting to Cursor

Add the following to your Cursor MCP config (`~/.cursor/mcp.json` or the project-level `.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "paper-research-helper": {
      "command": "python",
      "args": ["main.py", "serve"],
      "cwd": "/absolute/path/to/paper_research_helper",
      "env": {
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

Once connected you can call the MCP tools directly from Cursor chat:
- `search_papers` — find papers by keyword
- `ingest_paper` — add a paper to your knowledge base
- `ask_question` — get answers grounded in your ingested papers

---

## Extending the Project

| Goal | Where to look |
|------|---------------|
| Add a new data source | Create a new adapter in `src/adapters/` |
| Add a new capability | Create a new tool in `src/tools/`, wire it into an agent |
| Change the reasoning flow | Edit or add a graph in `src/graphs/` |
| Expose a new MCP tool | Add a `@mcp.tool()` decorated function in `src/mcp/server.py` |
| Swap the vector store | Replace FAISS imports with your preferred store (Pinecone, Chroma, etc.) |
