# Architecture

## Overview

Paper Research Helper is built around three layers:

1. **Adapters** — thin wrappers around external APIs and file formats.
2. **Tools + Agents** — LangChain tools composed into LangGraph agent nodes.
3. **Graphs** — LangGraph state graphs that orchestrate multi-step research workflows.

An **MCP server** sits on top and exposes the graphs as callable tools to any MCP-compatible client (Cursor, Claude Desktop, etc.).

---

## Data Flow

### Ingestion

```
arXiv ID / local PDF
       │
       ▼
  PDFAdapter          ← downloads & extracts raw text
       │
       ▼
  RecursiveCharacterTextSplitter   ← splits into overlapping chunks
       │
       ▼
  OpenAIEmbeddings    ← embeds each chunk
       │
       ▼
  FAISS VectorStore   ← persisted to disk at VECTORSTORE_PATH
```

### Query

```
User question (via MCP tool or CLI)
       │
       ▼
  ResearchGraph (LangGraph)
  ┌────────────────────────────────────────┐
  │  [search_node]                         │
  │   └─ ResearchAgent (ReAct)             │
  │       ├─ PaperSearchTool → arXiv API   │
  │       └─ SummarizeTool → LLM           │
  │                                        │
  │  [answer_node]                         │
  │   └─ QAAgent (ReAct)                   │
  │       ├─ VectorStoreRetrievalTool      │
  │       │    └─ FAISS similarity search  │
  │       └─ SummarizeTool → LLM           │
  └────────────────────────────────────────┘
       │
       ▼
  final_answer → returned to MCP client / CLI
```

---

## Key Components

### `src/adapters/`

| File | Responsibility |
|------|---------------|
| `arxiv.py` | Wraps the `arxiv` Python client; returns `ArxivPaper` dataclasses |
| `semantic_scholar.py` | Calls Semantic Scholar REST API; returns `S2Paper` dataclasses |
| `pdf.py` | Downloads PDFs from URLs or reads local files; returns plain text |

### `src/tools/`

LangChain `BaseTool` subclasses. Agents can call these during their ReAct loop.

| Tool | Input | Output |
|------|-------|--------|
| `PaperSearchTool` | query, source, max_results | JSON list of papers |
| `VectorStoreRetrievalTool` | query, k | JSON list of passages with metadata |
| `SummarizeTool` | text, max_words | Plain-text summary |

### `src/agents/`

Each agent wraps a `langgraph.prebuilt.create_react_agent` with a specific system prompt and tool set.

| Agent | Tools | Purpose |
|-------|-------|---------|
| `ResearchAgent` | search + summarize | Explore the literature |
| `QAAgent` | retrieval + summarize | Answer grounded in indexed corpus |

### `src/graphs/`

`research_graph.py` compiles a `StateGraph` with the following nodes:

```
search_node → answer_node → END
```

State is typed (`ResearchState`) and uses LangGraph's `add_messages` reducer for message history.

### `src/pipeline/`

`PaperIngestionPipeline` is the single point of entry for adding papers. It handles the full fetch → chunk → embed → upsert lifecycle.

### `src/mcp/`

`FastMCP` server (from the `mcp` SDK) that wraps the graph and pipeline behind three named tools. Run with `python main.py serve` (stdio transport).

---

## Technology Choices

| Concern | Choice | Rationale |
|---------|--------|-----------|
| LLM orchestration | LangGraph | Native support for typed state, conditional edges, and human-in-the-loop |
| Embeddings | OpenAI `text-embedding-3-small` | Strong quality-to-cost ratio |
| Vector store | FAISS (local) | Zero-infrastructure; swap for Pinecone/Weaviate for production |
| MCP server | `mcp[cli]` FastMCP | Minimal boilerplate, stdio transport works out-of-the-box with Cursor |
