"""LangGraph state graph that orchestrates the full research pipeline.

Flow
----
  user_query
      │
      ▼
  [search_node]  ── searches arXiv / Semantic Scholar
      │
      ▼
  [retrieve_node]  ── retrieves relevant passages from vector store
      │
      ▼
  [answer_node]  ── synthesises a final answer using the QA agent
      │
      ▼
  END
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.vectorstores import VectorStore
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from src.agents.research_agent import ResearchAgent
from src.agents.qa_agent import QAAgent


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class ResearchState(TypedDict):
    """Shared state passed between graph nodes."""

    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    search_results: str
    retrieved_passages: str
    final_answer: str


# ---------------------------------------------------------------------------
# Node builders
# ---------------------------------------------------------------------------

def _build_search_node(research_agent: ResearchAgent):
    def search_node(state: ResearchState) -> dict:
        results = research_agent.run(state["query"])
        return {"search_results": results}

    return search_node


def _build_answer_node(qa_agent: QAAgent):
    def answer_node(state: ResearchState) -> dict:
        answer = qa_agent.run(state["query"])
        return {"final_answer": answer}

    return answer_node


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------

def build_research_graph(
    llm: BaseChatModel,
    vectorstore: VectorStore,
) -> StateGraph:
    """Compile and return the full research LangGraph."""

    research_agent = ResearchAgent(llm=llm)
    qa_agent = QAAgent(llm=llm, vectorstore=vectorstore)

    graph = StateGraph(ResearchState)

    graph.add_node("search", _build_search_node(research_agent))
    graph.add_node("answer", _build_answer_node(qa_agent))

    graph.set_entry_point("search")
    graph.add_edge("search", "answer")
    graph.add_edge("answer", END)

    return graph.compile()
