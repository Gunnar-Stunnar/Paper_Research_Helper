"""QA agent — answers questions grounded in the indexed paper corpus."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.vectorstores import VectorStore
from langgraph.prebuilt import create_react_agent

from src.tools.retrieval import VectorStoreRetrievalTool
from src.tools.summarize import SummarizeTool


_SYSTEM_PROMPT = """\
You are an academic research assistant. Answer the user's question using ONLY
information retrieved from the indexed paper corpus via the paper_retrieval tool.
If the answer cannot be found in the retrieved passages, say so clearly.
Always cite the source document for every claim you make.
"""


class QAAgent:
    """Retrieval-augmented QA agent backed by a local vector store."""

    def __init__(self, llm: BaseChatModel, vectorstore: VectorStore) -> None:
        self.llm = llm
        tools = [
            VectorStoreRetrievalTool(vectorstore=vectorstore),
            SummarizeTool(llm=llm),
        ]
        self._agent = create_react_agent(llm, tools, prompt=_SYSTEM_PROMPT)

    def run(self, question: str) -> str:
        """Answer *question* using passages from the indexed corpus."""
        result = self._agent.invoke({"messages": [HumanMessage(content=question)]})
        messages = result.get("messages", [])
        return messages[-1].content if messages else ""
