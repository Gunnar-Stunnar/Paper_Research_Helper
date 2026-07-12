"""LangChain tool for semantic retrieval from the local vector store."""

from __future__ import annotations

import json
from typing import Optional, Type

from langchain_core.tools import BaseTool
from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel, Field


class RetrievalInput(BaseModel):
    query: str = Field(description="Question or topic to retrieve relevant passages for.")
    k: int = Field(default=4, description="Number of document chunks to retrieve.")


class VectorStoreRetrievalTool(BaseTool):
    """Retrieve relevant passages from the indexed paper corpus."""

    name: str = "paper_retrieval"
    description: str = (
        "Retrieve the most relevant passages from previously ingested papers "
        "based on a natural-language query."
    )
    args_schema: Type[BaseModel] = RetrievalInput

    vectorstore: VectorStore

    class Config:
        arbitrary_types_allowed = True

    def _run(self, query: str, k: int = 4) -> str:
        docs = self.vectorstore.similarity_search(query, k=k)
        results = [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page"),
            }
            for doc in docs
        ]
        return json.dumps(results, indent=2)
