"""LangChain tool for summarizing a block of text (e.g. a paper abstract or section)."""

from __future__ import annotations

from typing import Type

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


_SUMMARIZE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a research assistant. Summarize the following academic text "
            "concisely, preserving key findings, methods, and contributions.",
        ),
        ("human", "{text}"),
    ]
)


class SummarizeInput(BaseModel):
    text: str = Field(description="The text to summarize (abstract, section, or full paper).")
    max_words: int = Field(default=150, description="Approximate maximum word count for the summary.")


class SummarizeTool(BaseTool):
    """Summarize a passage or paper using an LLM."""

    name: str = "summarize_text"
    description: str = (
        "Summarize a block of academic text — an abstract, section, or full paper. "
        "Returns a concise summary preserving key findings."
    )
    args_schema: Type[BaseModel] = SummarizeInput

    llm: BaseChatModel

    class Config:
        arbitrary_types_allowed = True

    def _run(self, text: str, max_words: int = 150) -> str:
        prompt = _SUMMARIZE_PROMPT.format_messages(text=text)
        response = self.llm.invoke(prompt)
        return response.content
