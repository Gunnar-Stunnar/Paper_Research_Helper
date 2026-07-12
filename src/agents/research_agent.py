"""Research agent — discovers and fetches relevant papers for a given topic."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from src.tools.search import PaperSearchTool
from src.tools.summarize import SummarizeTool


_SYSTEM_PROMPT = """\
You are a research assistant specialised in finding and summarising academic papers.
Given a research question or topic, you will:
1. Search for relevant papers using the paper_search tool.
2. Summarise the most relevant results using the summarize_text tool.
3. Return a structured overview of the key findings.

Always cite paper titles and authors in your response.
"""


class ResearchAgent:
    """Wraps a ReAct agent configured with paper-search and summarization tools."""

    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm
        tools = [
            PaperSearchTool(),
            SummarizeTool(llm=llm),
        ]
        self._agent = create_react_agent(llm, tools, prompt=_SYSTEM_PROMPT)

    def run(self, query: str) -> str:
        """Run the research agent on *query* and return a text summary."""
        result = self._agent.invoke({"messages": [HumanMessage(content=query)]})
        messages = result.get("messages", [])
        return messages[-1].content if messages else ""
