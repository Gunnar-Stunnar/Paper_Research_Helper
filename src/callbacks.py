"""Console logging callbacks for LangChain agents and graphs.

Provides a ``ConsoleCallbackHandler`` that prints every significant event
— graph node transitions, LLM prompts and responses, tool calls and results —
to the terminal using rich formatting.

Usage
-----
    from src.callbacks import ConsoleCallbackHandler

    handler = ConsoleCallbackHandler()

    # Pass to any LangChain / LangGraph invocation
    graph.invoke(state, config={"callbacks": [handler]})
    agent.invoke({"messages": [...]}, config={"callbacks": [handler]})
"""

from __future__ import annotations

import textwrap
from typing import Any, Union
from uuid import UUID

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text


_console = Console(stderr=False, highlight=False)

# ANSI-safe colour tokens that work in any terminal
_COLORS = {
    "node":   "bold cyan",
    "llm_in": "dim white",
    "llm_out":"bold green",
    "tool_in":"bold yellow",
    "tool_out":"bold blue",
    "error":  "bold red",
    "info":   "dim white",
}

_MAX_CONTENT_LEN = 1200  # truncate very long strings in the log


def _trim(text: str, limit: int = _MAX_CONTENT_LEN) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n  … [{len(text) - limit} chars truncated]"


def _msgs_to_str(messages: list[list[BaseMessage]]) -> str:
    lines = []
    for batch in messages:
        for m in batch:
            role = type(m).__name__.replace("Message", "").upper()
            content = m.content if isinstance(m.content, str) else str(m.content)
            lines.append(f"[{role}] {_trim(content, 400)}")
    return "\n".join(lines)


class ConsoleCallbackHandler(BaseCallbackHandler):
    """Prints a structured log of every agent/tool/LLM event to stdout.

    Instantiate once and pass via ``config={"callbacks": [handler]}`` to any
    LangChain or LangGraph invocation.
    """

    raise_error: bool = True  # re-raise errors after logging

    # ------------------------------------------------------------------
    # Chain / graph node events
    # ------------------------------------------------------------------

    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        if not serialized:
            return
        name = serialized.get("name") or (serialized.get("id") or ["?"])[-1]
        # Filter out noisy internal chain names; surface graph nodes clearly
        skip = {"RunnableSequence", "RunnableLambda", "RunnableParallel",
                "ChannelWrite", "ChannelRead", "RunnableCallable"}
        if name in skip:
            return
        _console.print(Rule(f"[{_COLORS['node']}]▶  {name}[/]", style="cyan"))

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        pass  # keep output clean; individual events already logged

    # ------------------------------------------------------------------
    # LLM events
    # ------------------------------------------------------------------

    def on_llm_start(
        self,
        serialized: dict[str, Any] | None,
        prompts: list[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        model = (serialized or {}).get("kwargs", {}).get("model_id") or (serialized or {}).get("name", "LLM")
        combined = "\n---\n".join(_trim(p, 600) for p in prompts)
        _console.print(Panel(
            combined,
            title=f"[{_COLORS['llm_in']}]🧠  {model} — prompt[/]",
            border_style="dim white",
            padding=(0, 1),
        ))

    def on_chat_model_start(
        self,
        serialized: dict[str, Any] | None,
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        s = serialized or {}
        model = (
            s.get("kwargs", {}).get("model_id")
            or s.get("kwargs", {}).get("model")
            or s.get("name", "LLM")
        )
        content = _msgs_to_str(messages)
        _console.print(Panel(
            _trim(content),
            title=f"[{_COLORS['llm_in']}]🧠  {model} — input[/]",
            border_style="dim white",
            padding=(0, 1),
        ))

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        for gen_list in response.generations:
            for gen in gen_list:
                text = getattr(gen, "text", None) or getattr(gen, "message", None)
                if text is None:
                    continue
                if hasattr(text, "content"):
                    text = text.content
                usage = response.llm_output or {}
                token_info = ""
                if usage:
                    tu = usage.get("token_usage") or usage.get("usage") or {}
                    if tu:
                        token_info = (
                            f"  in={tu.get('input_tokens') or tu.get('prompt_tokens', '?')} "
                            f"out={tu.get('output_tokens') or tu.get('completion_tokens', '?')}"
                        )
                _console.print(Panel(
                    _trim(str(text)),
                    title=f"[{_COLORS['llm_out']}]✅  LLM response{token_info}[/]",
                    border_style="green",
                    padding=(0, 1),
                ))

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        _console.print(f"[{_COLORS['error']}]❌  LLM error:[/] {error}")

    # ------------------------------------------------------------------
    # Tool events
    # ------------------------------------------------------------------

    def on_tool_start(
        self,
        serialized: dict[str, Any] | None,
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        name = (serialized or {}).get("name", "tool")
        _console.print(Panel(
            _trim(input_str, 600),
            title=f"[{_COLORS['tool_in']}]🔧  Tool call: {name}[/]",
            border_style="yellow",
            padding=(0, 1),
        ))

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        _console.print(Panel(
            _trim(str(output), 800),
            title=f"[{_COLORS['tool_out']}]📤  Tool result[/]",
            border_style="blue",
            padding=(0, 1),
        ))

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        _console.print(f"[{_COLORS['error']}]❌  Tool error:[/] {error}")

    # ------------------------------------------------------------------
    # Agent events
    # ------------------------------------------------------------------

    def on_agent_action(self, action: Any, *, run_id: UUID, **kwargs: Any) -> None:
        _console.print(
            f"[{_COLORS['tool_in']}]🤖  Agent → tool:[/] {action.tool}  "
            f"[{_COLORS['info']}]input:[/] {_trim(str(action.tool_input), 300)}"
        )

    def on_agent_finish(self, finish: Any, *, run_id: UUID, **kwargs: Any) -> None:
        output = finish.return_values.get("output", "")
        _console.print(Panel(
            _trim(str(output)),
            title=f"[{_COLORS['llm_out']}]🏁  Agent finished[/]",
            border_style="green",
            padding=(0, 1),
        ))
