"""Anthropic Claude LLM adapter.

Wraps ``langchain-anthropic`` ``ChatAnthropic`` to provide pre-configured
Claude chat models for use with agents, graphs, and tools in this project.

Supported models
----------------
claude-opus-4-5          Most capable — best for complex reasoning tasks
claude-sonnet-4-5        Balanced performance / cost  (default)
claude-haiku-3-5         Fast and cost-efficient — good for simple lookups

Environment variables
---------------------
ANTHROPIC_API_KEY     Your Anthropic API key (required)
CLAUDE_MODEL          Model name override  (default: claude-sonnet-4-5)
CLAUDE_MAX_TOKENS     Max output tokens    (default: 4096)
CLAUDE_TEMPERATURE    Sampling temperature (default: 0)

Usage
-----
    from src.adapters.claude import ClaudeAdapter

    # Default — claude-sonnet-4-5
    llm = ClaudeAdapter().chat_model()

    # Specific model
    llm = ClaudeAdapter(model="claude-opus-4-5").chat_model()

    # Drop-in replacement for ChatOpenAI inside agents / graphs
    from src.agents.research_agent import ResearchAgent
    agent = ResearchAgent(llm=ClaudeAdapter().chat_model())
"""

from __future__ import annotations

import os
from typing import Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel


# Shorthand aliases so callers don't have to memorise full model strings
MODEL_ALIASES: dict[str, str] = {
    "opus":    "claude-opus-4-5",
    "sonnet":  "claude-sonnet-4-5",
    "haiku":   "claude-haiku-3-5",
}

DEFAULT_MODEL = "claude-sonnet-4-5"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0


class ClaudeAdapter:
    """Factory for pre-configured ``ChatAnthropic`` instances.

    Parameters
    ----------
    model:
        Full Anthropic model name **or** a shorthand alias
        (``'opus'``, ``'sonnet'``, ``'haiku'``).
        Falls back to the ``CLAUDE_MODEL`` env var, then ``claude-sonnet-4-5``.
    api_key:
        Anthropic API key. Falls back to ``ANTHROPIC_API_KEY`` env var.
    temperature:
        Sampling temperature (0 = deterministic). Falls back to
        ``CLAUDE_TEMPERATURE`` env var, then 0.
    max_tokens:
        Maximum tokens in the completion. Falls back to ``CLAUDE_MAX_TOKENS``
        env var, then 4096.
    streaming:
        When *True* the model will stream tokens. Wire with a streaming
        callback or use ``astream`` for async streaming.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        streaming: bool = False,
    ) -> None:
        raw_model = model or os.getenv("CLAUDE_MODEL", DEFAULT_MODEL)
        self._model = MODEL_ALIASES.get(raw_model, raw_model)
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._temperature = (
            temperature
            if temperature is not None
            else float(os.getenv("CLAUDE_TEMPERATURE", DEFAULT_TEMPERATURE))
        )
        self._max_tokens = (
            max_tokens
            if max_tokens is not None
            else int(os.getenv("CLAUDE_MAX_TOKENS", DEFAULT_MAX_TOKENS))
        )
        self._streaming = streaming

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat_model(self) -> BaseChatModel:
        """Return a ``ChatAnthropic`` instance ready to use with LangChain.

        The returned object is a drop-in replacement for ``ChatOpenAI``
        anywhere a ``BaseChatModel`` is accepted (agents, graphs, chains).
        """
        return ChatAnthropic(
            model=self._model,
            api_key=self._api_key,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            streaming=self._streaming,
        )

    # Convenience constructors -------------------------------------------

    @classmethod
    def opus(cls, **kwargs) -> "ClaudeAdapter":
        """Return an adapter configured for ``claude-opus-4-5``."""
        return cls(model="claude-opus-4-5", **kwargs)

    @classmethod
    def sonnet(cls, **kwargs) -> "ClaudeAdapter":
        """Return an adapter configured for ``claude-sonnet-4-5`` (default)."""
        return cls(model="claude-sonnet-4-5", **kwargs)

    @classmethod
    def haiku(cls, **kwargs) -> "ClaudeAdapter":
        """Return an adapter configured for ``claude-haiku-3-5``."""
        return cls(model="claude-haiku-3-5", **kwargs)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ClaudeAdapter(model={self._model!r}, "
            f"temperature={self._temperature}, "
            f"max_tokens={self._max_tokens})"
        )
