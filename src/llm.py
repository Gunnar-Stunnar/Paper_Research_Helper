"""Central LLM and embeddings factory for the research assistant.

Bedrock is the primary provider. If ``BEDROCK_API_KEY`` or AWS credentials
are available, both the chat model and embeddings use Bedrock automatically.

Provider resolution order (when *provider* is ``'auto'``)
---------------------------------------------------------
1. Bedrock   — when ``BEDROCK_API_KEY`` **or** ``AWS_ACCESS_KEY_ID``
               **or** ``AWS_PROFILE`` is set in the environment.
2. Anthropic — when ``ANTHROPIC_API_KEY`` is set.
               (embeddings fall through to OpenAI; Anthropic has no embeddings API)
3. OpenAI    — fallback when ``OPENAI_API_KEY`` is set.

Usage
-----
    from src.llm import get_llm, get_embeddings

    llm        = get_llm()         # auto-selects Bedrock if configured
    embeddings = get_embeddings()  # matches the same provider

    # Explicit provider override
    llm        = get_llm(provider="bedrock")
    embeddings = get_embeddings(provider="openai")
"""

from __future__ import annotations

import os
from typing import Literal

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel


Provider = Literal["auto", "bedrock", "anthropic", "openai"]

# Bedrock Titan embedding model (requires IAM credentials, not just API key)
_BEDROCK_EMBED_MODEL = os.getenv(
    "BEDROCK_EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0"
)

# Local sentence-transformers model (no API key needed, runs on-device)
_LOCAL_EMBED_MODEL = os.getenv("LOCAL_EMBED_MODEL", "all-MiniLM-L6-v2")


# ---------------------------------------------------------------------------
# Availability checks
# ---------------------------------------------------------------------------

def _bedrock_available() -> bool:
    """True when any Bedrock auth method is present (API key OR IAM)."""
    return bool(
        os.getenv("BEDROCK_API_KEY")
        or os.getenv("AWS_ACCESS_KEY_ID")
        or os.getenv("AWS_PROFILE")
    )


def _bedrock_iam_available() -> bool:
    """True only when real IAM credentials are present.

    BedrockEmbeddings does NOT support the Bedrock API key — it requires IAM.
    Use this check before creating BedrockEmbeddings.
    """
    return bool(os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("AWS_PROFILE"))


def _anthropic_available() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def _openai_available() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def _resolve_provider(provider: Provider) -> str:
    if provider != "auto":
        return provider
    if _bedrock_available():
        return "bedrock"
    if _anthropic_available():
        return "anthropic"
    if _openai_available():
        return "openai"
    return "none"


# ---------------------------------------------------------------------------
# Chat model factory
# ---------------------------------------------------------------------------

def get_llm(provider: Provider = "auto") -> BaseChatModel:
    """Return a ``BaseChatModel`` for the selected or auto-detected provider.

    Parameters
    ----------
    provider:
        ``'auto'``      — detect from environment (Bedrock → Anthropic → OpenAI)
        ``'bedrock'``   — AWS Bedrock (requires ``BEDROCK_API_KEY`` or IAM creds)
        ``'anthropic'`` — direct Anthropic API (requires ``ANTHROPIC_API_KEY``)
        ``'openai'``    — OpenAI (requires ``OPENAI_API_KEY``)

    Raises
    ------
    RuntimeError
        When no credentials are available for the resolved provider.
    """
    resolved = _resolve_provider(provider)

    if resolved == "bedrock":
        return _make_bedrock_llm()
    if resolved == "anthropic":
        return _make_anthropic_llm()
    if resolved == "openai":
        return _make_openai_llm()

    raise RuntimeError(
        "No LLM credentials found. Set one of: BEDROCK_API_KEY, "
        "AWS_ACCESS_KEY_ID, ANTHROPIC_API_KEY, or OPENAI_API_KEY."
    )


def _make_bedrock_llm() -> BaseChatModel:
    from src.adapters.bedrock import BedrockAdapter

    if not _bedrock_available():
        raise RuntimeError(
            "Bedrock provider selected but no credentials found. "
            "Set BEDROCK_API_KEY, AWS_ACCESS_KEY_ID, or AWS_PROFILE."
        )
    return BedrockAdapter().chat_model()


def _make_anthropic_llm() -> BaseChatModel:
    from src.adapters.claude import ClaudeAdapter

    if not _anthropic_available():
        raise RuntimeError(
            "Anthropic provider selected but ANTHROPIC_API_KEY is not set."
        )
    return ClaudeAdapter().chat_model()


def _make_openai_llm() -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    if not _openai_available():
        raise RuntimeError(
            "OpenAI provider selected but OPENAI_API_KEY is not set."
        )
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
    )


# ---------------------------------------------------------------------------
# Embeddings factory
# ---------------------------------------------------------------------------

def get_embeddings(provider: Provider = "auto") -> Embeddings:
    """Return an ``Embeddings`` instance for the selected or auto-detected provider.

    Embedding provider resolution
    ------------------------------
    Bedrock (IAM)  — ``BedrockEmbeddings`` with Titan Embed v2.
                     Requires ``AWS_ACCESS_KEY_ID`` or ``AWS_PROFILE``.
                     The Bedrock API key alone is NOT sufficient here — AWS
                     does not support Bearer-token auth for ``InvokeModel``
                     embeddings calls.
    OpenAI         — ``OpenAIEmbeddings`` (text-embedding-3-small).
    Local          — ``HuggingFaceEmbeddings`` running on-device via
                     ``sentence-transformers``. No API key required.
                     Used automatically when Bedrock IAM and OpenAI are both
                     unavailable (e.g. Bedrock API-key-only setups).

    Override the local model with the ``LOCAL_EMBED_MODEL`` env var
    (default: ``all-MiniLM-L6-v2``).
    """
    resolved = _resolve_provider(provider)

    if resolved == "bedrock":
        if _bedrock_iam_available():
            return _make_bedrock_embeddings()
        # API-key-only Bedrock: fall back to local embeddings
        print(
            "[llm] BedrockEmbeddings requires IAM credentials; "
            f"falling back to local model '{_LOCAL_EMBED_MODEL}'."
        )
        return _make_local_embeddings()

    if resolved == "openai":
        return _make_openai_embeddings()

    if resolved == "anthropic":
        # Anthropic has no embeddings API — use local
        return _make_local_embeddings()

    # Last resort: local on-device embeddings need no credentials at all
    return _make_local_embeddings()


def _make_bedrock_embeddings() -> Embeddings:
    from langchain_aws import BedrockEmbeddings
    import boto3

    session_kwargs: dict = {"region_name": os.getenv("BEDROCK_REGION", "us-east-1")}
    profile = os.getenv("AWS_PROFILE")
    if profile:
        session_kwargs["profile_name"] = profile

    session = boto3.Session(**session_kwargs)
    client = session.client("bedrock-runtime")

    return BedrockEmbeddings(model_id=_BEDROCK_EMBED_MODEL, client=client)


def _make_openai_embeddings() -> Embeddings:
    from langchain_openai import OpenAIEmbeddings

    if not _openai_available():
        raise RuntimeError(
            "OpenAI embeddings selected but OPENAI_API_KEY is not set."
        )
    return OpenAIEmbeddings(model="text-embedding-3-small")


def _make_local_embeddings() -> Embeddings:
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=_LOCAL_EMBED_MODEL)
