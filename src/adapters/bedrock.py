"""AWS Bedrock LLM adapter.

Wraps ``langchain-aws`` ``ChatBedrock`` to provide pre-configured foundation
models from AWS Bedrock for use with agents, graphs, and tools in this project.

Bedrock credential resolution (standard boto3 chain, in order)
--------------------------------------------------------------
1. Explicit ``aws_access_key_id`` / ``aws_secret_access_key`` constructor args
2. ``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY`` environment variables
3. ``~/.aws/credentials`` file (named profile via ``AWS_PROFILE`` or ``aws_profile``)
4. IAM instance profile / ECS task role / EKS pod identity (when running on AWS)

Supported model families
------------------------
Anthropic Claude   anthropic.claude-*
Amazon Nova        amazon.nova-*
Amazon Titan       amazon.titan-*
Meta Llama         meta.llama*
Mistral            mistral.*

Use cross-region inference profile IDs (``us.`` / ``eu.`` prefix) when you
need higher throughput limits — e.g. ``us.anthropic.claude-sonnet-4-5-20251101-v1:0``.

Environment variables
---------------------
BEDROCK_MODEL_ID      Bedrock model ID or inference profile ID
                      (default: us.anthropic.claude-sonnet-4-5-20251101-v1:0)
BEDROCK_REGION        AWS region for the Bedrock endpoint (default: us-east-1)
AWS_PROFILE           Named boto3 profile to use
BEDROCK_MAX_TOKENS    Max output tokens                   (default: 4096)
BEDROCK_TEMPERATURE   Sampling temperature                (default: 0)

Usage
-----
    from src.adapters.bedrock import BedrockAdapter

    # Default — Claude Sonnet via cross-region inference profile
    llm = BedrockAdapter().chat_model()

    # Specific model
    llm = BedrockAdapter(model_id="amazon.nova-pro-v1:0").chat_model()

    # Named convenience constructors
    llm = BedrockAdapter.claude_sonnet().chat_model()
    llm = BedrockAdapter.claude_haiku().chat_model()
    llm = BedrockAdapter.nova_pro().chat_model()
    llm = BedrockAdapter.llama3().chat_model()

    # Plug into any agent or graph
    from src.agents.research_agent import ResearchAgent
    agent = ResearchAgent(llm=BedrockAdapter.claude_sonnet().chat_model())
"""

from __future__ import annotations

import os
from typing import Optional

import boto3
from langchain_aws import ChatBedrock
from langchain_core.language_models import BaseChatModel


# ---------------------------------------------------------------------------
# Model catalogue — well-known IDs for convenience constructors
# ---------------------------------------------------------------------------

class ModelID:
    """Well-known Bedrock model and inference-profile IDs."""

    # Anthropic Claude — cross-region inference profiles (higher throughput)
    CLAUDE_OPUS         = "us.anthropic.claude-opus-4-5-20251101-v1:0"
    CLAUDE_SONNET       = "us.anthropic.claude-sonnet-4-5-20251101-v1:0"
    CLAUDE_HAIKU        = "us.anthropic.claude-haiku-3-5-20251212-v1:0"

    # Amazon Nova
    NOVA_PRO            = "amazon.nova-pro-v1:0"
    NOVA_LITE           = "amazon.nova-lite-v1:0"
    NOVA_MICRO          = "amazon.nova-micro-v1:0"

    # Amazon Titan
    TITAN_TEXT_EXPRESS  = "amazon.titan-text-express-v1"
    TITAN_TEXT_LITE     = "amazon.titan-text-lite-v1"

    # Meta Llama
    LLAMA3_70B          = "meta.llama3-70b-instruct-v1:0"
    LLAMA3_8B           = "meta.llama3-8b-instruct-v1:0"

    # Mistral
    MISTRAL_LARGE       = "mistral.mistral-large-2402-v1:0"
    MISTRAL_7B          = "mistral.mistral-7b-instruct-v0:2"


DEFAULT_MODEL_ID    = ModelID.CLAUDE_SONNET
DEFAULT_REGION      = "us-east-1"
DEFAULT_MAX_TOKENS  = 4096
DEFAULT_TEMPERATURE = 0


class BedrockAdapter:
    """Factory for pre-configured ``ChatBedrock`` instances.

    Parameters
    ----------
    model_id:
        Bedrock model ID or cross-region inference profile ID.
        Falls back to the ``BEDROCK_MODEL_ID`` env var, then
        ``us.anthropic.claude-sonnet-4-5-20251101-v1:0``.
    region:
        AWS region for the Bedrock endpoint. Falls back to
        ``BEDROCK_REGION`` env var, then ``us-east-1``.
    aws_profile:
        Named boto3 credentials profile. Falls back to ``AWS_PROFILE`` env var.
    aws_access_key_id / aws_secret_access_key / aws_session_token:
        Explicit AWS credentials. When provided these take precedence over
        the profile and environment variables.
    temperature:
        Sampling temperature. Falls back to ``BEDROCK_TEMPERATURE``, then 0.
    max_tokens:
        Max completion tokens. Falls back to ``BEDROCK_MAX_TOKENS``, then 4096.
    streaming:
        Enable token streaming.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        region: Optional[str] = None,
        aws_profile: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        streaming: bool = False,
    ) -> None:
        self._model_id = model_id or os.getenv("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID)
        self._region = region or os.getenv("BEDROCK_REGION", DEFAULT_REGION)
        self._temperature = (
            temperature
            if temperature is not None
            else float(os.getenv("BEDROCK_TEMPERATURE", DEFAULT_TEMPERATURE))
        )
        self._max_tokens = (
            max_tokens
            if max_tokens is not None
            else int(os.getenv("BEDROCK_MAX_TOKENS", DEFAULT_MAX_TOKENS))
        )
        self._streaming = streaming

        # Build a boto3 session so credential resolution is explicit and
        # testable rather than relying on implicit global state.
        session_kwargs: dict = {"region_name": self._region}
        profile = aws_profile or os.getenv("AWS_PROFILE")
        if profile:
            session_kwargs["profile_name"] = profile

        session = boto3.Session(**session_kwargs)

        # Override with explicit credentials if provided
        if aws_access_key_id and aws_secret_access_key:
            self._client = session.client(
                "bedrock-runtime",
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,
            )
        else:
            self._client = session.client("bedrock-runtime")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat_model(self) -> BaseChatModel:
        """Return a ``ChatBedrock`` instance ready to use with LangChain.

        The returned object is a drop-in replacement for ``ChatOpenAI`` or
        ``ChatAnthropic`` anywhere a ``BaseChatModel`` is accepted.
        """
        return ChatBedrock(
            model_id=self._model_id,
            client=self._client,
            model_kwargs={
                "temperature": self._temperature,
                "max_tokens": self._max_tokens,
            },
            streaming=self._streaming,
        )

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def claude_opus(cls, **kwargs) -> "BedrockAdapter":
        """Claude Opus 4.5 via cross-region inference profile."""
        return cls(model_id=ModelID.CLAUDE_OPUS, **kwargs)

    @classmethod
    def claude_sonnet(cls, **kwargs) -> "BedrockAdapter":
        """Claude Sonnet 4.5 via cross-region inference profile (default)."""
        return cls(model_id=ModelID.CLAUDE_SONNET, **kwargs)

    @classmethod
    def claude_haiku(cls, **kwargs) -> "BedrockAdapter":
        """Claude Haiku 3.5 via cross-region inference profile."""
        return cls(model_id=ModelID.CLAUDE_HAIKU, **kwargs)

    @classmethod
    def nova_pro(cls, **kwargs) -> "BedrockAdapter":
        """Amazon Nova Pro — strong multimodal reasoning."""
        return cls(model_id=ModelID.NOVA_PRO, **kwargs)

    @classmethod
    def nova_lite(cls, **kwargs) -> "BedrockAdapter":
        """Amazon Nova Lite — fast, cost-efficient."""
        return cls(model_id=ModelID.NOVA_LITE, **kwargs)

    @classmethod
    def nova_micro(cls, **kwargs) -> "BedrockAdapter":
        """Amazon Nova Micro — text-only, lowest latency."""
        return cls(model_id=ModelID.NOVA_MICRO, **kwargs)

    @classmethod
    def llama3(cls, size: str = "70b", **kwargs) -> "BedrockAdapter":
        """Meta Llama 3. *size* is ``'70b'`` (default) or ``'8b'``."""
        model = ModelID.LLAMA3_70B if size == "70b" else ModelID.LLAMA3_8B
        return cls(model_id=model, **kwargs)

    @classmethod
    def mistral_large(cls, **kwargs) -> "BedrockAdapter":
        """Mistral Large."""
        return cls(model_id=ModelID.MISTRAL_LARGE, **kwargs)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"BedrockAdapter(model_id={self._model_id!r}, "
            f"region={self._region!r}, "
            f"temperature={self._temperature})"
        )
