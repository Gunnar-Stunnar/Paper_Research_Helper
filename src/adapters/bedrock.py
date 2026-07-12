"""AWS Bedrock LLM adapter.

Wraps ``langchain-aws`` ``ChatBedrock`` to provide pre-configured foundation
models from AWS Bedrock for use with agents, graphs, and tools in this project.

Authentication modes (in priority order)
-----------------------------------------
1. **Bedrock API key** — create a key in the AWS Bedrock console and set
   ``BEDROCK_API_KEY`` (or pass ``api_key=``). No IAM credentials required.
   See: https://docs.aws.amazon.com/bedrock/latest/userguide/api-keys-use.html

2. **IAM credentials** — standard boto3 credential chain:
   a. Explicit constructor args (``aws_access_key_id`` / ``aws_secret_access_key``)
   b. ``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY`` environment variables
   c. Named profile (``AWS_PROFILE`` env var or ``aws_profile=`` arg)
   d. IAM instance/task/pod role (when running on AWS infrastructure)

Supported model families
------------------------
Anthropic Claude   anthropic.claude-*  (via cross-region inference profiles)
Amazon Nova        amazon.nova-*
Amazon Titan       amazon.titan-*
Meta Llama         meta.llama*
Mistral            mistral.*

Environment variables
---------------------
BEDROCK_API_KEY       Bedrock API key — preferred over IAM when set
BEDROCK_REGION        AWS region for the Bedrock endpoint (default: us-east-1)
BEDROCK_MODEL_ID      Model ID or inference profile ID
                      (default: us.anthropic.claude-sonnet-4-5-20251101-v1:0)
BEDROCK_MAX_TOKENS    Max output tokens    (default: 4096)
BEDROCK_TEMPERATURE   Sampling temperature (default: 0)

IAM-only env vars (ignored when BEDROCK_API_KEY is set)
AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN
AWS_PROFILE

Usage — API key (simplest)
--------------------------
    import os
    os.environ["BEDROCK_API_KEY"] = "your-bedrock-api-key"

    from src.adapters.bedrock import BedrockAdapter
    llm = BedrockAdapter().chat_model()

Usage — IAM credentials
-----------------------
    llm = BedrockAdapter(
        aws_access_key_id="AKIA...",
        aws_secret_access_key="secret",
    ).chat_model()

Usage — named convenience constructors
---------------------------------------
    llm = BedrockAdapter.claude_sonnet().chat_model()
    llm = BedrockAdapter.claude_haiku(streaming=True).chat_model()
    llm = BedrockAdapter.nova_pro().chat_model()
    llm = BedrockAdapter.llama3(size="70b").chat_model()
"""

from __future__ import annotations

import os
from typing import Optional

from langchain_aws import ChatBedrock
from langchain_core.language_models import BaseChatModel


# ---------------------------------------------------------------------------
# Model catalogue
# ---------------------------------------------------------------------------

class ModelID:
    """Well-known Bedrock model and cross-region inference profile IDs."""

    # Anthropic Claude — cross-region inference profiles (higher throughput)
    CLAUDE_OPUS    = "us.anthropic.claude-opus-4-5-20251101-v1:0"
    CLAUDE_SONNET  = "us.anthropic.claude-sonnet-4-5-20251101-v1:0"
    CLAUDE_HAIKU   = "us.anthropic.claude-haiku-3-5-20251212-v1:0"

    # Amazon Nova
    NOVA_PRO       = "amazon.nova-pro-v1:0"
    NOVA_LITE      = "amazon.nova-lite-v1:0"
    NOVA_MICRO     = "amazon.nova-micro-v1:0"

    # Amazon Titan
    TITAN_EXPRESS  = "amazon.titan-text-express-v1"
    TITAN_LITE     = "amazon.titan-text-lite-v1"

    # Meta Llama
    LLAMA3_70B     = "meta.llama3-70b-instruct-v1:0"
    LLAMA3_8B      = "meta.llama3-8b-instruct-v1:0"

    # Mistral
    MISTRAL_LARGE  = "mistral.mistral-large-2402-v1:0"
    MISTRAL_7B     = "mistral.mistral-7b-instruct-v0:2"


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
        Falls back to ``BEDROCK_MODEL_ID`` env var.
    api_key:
        Bedrock API key. Falls back to ``BEDROCK_API_KEY`` env var.
        When set, IAM credential parameters are ignored.
    region:
        AWS region. Falls back to ``BEDROCK_REGION`` env var.
    aws_profile:
        Named boto3 credentials profile. Falls back to ``AWS_PROFILE``.
        Ignored when *api_key* is provided.
    aws_access_key_id / aws_secret_access_key / aws_session_token:
        Explicit IAM credentials. Ignored when *api_key* is provided.
    temperature:
        Sampling temperature. Falls back to ``BEDROCK_TEMPERATURE``.
    max_tokens:
        Max completion tokens. Falls back to ``BEDROCK_MAX_TOKENS``.
    streaming:
        Enable token streaming.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        api_key: Optional[str] = None,
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

        # Resolve API key — takes priority over all IAM credential options
        self._api_key = api_key or os.getenv("BEDROCK_API_KEY")

        # IAM credential fields (only used when no API key is present)
        self._aws_profile = aws_profile or os.getenv("AWS_PROFILE")
        self._aws_access_key_id = aws_access_key_id
        self._aws_secret_access_key = aws_secret_access_key
        self._aws_session_token = aws_session_token

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat_model(self) -> BaseChatModel:
        """Return a ``ChatBedrock`` instance ready to use with LangChain.

        The returned object is a drop-in replacement for ``ChatOpenAI`` or
        ``ChatAnthropic`` anywhere a ``BaseChatModel`` is accepted.

        When *api_key* is set it is passed directly to ``ChatBedrock``, which
        stores it as the ``AWS_BEARER_TOKEN_BEDROCK`` process environment
        variable and sends it as a Bearer token — no IAM credentials required.
        """
        kwargs: dict = {
            "model_id": self._model_id,
            "region_name": self._region,
            "streaming": self._streaming,
            "model_kwargs": {
                "temperature": self._temperature,
                "max_tokens": self._max_tokens,
            },
        }

        if self._api_key:
            # API key auth — library handles Bearer token injection
            kwargs["api_key"] = self._api_key
        else:
            # IAM credential chain
            if self._aws_profile:
                kwargs["credentials_profile_name"] = self._aws_profile
            if self._aws_access_key_id:
                kwargs["aws_access_key_id"] = self._aws_access_key_id
            if self._aws_secret_access_key:
                kwargs["aws_secret_access_key"] = self._aws_secret_access_key
            if self._aws_session_token:
                kwargs["aws_session_token"] = self._aws_session_token

        return ChatBedrock(**kwargs)

    @property
    def auth_mode(self) -> str:
        """Return a human-readable description of the active auth mode."""
        return "api_key" if self._api_key else "iam"

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
            f"auth={self.auth_mode!r}, "
            f"temperature={self._temperature})"
        )
