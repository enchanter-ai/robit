"""robit.llm — narrow LLM client abstraction.

Public API:

    from robit.llm import (
        LlmClient,          # Protocol — engines depend on this
        AnthropicClient,    # Real implementation (requires anthropic>=0.40)
        MockLlmClient,      # Test stub — no network, no anthropic dep
        Message,
        CompletionRequest,
        CompletionResponse,
    )
"""

from .anthropic_client import AnthropicClient
from .chatgpt_client import ChatGptClient, ConfigurationError
from .mock_client import MockLlmClient
from .protocol import LlmClient
from .types import CompletionRequest, CompletionResponse, Message

__all__ = [
    "AnthropicClient",
    "ChatGptClient",
    "CompletionRequest",
    "CompletionResponse",
    "ConfigurationError",
    "LlmClient",
    "Message",
    "MockLlmClient",
]
