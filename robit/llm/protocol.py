"""robit.llm.protocol — LlmClient Protocol.

Engines depend on this narrow interface, not on any concrete SDK class.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .types import CompletionRequest, CompletionResponse


@runtime_checkable
class LlmClient(Protocol):
    """Minimal async interface for LLM completion calls."""

    async def complete(self, req: CompletionRequest) -> CompletionResponse:
        """Execute a completion and return the structured response."""
        ...
