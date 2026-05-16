"""robit.llm.anthropic_client — AnthropicClient wrapping the Anthropic SDK.

The ``anthropic`` import is deferred to ``__init__`` so that importing
``robit.llm`` (e.g., to get ``MockLlmClient``) never fails when the
``anthropic`` package is not installed.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from .types import CompletionRequest, CompletionResponse, Message

if TYPE_CHECKING:
    # Only for type-checker; not executed at runtime.
    import anthropic as _anthropic_sdk


class AnthropicClient:
    """Production LlmClient that delegates to AsyncAnthropic.

    Supports two auth modes:

    * **API key** (``x-api-key`` header) — pay-per-token developer access.
      Provide via ``api_key=`` or ``ANTHROPIC_API_KEY``.
    * **OAuth bearer token** (``Authorization: Bearer …``) — Claude.ai
      Pro / Max subscription. Provide via ``auth_token=`` or
      ``CLAUDE_CODE_OAUTH_TOKEN`` / ``ANTHROPIC_AUTH_TOKEN``.

    Auto-detection order (when neither is passed explicitly):
    ``ANTHROPIC_API_KEY`` → ``CLAUDE_CODE_OAUTH_TOKEN`` → ``ANTHROPIC_AUTH_TOKEN``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        auth_token: str | None = None,
    ) -> None:
        # Deferred import — so `from robit.llm import MockLlmClient` works
        # even when `anthropic` is not installed.
        try:
            import anthropic  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The 'anthropic' package is required to use AnthropicClient. "
                "Install it with: pip install anthropic"
            ) from exc

        if api_key and auth_token:
            raise ValueError("Provide api_key OR auth_token, not both.")

        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        resolved_token = auth_token or (
            None if resolved_key else (
                os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
                or os.environ.get("ANTHROPIC_AUTH_TOKEN")
            )
        )

        if resolved_key:
            self._client: _anthropic_sdk.AsyncAnthropic = anthropic.AsyncAnthropic(
                api_key=resolved_key
            )
            self.auth_mode: str = "api_key"
        elif resolved_token:
            # OAuth subscription mode. The SDK sends Authorization: Bearer <token>
            # when auth_token= is set. The anthropic-beta: oauth-2025-04-20 header
            # is required for OAuth-issued tokens to reach the inference endpoint.
            self._client = anthropic.AsyncAnthropic(
                auth_token=resolved_token,
                default_headers={"anthropic-beta": "oauth-2025-04-20"},
            )
            self.auth_mode = "oauth"
        else:
            raise ValueError(
                "No Anthropic credentials provided. Pass api_key= or auth_token=, "
                "or set ANTHROPIC_API_KEY / CLAUDE_CODE_OAUTH_TOKEN."
            )

    async def complete(self, req: CompletionRequest) -> CompletionResponse:
        """Translate a CompletionRequest to an SDK call and back."""
        sdk_messages = [
            {"role": m.role, "content": m.content} for m in req.messages
        ]

        kwargs: dict = {
            "model": req.model,
            "messages": sdk_messages,
            "max_tokens": req.max_tokens,
        }
        if req.system is not None:
            kwargs["system"] = req.system
        if req.temperature is not None:
            kwargs["temperature"] = req.temperature
        if req.stop_sequences:
            kwargs["stop_sequences"] = list(req.stop_sequences)
        if req.tools:
            kwargs["tools"] = req.tools

        response = await self._client.messages.create(**kwargs)

        # Extract text and tool_calls from the content blocks.
        text_parts: list[str] = []
        tool_calls: list[dict] = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    {
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )

        return CompletionResponse(
            text="\n".join(text_parts),
            model=response.model,
            stop_reason=response.stop_reason or "end_turn",
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            tool_calls=tool_calls,
        )
