"""enchanter.llm.types — shared dataclasses for LLM request/response.

System prompts follow the Anthropic convention: they are passed in a
dedicated ``system`` field rather than as a message role.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class Message:
    """A single turn in the conversation."""

    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True)
class CompletionRequest:
    """Everything needed to call an LLM."""

    model: str
    messages: list[Message]
    system: str | None = None
    max_tokens: int = 1024
    temperature: float | None = None
    stop_sequences: tuple[str, ...] | None = None
    tools: list[dict] | None = None

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("CompletionRequest.messages must not be empty.")


@dataclass(frozen=True)
class CompletionResponse:
    """Everything returned from an LLM call."""

    text: str
    model: str
    stop_reason: str
    input_tokens: int
    output_tokens: int
    tool_calls: list[dict] = field(default_factory=list)
