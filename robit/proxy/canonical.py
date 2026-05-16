"""robit.proxy.canonical — provider-neutral request/response dataclasses.

These are the contract types that all proxy adapters (Anthropic, OpenAI, Gemini)
parse into on the way in, and render from on the way out.  The shapes lean
toward Anthropic's event model on the streaming side because that model is the
richest of the three majors — every other provider's chunk shape can be
*rendered down* from a CanonicalChunk stream without information loss, whereas
the reverse (OpenAI/Gemini deltas → Anthropic events) would require
re-synthesising block boundaries from finish_reason hints alone.

Design rules:

* All dataclasses are ``frozen=True`` so instances are hashable and safe to
  pass across async boundaries without defensive copies.
* Sequences are typed as ``tuple[...]`` rather than ``list[...]`` so the
  immutability extends into the containers.
* Streaming events follow Anthropic's lifecycle:
  ``message_start`` → (``content_block_start`` → ``text_delta*`` |
  ``input_json_delta*`` → ``content_block_stop``)+ → ``message_delta``? →
  ``message_stop``.  ``index`` identifies which content block a per-block
  event applies to.

Nothing in this module imports LiteLLM or any provider SDK — keep it pure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Union


# ---------------------------------------------------------------------------
# Content parts — the atoms of a message body.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TextPart:
    """A literal text span."""

    text: str
    type: Literal["text"] = "text"


@dataclass(frozen=True)
class ToolUsePart:
    """An assistant-emitted tool call.

    ``input`` is the (already-parsed) JSON arguments object.  Streaming
    callers should accumulate ``input_json_delta`` chunks and parse the
    final string into a dict before constructing this part.
    """

    id: str
    name: str
    input: dict
    type: Literal["tool_use"] = "tool_use"


@dataclass(frozen=True)
class ToolResultPart:
    """A user-supplied result for a previous tool call.

    ``content`` is the stringified result body.  Providers that accept
    structured tool results (Anthropic) will wrap this into their native
    shape on render; OpenAI-style adapters embed it inline as a tool
    message.
    """

    tool_use_id: str
    content: str
    is_error: bool = False
    type: Literal["tool_result"] = "tool_result"


ContentPart = Union[TextPart, ToolUsePart, ToolResultPart]


# ---------------------------------------------------------------------------
# Messages and tools.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Message:
    """A single conversation turn.

    Pure-text messages have a one-element ``content`` tuple containing a
    single :class:`TextPart`.  Mixed turns (e.g. an assistant that emits
    text + a tool_use) carry multiple parts in source order.
    """

    role: Literal["user", "assistant", "system", "tool"]
    content: tuple[ContentPart, ...]


@dataclass(frozen=True)
class Tool:
    """A tool/function definition the model may invoke."""

    name: str
    description: str
    input_schema: dict  # JSON Schema describing the tool's arguments.


# ---------------------------------------------------------------------------
# Request.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CanonicalRequest:
    """Provider-neutral inbound request.

    Notes
    -----
    * ``model`` is passed through unchanged; LiteLLM picks the provider
      from the string prefix (e.g. ``anthropic/…``, ``gpt-4o-mini``,
      ``gemini/…``) or its own configured model_list.
    * ``tool_choice`` accepts the three string verbs (``"auto"``, ``"any"``,
      ``"none"``) or a dict like ``{"type": "tool", "name": "foo"}`` for
      a forced selection.  Adapters lower this to provider-native shapes.
    * ``metadata`` is an opaque pass-through bag — useful for routing
      hints, request IDs, billing tags, etc.  Never read inside the
      proxy core.
    """

    model: str
    messages: tuple[Message, ...]
    system: str | None = None
    tools: tuple[Tool, ...] = ()
    tool_choice: Union[Literal["auto", "any", "none"], dict, None] = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop_sequences: tuple[str, ...] = ()
    stream: bool = False
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Response.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CanonicalUsage:
    """Token accounting from the upstream call."""

    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class CanonicalResponse:
    """Non-streaming completion result.

    ``stop_reason`` uses Anthropic's vocabulary; OpenAI/Gemini adapters map
    their finish reasons into this enum on the way through.
    """

    model: str
    content: tuple[ContentPart, ...]
    stop_reason: Union[
        Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"], None
    ]
    usage: CanonicalUsage


# ---------------------------------------------------------------------------
# Streaming chunk.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CanonicalChunk:
    """One streaming delta in the canonical event stream.

    Event types mirror Anthropic's content-block lifecycle:

    ============================  ============================================
    type                          payload
    ============================  ============================================
    ``message_start``             — (signals stream begin; ``model`` available
                                  to consumers via the request context)
    ``content_block_start``       ``index`` of the block being opened, plus
                                  ``block_kind`` (``"text"`` or ``"tool_use"``)
                                  and — for tool_use only — ``tool_id`` and
                                  ``tool_name``
    ``text_delta``                ``index`` + ``text`` (incremental text)
    ``input_json_delta``          ``index`` + ``partial_json`` (incremental
                                  tool-use arguments, JSON fragment)
    ``content_block_stop``        ``index`` of the block being closed
    ``message_delta``             ``stop_reason`` and/or ``usage`` updates
    ``message_stop``              — (signals stream end)
    ============================  ============================================

    ``index`` defaults to 0 because most streams contain a single content
    block; multi-block streams (text + tool_use) bump it per block.

    The ``block_kind`` / ``tool_id`` / ``tool_name`` slots are populated
    *only* on ``content_block_start`` events and are ``None`` on every
    other event type.  ``block_kind="text"`` SHOULD be set on
    ``content_block_start`` for text blocks too (so adapters don't have
    to defer to the first delta to learn the block shape); legacy
    producers that leave it ``None`` still work via the adapters'
    fallback paths.
    """

    type: Literal[
        "message_start",
        "content_block_start",
        "text_delta",
        "input_json_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    index: int = 0
    text: str | None = None
    partial_json: str | None = None
    stop_reason: str | None = None
    usage: CanonicalUsage | None = None
    # Populated only on ``content_block_start``; ``None`` on every other event.
    block_kind: Literal["text", "tool_use"] | None = None
    # Populated only on ``content_block_start`` for ``tool_use`` blocks;
    # ``None`` on every other event type (and on text-block starts).
    tool_id: str | None = None
    tool_name: str | None = None


__all__ = [
    "TextPart",
    "ToolUsePart",
    "ToolResultPart",
    "ContentPart",
    "Message",
    "Tool",
    "CanonicalRequest",
    "CanonicalUsage",
    "CanonicalResponse",
    "CanonicalChunk",
]
