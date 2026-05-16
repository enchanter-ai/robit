"""robit.proxy.adapters.anthropic — Anthropic Messages API wire adapter.

Translates between the Anthropic ``/v1/messages`` JSON wire format and the
provider-neutral :mod:`robit.proxy.canonical` dataclasses.

The adapter is **pure wire translation** — it does not call any upstream
provider SDK.  The proxy server is responsible for handing the parsed
:class:`~robit.proxy.canonical.CanonicalRequest` to
:func:`robit.proxy.upstream.call_upstream` (or ``stream_upstream``) and
then handing the result back to :meth:`AnthropicAdapter.render_response`
(or ``render_stream``).

Wire-format reference: https://docs.anthropic.com/en/api/messages
"""

from __future__ import annotations

import json
import secrets
from typing import Any, AsyncIterator, ClassVar

from ..canonical import (
    CanonicalChunk,
    CanonicalRequest,
    CanonicalResponse,
    ContentPart,
    Message,
    TextPart,
    Tool,
    ToolResultPart,
    ToolUsePart,
)


from .errors import AdapterParseError  # re-exported by adapters/__init__.py


class AnthropicAdapter:
    """Wire adapter for Anthropic's ``/v1/messages`` endpoint."""

    paths: ClassVar[tuple[str, ...]] = ("/v1/messages",)

    # ------------------------------------------------------------------
    # Routing.
    # ------------------------------------------------------------------

    @staticmethod
    def matches(method: str, path: str) -> bool:
        return method == "POST" and path.split("?", 1)[0] == "/v1/messages"

    # ------------------------------------------------------------------
    # Request parsing.
    # ------------------------------------------------------------------

    @staticmethod
    def parse_request(
        body: bytes, path: str, headers: dict[str, str]
    ) -> CanonicalRequest:
        """Parse a ``/v1/messages`` JSON body into a CanonicalRequest."""
        try:
            payload = json.loads(body)
        except (ValueError, TypeError) as exc:
            raise AdapterParseError(f"invalid JSON body: {exc}") from exc

        if not isinstance(payload, dict):
            raise AdapterParseError("request body must be a JSON object")

        model = payload.get("model")
        if not isinstance(model, str) or not model:
            raise AdapterParseError("missing or invalid 'model'")

        # Anthropic requires max_tokens — surface this as a 400.
        max_tokens_raw = payload.get("max_tokens")
        if max_tokens_raw is None:
            raise AdapterParseError("'max_tokens' is required")
        try:
            max_tokens = int(max_tokens_raw)
        except (TypeError, ValueError) as exc:
            raise AdapterParseError("'max_tokens' must be an integer") from exc

        raw_messages = payload.get("messages")
        if not isinstance(raw_messages, list):
            raise AdapterParseError("'messages' must be a list")

        messages = tuple(_parse_message(m) for m in raw_messages)

        system = payload.get("system")
        if system is not None and not isinstance(system, str):
            # Anthropic also allows a list-of-blocks for system; flatten to text.
            if isinstance(system, list):
                system = "".join(
                    block.get("text", "")
                    for block in system
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            else:
                raise AdapterParseError("'system' must be a string or block list")

        tools = tuple(_parse_tool(t) for t in payload.get("tools", []) or [])

        tool_choice = _parse_tool_choice(payload.get("tool_choice"))

        stop_sequences_raw = payload.get("stop_sequences", []) or []
        if not isinstance(stop_sequences_raw, list):
            raise AdapterParseError("'stop_sequences' must be a list")
        stop_sequences = tuple(str(s) for s in stop_sequences_raw)

        temperature = payload.get("temperature")
        if temperature is not None and not isinstance(temperature, (int, float)):
            raise AdapterParseError("'temperature' must be a number")

        top_p = payload.get("top_p")
        if top_p is not None and not isinstance(top_p, (int, float)):
            raise AdapterParseError("'top_p' must be a number")

        stream = bool(payload.get("stream", False))

        metadata_raw = payload.get("metadata") or {}
        if not isinstance(metadata_raw, dict):
            raise AdapterParseError("'metadata' must be an object")

        return CanonicalRequest(
            model=model,
            messages=messages,
            system=system if system else None,
            tools=tools,
            tool_choice=tool_choice,
            temperature=float(temperature) if temperature is not None else None,
            top_p=float(top_p) if top_p is not None else None,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
            stream=stream,
            metadata=dict(metadata_raw),
        )

    # ------------------------------------------------------------------
    # Response rendering — non-streaming.
    # ------------------------------------------------------------------

    @staticmethod
    def render_response(resp: CanonicalResponse) -> bytes:
        """Render a :class:`CanonicalResponse` into Anthropic JSON bytes.

        The synthesised ``id`` follows the shape ``msg_<8 hex chars>`` —
        not the precise Anthropic-internal format, but a stable,
        client-recognisable prefix.  Clients that key off the exact
        internal id format will need to relax that check.
        """
        content_blocks = [_render_content_part(part) for part in resp.content]
        body = {
            "id": f"msg_{secrets.token_hex(4)}",
            "type": "message",
            "role": "assistant",
            "model": resp.model,
            "content": content_blocks,
            "stop_reason": resp.stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
            },
        }
        return json.dumps(body).encode("utf-8")

    # ------------------------------------------------------------------
    # Response rendering — streaming SSE.
    # ------------------------------------------------------------------

    @staticmethod
    async def render_stream(
        stream: AsyncIterator[CanonicalChunk],
    ) -> AsyncIterator[bytes]:
        """Translate canonical chunks into Anthropic SSE event bytes.

        Each yielded bytes object is a complete SSE block of the form::

            event: <name>\\n
            data: <json>\\n
            \\n

        Per-block ``content_block_start`` events need to remember whether
        the block is text or tool_use so the wire shape carries the right
        ``content_block`` payload.  Since the canonical
        ``content_block_start`` chunk does not carry a type discriminator,
        we infer from the *next* delta: an ``input_json_delta`` indicates
        tool_use, otherwise text.  To keep ordering correct we buffer the
        most recent ``content_block_start`` until the first delta on the
        same index is seen (or the block is closed empty).
        """
        # Track per-block metadata so we can synthesise the right wrapper.
        # block_type[index] in {"text", "tool_use"}.
        block_type: dict[int, str] = {}
        # Pending content_block_start that has not been emitted yet because
        # we are waiting to see the first delta on that index.
        pending_start: dict[int, bool] = {}
        # Track usage and stop_reason from any message_delta seen.
        last_model: str | None = None

        async for chunk in stream:
            if chunk.type == "message_start":
                yield _sse(
                    "message_start",
                    {
                        "type": "message_start",
                        "message": {
                            "id": f"msg_{secrets.token_hex(4)}",
                            "type": "message",
                            "role": "assistant",
                            "model": last_model or "",
                            "content": [],
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": {"input_tokens": 0, "output_tokens": 0},
                        },
                    },
                )
            elif chunk.type == "content_block_start":
                idx = chunk.index
                # Preferred path: the canonical chunk carries block_kind
                # (and, for tool_use, the real tool id/name).  Emit the
                # Anthropic content_block_start eagerly so the wire
                # carries authentic metadata.
                if chunk.block_kind == "tool_use":
                    block_type[idx] = "tool_use"
                    yield _sse(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": idx,
                            "content_block": {
                                "type": "tool_use",
                                "id": chunk.tool_id or "",
                                "name": chunk.tool_name or "",
                                "input": {},
                            },
                        },
                    )
                elif chunk.block_kind == "text":
                    block_type[idx] = "text"
                    yield _sse(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": idx,
                            "content_block": {"type": "text", "text": ""},
                        },
                    )
                else:
                    # Legacy producer: no block_kind set.  Defer until we
                    # see the first delta and can resolve the block type
                    # from the delta type alone.
                    pending_start[idx] = True
            elif chunk.type == "text_delta":
                idx = chunk.index
                if pending_start.pop(idx, False):
                    block_type[idx] = "text"
                    yield _sse(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": idx,
                            "content_block": {"type": "text", "text": ""},
                        },
                    )
                yield _sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": idx,
                        "delta": {"type": "text_delta", "text": chunk.text or ""},
                    },
                )
            elif chunk.type == "input_json_delta":
                idx = chunk.index
                if pending_start.pop(idx, False):
                    block_type[idx] = "tool_use"
                    yield _sse(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": idx,
                            # Legacy fallback: we don't know the tool id
                            # or name (the canonical chunk didn't carry
                            # them on content_block_start).  Emit a
                            # placeholder shell.
                            "content_block": {
                                "type": "tool_use",
                                "id": "",
                                "name": "",
                                "input": {},
                            },
                        },
                    )
                yield _sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": idx,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": chunk.partial_json or "",
                        },
                    },
                )
            elif chunk.type == "content_block_stop":
                idx = chunk.index
                # If the block was never resolved (empty), emit an empty
                # text block_start now so the client sees a balanced pair.
                if pending_start.pop(idx, False):
                    block_type[idx] = "text"
                    yield _sse(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": idx,
                            "content_block": {"type": "text", "text": ""},
                        },
                    )
                yield _sse(
                    "content_block_stop",
                    {"type": "content_block_stop", "index": idx},
                )
            elif chunk.type == "message_delta":
                payload: dict[str, Any] = {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": chunk.stop_reason,
                        "stop_sequence": None,
                    },
                }
                if chunk.usage is not None:
                    payload["usage"] = {
                        "input_tokens": chunk.usage.input_tokens,
                        "output_tokens": chunk.usage.output_tokens,
                    }
                else:
                    payload["usage"] = {"output_tokens": 0}
                yield _sse("message_delta", payload)
            elif chunk.type == "message_stop":
                yield _sse("message_stop", {"type": "message_stop"})
            # Unknown chunk types are silently dropped — forward
            # compatibility with future canonical extensions.


# ---------------------------------------------------------------------------
# Parsing helpers.
# ---------------------------------------------------------------------------


def _parse_message(raw: Any) -> Message:
    if not isinstance(raw, dict):
        raise AdapterParseError("each message must be an object")
    role = raw.get("role")
    if role not in ("user", "assistant", "system", "tool"):
        raise AdapterParseError(f"invalid message role: {role!r}")
    content_raw = raw.get("content")
    if content_raw is None:
        raise AdapterParseError("message missing 'content'")
    parts = tuple(_parse_content(content_raw))
    return Message(role=role, content=parts)


def _parse_content(content: Any) -> list[ContentPart]:
    if isinstance(content, str):
        # Normalise plain string to a single TextPart.
        return [TextPart(text=content)]
    if not isinstance(content, list):
        raise AdapterParseError("message 'content' must be a string or list")
    out: list[ContentPart] = []
    for block in content:
        if not isinstance(block, dict):
            raise AdapterParseError("content block must be an object")
        btype = block.get("type")
        if btype == "text":
            text = block.get("text", "")
            if not isinstance(text, str):
                raise AdapterParseError("text block 'text' must be a string")
            out.append(TextPart(text=text))
        elif btype == "tool_use":
            block_id = block.get("id")
            name = block.get("name")
            tool_input = block.get("input")
            if not isinstance(block_id, str):
                raise AdapterParseError("tool_use 'id' must be a string")
            if not isinstance(name, str):
                raise AdapterParseError("tool_use 'name' must be a string")
            if tool_input is None:
                tool_input = {}
            if not isinstance(tool_input, dict):
                raise AdapterParseError("tool_use 'input' must be an object")
            out.append(ToolUsePart(id=block_id, name=name, input=tool_input))
        elif btype == "tool_result":
            tool_use_id = block.get("tool_use_id")
            if not isinstance(tool_use_id, str):
                raise AdapterParseError("tool_result 'tool_use_id' must be a string")
            raw_content = block.get("content", "")
            # Anthropic allows the content to be a string or a list of blocks
            # (typically text).  Flatten to a single string.
            if isinstance(raw_content, list):
                flat = []
                for sub in raw_content:
                    if isinstance(sub, dict) and sub.get("type") == "text":
                        flat.append(sub.get("text", ""))
                    elif isinstance(sub, str):
                        flat.append(sub)
                tool_content = "".join(flat)
            elif isinstance(raw_content, str):
                tool_content = raw_content
            else:
                tool_content = json.dumps(raw_content)
            is_error = bool(block.get("is_error", False))
            out.append(
                ToolResultPart(
                    tool_use_id=tool_use_id,
                    content=tool_content,
                    is_error=is_error,
                )
            )
        else:
            raise AdapterParseError(f"unknown content block type: {btype!r}")
    return out


def _parse_tool(raw: Any) -> Tool:
    if not isinstance(raw, dict):
        raise AdapterParseError("each tool must be an object")
    name = raw.get("name")
    if not isinstance(name, str):
        raise AdapterParseError("tool 'name' must be a string")
    description = raw.get("description", "")
    if not isinstance(description, str):
        raise AdapterParseError("tool 'description' must be a string")
    schema = raw.get("input_schema", {})
    if not isinstance(schema, dict):
        raise AdapterParseError("tool 'input_schema' must be an object")
    return Tool(name=name, description=description, input_schema=schema)


def _parse_tool_choice(raw: Any) -> Any:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise AdapterParseError("'tool_choice' must be an object")
    kind = raw.get("type")
    if kind == "auto":
        return "auto"
    if kind == "any":
        return "any"
    if kind == "tool":
        name = raw.get("name")
        if not isinstance(name, str):
            raise AdapterParseError("tool_choice tool 'name' must be a string")
        return {"type": "tool", "name": name}
    raise AdapterParseError(f"unknown tool_choice type: {kind!r}")


# ---------------------------------------------------------------------------
# Rendering helpers.
# ---------------------------------------------------------------------------


def _render_content_part(part: ContentPart) -> dict[str, Any]:
    if isinstance(part, TextPart):
        return {"type": "text", "text": part.text}
    if isinstance(part, ToolUsePart):
        return {
            "type": "tool_use",
            "id": part.id,
            "name": part.name,
            "input": part.input,
        }
    if isinstance(part, ToolResultPart):
        # Assistant responses normally do not contain tool_result, but
        # render it round-trip safely if encountered.
        return {
            "type": "tool_result",
            "tool_use_id": part.tool_use_id,
            "content": part.content,
            "is_error": part.is_error,
        }
    raise TypeError(f"unknown content part type: {type(part).__name__}")


def _sse(event: str, data: dict[str, Any]) -> bytes:
    """Format a single SSE event block as UTF-8 bytes."""
    body = json.dumps(data, separators=(",", ":"))
    return f"event: {event}\ndata: {body}\n\n".encode("utf-8")


__all__ = [
    "AnthropicAdapter",
    "AdapterParseError",
]
