"""robit.proxy.adapters.openai — OpenAI Chat Completions adapter.

Parses inbound ``POST /v1/chat/completions`` bodies into
:class:`~robit.proxy.canonical.CanonicalRequest` instances and renders
canonical responses / streams back out in OpenAI's wire format.

Scope:

* Non-streaming JSON response (``application/json``).
* Streaming SSE response (``text/event-stream``) using OpenAI's
  ``chat.completion.chunk`` shape, terminated by ``data: [DONE]``.

Out of scope: image inputs (``image_url`` parts are dropped — text only),
logprobs, the legacy ``/v1/completions`` endpoint, and the new Responses
API (``/v1/responses``).  Those land in later waves if the registry needs
them.
"""

from __future__ import annotations

import json
import secrets
import time
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


# ---------------------------------------------------------------------------
# Errors.
# ---------------------------------------------------------------------------


from .errors import AdapterParseError  # re-exported by adapters/__init__.py


# ---------------------------------------------------------------------------
# Adapter.
# ---------------------------------------------------------------------------


class OpenAIAdapter:
    """Adapter for OpenAI's ``/v1/chat/completions`` endpoint."""

    paths: ClassVar[tuple[str, ...]] = ("/v1/chat/completions",)

    # -- Routing -----------------------------------------------------------

    @staticmethod
    def matches(method: str, path: str) -> bool:
        """Return ``True`` for ``POST /v1/chat/completions`` (any query string)."""
        if method != "POST":
            return False
        bare = path.split("?", 1)[0]
        return bare == "/v1/chat/completions"

    # -- Inbound: OpenAI → canonical --------------------------------------

    @staticmethod
    def parse_request(
        body: bytes,
        path: str,  # noqa: ARG004 — kept for adapter-protocol parity
        headers: dict[str, str],  # noqa: ARG004
    ) -> CanonicalRequest:
        """Parse an OpenAI Chat Completions JSON body into canonical form."""
        try:
            data = json.loads(body)
        except (TypeError, ValueError) as exc:
            raise AdapterParseError(f"invalid JSON body: {exc}") from exc

        if not isinstance(data, dict):
            raise AdapterParseError("request body must be a JSON object")

        model = data.get("model")
        if not isinstance(model, str) or not model:
            raise AdapterParseError("'model' is required and must be a non-empty string")

        raw_messages = data.get("messages")
        if not isinstance(raw_messages, list) or not raw_messages:
            raise AdapterParseError("'messages' must be a non-empty list")

        system, conv_messages = _split_system_and_messages(raw_messages)

        tools = _parse_tools(data.get("tools"))
        tool_choice = _parse_tool_choice(data.get("tool_choice"))
        stop = _normalise_stop(data.get("stop"))

        metadata: dict[str, Any] = {}
        if "user" in data and data["user"] is not None:
            metadata["user"] = data["user"]

        return CanonicalRequest(
            model=model,
            messages=tuple(conv_messages),
            system=system,
            tools=tuple(tools),
            tool_choice=tool_choice,
            temperature=_optional_float(data.get("temperature")),
            top_p=_optional_float(data.get("top_p")),
            max_tokens=_optional_int(data.get("max_tokens")),
            stop_sequences=stop,
            stream=bool(data.get("stream", False)),
            metadata=metadata,
        )

    # -- Outbound: canonical → OpenAI (non-streaming) ---------------------

    @staticmethod
    def render_response(resp: CanonicalResponse) -> bytes:
        """Render a canonical response as OpenAI's ``chat.completion`` JSON."""
        message_obj, finish_reason = _build_message_and_finish(resp)
        body = {
            "id": _new_chatcmpl_id(),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": resp.model,
            "choices": [
                {
                    "index": 0,
                    "message": message_obj,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": resp.usage.input_tokens,
                "completion_tokens": resp.usage.output_tokens,
                "total_tokens": resp.usage.input_tokens + resp.usage.output_tokens,
            },
        }
        return json.dumps(body).encode("utf-8")

    # -- Outbound: canonical → OpenAI (streaming SSE) ---------------------

    @staticmethod
    async def render_stream(
        stream: AsyncIterator[CanonicalChunk],
    ) -> AsyncIterator[bytes]:
        """Translate a canonical chunk stream into OpenAI SSE bytes."""
        chatcmpl_id = _new_chatcmpl_id()
        created = int(time.time())
        model: str = "unknown"  # OpenAI requires a model field on every chunk.

        # Map canonical block index → OpenAI tool_call index for tool_use blocks.
        # Text blocks don't get an entry — they just stream as delta.content.
        tool_block_to_idx: dict[int, int] = {}
        next_tool_idx = 0
        # Track block kind (text vs tool_use) by index so we know whether a
        # subsequent input_json_delta routes to a tool_call entry.
        block_kind: dict[int, str] = {}

        async for chunk in stream:
            if chunk.type == "message_start":
                payload = _sse_envelope(chatcmpl_id, created, model)
                payload["choices"] = [
                    {
                        "index": 0,
                        "delta": {"role": "assistant"},
                        "finish_reason": None,
                    }
                ]
                yield _sse_line(payload)

            elif chunk.type == "content_block_start":
                # Preferred path: canonical carries block_kind (and, for
                # tool_use, the real id/name).  Open the tool_calls
                # envelope eagerly with authentic metadata — no
                # synthesised id, no empty name.
                if chunk.block_kind == "tool_use":
                    tool_idx = next_tool_idx
                    next_tool_idx += 1
                    tool_block_to_idx[chunk.index] = tool_idx
                    block_kind[chunk.index] = "tool_use"
                    open_payload = _sse_envelope(chatcmpl_id, created, model)
                    open_payload["choices"] = [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": tool_idx,
                                        "id": chunk.tool_id or "",
                                        "type": "function",
                                        "function": {
                                            "name": chunk.tool_name or "",
                                            "arguments": "",
                                        },
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ]
                    yield _sse_line(open_payload)
                elif chunk.block_kind == "text":
                    block_kind[chunk.index] = "text"
                    # No SSE event needed — text deltas stream directly
                    # as delta.content on subsequent text_delta chunks.
                else:
                    # Legacy producer: no block_kind.  Defer until the
                    # first delta tells us the block type.
                    block_kind.setdefault(chunk.index, "unknown")

            elif chunk.type == "text_delta":
                block_kind[chunk.index] = "text"
                payload = _sse_envelope(chatcmpl_id, created, model)
                payload["choices"] = [
                    {
                        "index": 0,
                        "delta": {"content": chunk.text or ""},
                        "finish_reason": None,
                    }
                ]
                yield _sse_line(payload)

            elif chunk.type == "input_json_delta":
                # First fragment on this block → open a tool_calls entry
                # if the content_block_start path didn't already do so
                # (i.e. legacy producer with no block_kind).
                if chunk.index not in tool_block_to_idx:
                    tool_idx = next_tool_idx
                    next_tool_idx += 1
                    tool_block_to_idx[chunk.index] = tool_idx
                    block_kind[chunk.index] = "tool_use"
                    # Legacy fallback: synthesise an id and emit empty
                    # name (the canonical chunk didn't carry them).
                    open_payload = _sse_envelope(chatcmpl_id, created, model)
                    open_payload["choices"] = [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": tool_idx,
                                        "id": f"call_{secrets.token_hex(6)}",
                                        "type": "function",
                                        "function": {
                                            "name": "",
                                            "arguments": "",
                                        },
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ]
                    yield _sse_line(open_payload)

                tool_idx = tool_block_to_idx[chunk.index]
                payload = _sse_envelope(chatcmpl_id, created, model)
                payload["choices"] = [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": tool_idx,
                                    "function": {
                                        "arguments": chunk.partial_json or "",
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
                yield _sse_line(payload)

            elif chunk.type == "content_block_stop":
                # OpenAI has no analog; suppress.
                pass

            elif chunk.type == "message_delta":
                openai_finish = _canonical_stop_to_openai(chunk.stop_reason)
                payload = _sse_envelope(chatcmpl_id, created, model)
                payload["choices"] = [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": openai_finish,
                    }
                ]
                if chunk.usage is not None:
                    payload["usage"] = {
                        "prompt_tokens": chunk.usage.input_tokens,
                        "completion_tokens": chunk.usage.output_tokens,
                        "total_tokens": (
                            chunk.usage.input_tokens + chunk.usage.output_tokens
                        ),
                    }
                yield _sse_line(payload)

            elif chunk.type == "message_stop":
                yield b"data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Inbound helpers.
# ---------------------------------------------------------------------------


def _split_system_and_messages(
    raw_messages: list[Any],
) -> tuple[str | None, list[Message]]:
    """Pull leading system messages out, parse the rest into canonical Messages."""
    system_chunks: list[str] = []
    rest: list[dict[str, Any]] = []
    saw_non_system = False

    for raw in raw_messages:
        if not isinstance(raw, dict):
            raise AdapterParseError("each message must be a JSON object")
        role = raw.get("role")
        if role == "system":
            if saw_non_system:
                raise AdapterParseError(
                    "system messages must appear before all other messages"
                )
            content = raw.get("content", "")
            system_chunks.append(_flatten_content_to_string(content))
        else:
            saw_non_system = True
            rest.append(raw)

    system = "\n\n".join(s for s in system_chunks if s) or None

    conv: list[Message] = []
    for raw in rest:
        conv.append(_parse_one_message(raw))

    return system, conv


def _parse_one_message(raw: dict[str, Any]) -> Message:
    """Parse a single non-system OpenAI message into a canonical Message."""
    role = raw.get("role")
    if role not in ("user", "assistant", "tool"):
        raise AdapterParseError(f"unsupported message role: {role!r}")

    if role == "tool":
        tool_call_id = raw.get("tool_call_id")
        if not isinstance(tool_call_id, str) or not tool_call_id:
            raise AdapterParseError("tool-role message requires 'tool_call_id'")
        content = raw.get("content", "")
        content_str = _flatten_content_to_string(content)
        return Message(
            role="tool",
            content=(
                ToolResultPart(
                    tool_use_id=tool_call_id,
                    content=content_str,
                    is_error=False,
                ),
            ),
        )

    parts: list[ContentPart] = []

    raw_content = raw.get("content")
    if isinstance(raw_content, str):
        if raw_content:
            parts.append(TextPart(text=raw_content))
    elif isinstance(raw_content, list):
        for piece in raw_content:
            if not isinstance(piece, dict):
                raise AdapterParseError("content list entries must be objects")
            ptype = piece.get("type")
            if ptype == "text":
                text = piece.get("text", "")
                if not isinstance(text, str):
                    raise AdapterParseError("text part 'text' must be a string")
                if text:
                    parts.append(TextPart(text=text))
            elif ptype == "image_url":
                # Out of scope — drop silently.  Wave 2 may surface this.
                continue
            else:
                # Unknown part type — drop to stay lenient.
                continue
    elif raw_content is None:
        pass  # Assistant turns with only tool_calls have null content.
    else:
        raise AdapterParseError("'content' must be a string, list, or null")

    if role == "assistant":
        for tc in raw.get("tool_calls", []) or []:
            if not isinstance(tc, dict):
                raise AdapterParseError("tool_calls entries must be objects")
            tc_id = tc.get("id")
            fn = tc.get("function") or {}
            if not isinstance(tc_id, str) or not tc_id:
                raise AdapterParseError("tool_call requires a string 'id'")
            name = fn.get("name")
            if not isinstance(name, str) or not name:
                raise AdapterParseError("tool_call function requires 'name'")
            args_raw = fn.get("arguments", "")
            if not isinstance(args_raw, str):
                raise AdapterParseError("tool_call arguments must be a JSON string")
            try:
                args = json.loads(args_raw) if args_raw else {}
            except (TypeError, ValueError) as exc:
                raise AdapterParseError(
                    f"tool_call arguments is not valid JSON: {exc}"
                ) from exc
            if not isinstance(args, dict):
                raise AdapterParseError("tool_call arguments must decode to an object")
            parts.append(ToolUsePart(id=tc_id, name=name, input=args))

    return Message(role=role, content=tuple(parts))


def _flatten_content_to_string(content: Any) -> str:
    """Reduce a string-or-parts-list content payload to a flat string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: list[str] = []
        for piece in content:
            if isinstance(piece, dict) and piece.get("type") == "text":
                t = piece.get("text", "")
                if isinstance(t, str):
                    out.append(t)
        return "".join(out)
    if content is None:
        return ""
    return str(content)


def _parse_tools(raw: Any) -> list[Tool]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise AdapterParseError("'tools' must be a list")
    out: list[Tool] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise AdapterParseError("each tool entry must be an object")
        if entry.get("type") != "function":
            # Other tool kinds (e.g. retrieval, code_interpreter) — drop.
            continue
        fn = entry.get("function") or {}
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            raise AdapterParseError("tool function requires 'name'")
        description = fn.get("description") or ""
        params = fn.get("parameters") or {}
        if not isinstance(params, dict):
            raise AdapterParseError("tool function 'parameters' must be an object")
        out.append(
            Tool(name=name, description=description, input_schema=params)
        )
    return out


def _parse_tool_choice(raw: Any) -> Any:
    if raw is None:
        return None
    if isinstance(raw, str):
        if raw == "required":
            return "any"
        if raw in ("auto", "none"):
            return raw
        raise AdapterParseError(f"unsupported tool_choice string: {raw!r}")
    if isinstance(raw, dict):
        if raw.get("type") == "function":
            fn = raw.get("function") or {}
            name = fn.get("name")
            if not isinstance(name, str) or not name:
                raise AdapterParseError("tool_choice function requires 'name'")
            return {"type": "tool", "name": name}
        # Unknown dict shape — pass through unchanged.
        return raw
    raise AdapterParseError("tool_choice must be a string or object")


def _normalise_stop(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        return (raw,)
    if isinstance(raw, list):
        out: list[str] = []
        for item in raw:
            if not isinstance(item, str):
                raise AdapterParseError("'stop' list entries must be strings")
            out.append(item)
        return tuple(out)
    raise AdapterParseError("'stop' must be a string or list of strings")


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    raise AdapterParseError("numeric field must be a number")


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise AdapterParseError("integer field must be an int, not a bool")
    if isinstance(value, int):
        return value
    raise AdapterParseError("integer field must be an int")


# ---------------------------------------------------------------------------
# Outbound helpers.
# ---------------------------------------------------------------------------


def _build_message_and_finish(
    resp: CanonicalResponse,
) -> tuple[dict[str, Any], str]:
    """Build the OpenAI ``message`` object and finish_reason from a canonical response."""
    text_chunks: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for part in resp.content:
        if isinstance(part, TextPart):
            text_chunks.append(part.text)
        elif isinstance(part, ToolUsePart):
            tool_calls.append(
                {
                    "id": part.id,
                    "type": "function",
                    "function": {
                        "name": part.name,
                        "arguments": json.dumps(part.input),
                    },
                }
            )
        # ToolResultPart shouldn't appear in an assistant response; ignore.

    message: dict[str, Any] = {"role": "assistant"}

    if text_chunks:
        message["content"] = "".join(text_chunks)
    elif tool_calls:
        # OpenAI convention: null content when tool_calls are present.
        message["content"] = None
    else:
        message["content"] = ""

    if tool_calls:
        message["tool_calls"] = tool_calls

    finish_reason = _canonical_stop_to_openai(resp.stop_reason)
    return message, finish_reason


def _canonical_stop_to_openai(stop_reason: str | None) -> str:
    """Map canonical (Anthropic) stop_reason → OpenAI finish_reason."""
    if stop_reason == "max_tokens":
        return "length"
    if stop_reason == "tool_use":
        return "tool_calls"
    # "end_turn", "stop_sequence", None → "stop"
    return "stop"


def _new_chatcmpl_id() -> str:
    """Generate an OpenAI-style chat-completion ID."""
    return f"chatcmpl-{secrets.token_hex(4)}"


def _sse_envelope(
    chatcmpl_id: str, created: int, model: str
) -> dict[str, Any]:
    """Create the common scaffolding for a streaming chunk."""
    return {
        "id": chatcmpl_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
    }


def _sse_line(payload: dict[str, Any]) -> bytes:
    """Serialise a chunk payload as an SSE ``data:`` line."""
    return f"data: {json.dumps(payload)}\n\n".encode("utf-8")


__all__ = [
    "AdapterParseError",
    "OpenAIAdapter",
]
