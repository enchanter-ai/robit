"""robit.proxy.adapters.codex — Codex CLI (OpenAI Responses API) adapter.

Codex CLI (``codex-rs`` >= 0.130) speaks the **OpenAI Responses API** at
``POST /v1/responses``. The wire format differs structurally from
``/v1/chat/completions``:

* System prompt is a top-level ``instructions`` string (not a system-role
  message).
* User/assistant turns live under ``input: [{type:"message", role, content:
  [{type:"input_text"|"output_text", text}]}]``.
* A third role ``developer`` is accepted alongside ``user``/``assistant``.
* Streaming SSE events are Responses-API event types
  (``response.output_text.delta``, ``response.completed``, …) — not
  ``chat.completion.chunk``.

Auth modes (per Wave 16.0 audit ``docs/architecture/audits/codex-protocol.md``):

1. **API key:** ``Authorization: Bearer sk-...`` → upstream
   ``https://api.openai.com/v1/responses`` (the default LiteLLM route for
   ``gpt-5-codex`` and similar).
2. **ChatGPT login:** ``Authorization: Bearer <jwt>`` plus
   ``ChatGPT-Account-ID: <id>`` → upstream
   ``https://chatgpt.com/backend-api/codex/responses``.

v1 limitations (documented here so consumers can rely on them):

* **No WebSocket transport.** Codex prefers ``OpenAI-Beta: responses_websockets``
  but falls back to HTTP-SSE; we support only the fallback.
* **No attestation header** (``x-oai-attestation``). Deferred — Wave 16.0 flagged.
* **Pass-through Codex-private headers verbatim.** ``x-codex-turn-state``,
  ``x-codex-installation-id``, etc. are accepted on the inbound request but the
  proxy does not synthesise or mutate them.
* **Developer-role collapse.** Codex's ``"developer"`` role represents
  tool/system instructions distinct from the top-level ``instructions`` field;
  the canonical pipeline has no developer role, so we fold it into
  ``"system"``. Round-tripping a Codex turn that depends on the developer/system
  distinction will degrade.
* **No tool-call streaming.** Canonical ``input_json_delta`` chunks are
  *not* emitted as ``response.function_call_arguments.delta`` events in v1 —
  tool-using Codex sessions may degrade to text-only responses.
* **ChatGPT-internal upstream URL override deferred.** The proxy does not
  currently rewrite LiteLLM's base URL to ``chatgpt.com/backend-api/codex``
  when it sees a ChatGPT JWT — Codex's ChatGPT-login flow against this proxy
  is therefore not yet end-to-end functional; API-key mode is.
* **Reasoning summaries / encrypted reasoning carry-forward** (``include:
  ["reasoning.encrypted_content"]``, ``previous_response_id``) are accepted
  on the request and dropped — no v1 plumbing.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, ClassVar

from ...llm._codex_responses import (
    EVT_OUTPUT_TEXT_DELTA,
    EVT_OUTPUT_TEXT_DONE,
    EVT_RESPONSE_COMPLETED,
    EVT_RESPONSE_CREATED,
    new_message_id,
    new_response_id,
    now_ts,
    render_sse_event,
)
from ..canonical import (
    CanonicalChunk,
    CanonicalRequest,
    CanonicalResponse,
    ContentPart,
    Message,
    TextPart,
    Tool,
    ToolUsePart,
)
from .errors import AdapterParseError


class CodexAdapter:
    """Wire adapter for Codex CLI talking to ``/v1/responses``.

    See module docstring for the dual-auth-mode routing notes and the v1
    limitation list.
    """

    paths: ClassVar[tuple[str, ...]] = ("/v1/responses",)

    # -- Routing ----------------------------------------------------------

    @staticmethod
    def matches(method: str, path: str) -> bool:
        """``POST /v1/responses`` (any query string)."""
        if method != "POST":
            return False
        return path.split("?", 1)[0] == "/v1/responses"

    # -- Inbound: Responses-API JSON → CanonicalRequest -------------------

    @staticmethod
    def parse_request(
        body: bytes,
        path: str,  # noqa: ARG004 — adapter-protocol parity
        headers: dict[str, str],  # noqa: ARG004
    ) -> CanonicalRequest:
        """Parse a Responses-API JSON body into a :class:`CanonicalRequest`."""
        try:
            data = json.loads(body)
        except (TypeError, ValueError) as exc:
            raise AdapterParseError(f"invalid JSON body: {exc}") from exc

        if not isinstance(data, dict):
            raise AdapterParseError("request body must be a JSON object")

        model = data.get("model")
        if not isinstance(model, str) or not model:
            raise AdapterParseError("'model' is required and must be a non-empty string")

        # System prompt — top-level string. Optional in our parser (Codex's
        # upstream returns 400 if absent, but Enchanter shouldn't add a
        # stricter contract than the upstream itself).
        system_raw = data.get("instructions")
        if system_raw is not None and not isinstance(system_raw, str):
            raise AdapterParseError("'instructions' must be a string when present")
        system: str | None = system_raw or None

        # input[] — required.
        raw_input = data.get("input")
        if not isinstance(raw_input, list) or not raw_input:
            raise AdapterParseError("'input' must be a non-empty list")

        # Hoist developer-role messages into the system block (canonical
        # has no developer role). Their text is concatenated after any
        # top-level instructions.
        extra_system: list[str] = []
        messages: list[Message] = []
        for item in raw_input:
            if not isinstance(item, dict):
                raise AdapterParseError("each input item must be a JSON object")
            itype = item.get("type", "message")
            if itype != "message":
                # function_call / function_call_output etc. — drop in v1.
                continue
            role = item.get("role")
            if role not in ("user", "assistant", "developer", "system"):
                raise AdapterParseError(f"unsupported input role: {role!r}")
            text = _flatten_response_content(item.get("content"))
            if role in ("developer", "system"):
                if text:
                    extra_system.append(text)
                continue
            parts: tuple[ContentPart, ...] = (
                (TextPart(text=text),) if text else ()
            )
            messages.append(Message(role=role, content=parts))

        if extra_system:
            system = "\n\n".join(
                s for s in ([system] if system else []) + extra_system if s
            )

        if not messages:
            raise AdapterParseError("'input' contained no user/assistant messages")

        tools = _parse_tools(data.get("tools"))
        tool_choice = _parse_tool_choice(data.get("tool_choice"))
        stop = _normalise_stop(data.get("stop"))

        metadata: dict[str, Any] = {}
        if data.get("prompt_cache_key"):
            metadata["prompt_cache_key"] = data["prompt_cache_key"]

        return CanonicalRequest(
            model=model,
            messages=tuple(messages),
            system=system,
            tools=tuple(tools),
            tool_choice=tool_choice,
            temperature=_optional_float(data.get("temperature")),
            top_p=_optional_float(data.get("top_p")),
            max_tokens=_optional_int(data.get("max_output_tokens")),
            stop_sequences=stop,
            stream=bool(data.get("stream", False)),
            metadata=metadata,
        )

    # -- Outbound: CanonicalResponse → Responses-API JSON -----------------

    @staticmethod
    def render_response(resp: CanonicalResponse) -> bytes:
        """Render a CanonicalResponse as a Responses-API JSON body."""
        text_chunks: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for part in resp.content:
            if isinstance(part, TextPart):
                text_chunks.append(part.text)
            elif isinstance(part, ToolUsePart):
                tool_calls.append(
                    {
                        "type": "function_call",
                        "id": new_message_id().replace("msg_", "fc_"),
                        "call_id": part.id,
                        "name": part.name,
                        "arguments": json.dumps(part.input),
                    }
                )
        output: list[dict[str, Any]] = []
        if text_chunks:
            output.append(
                {
                    "type": "message",
                    "id": new_message_id(),
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": "".join(text_chunks)}
                    ],
                }
            )
        output.extend(tool_calls)
        body = {
            "id": new_response_id(),
            "object": "response",
            "created": now_ts(),
            "model": resp.model,
            "output": output,
            "usage": {
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
                "total_tokens": resp.usage.input_tokens + resp.usage.output_tokens,
            },
            "status": "completed",
        }
        return json.dumps(body).encode("utf-8")

    # -- Outbound: canonical chunks → Responses-API SSE -------------------

    @staticmethod
    async def render_stream(
        stream: AsyncIterator[CanonicalChunk],
    ) -> AsyncIterator[bytes]:
        """Translate a canonical chunk stream into Responses-API SSE bytes.

        v1 maps text deltas to ``response.output_text.delta`` events. Tool-call
        argument deltas (canonical ``input_json_delta``) are dropped — see the
        module-level v1 limitations list.
        """
        response_id = new_response_id()
        item_id = new_message_id()
        created = now_ts()
        model = "unknown"  # Codex doesn't require a model echo per chunk; we
                           # still populate response.created and response.completed.
        output_text_open = False
        accumulated: list[str] = []
        usage_input = 0
        usage_output = 0

        async for chunk in stream:
            if chunk.type == "message_start":
                yield render_sse_event(
                    EVT_RESPONSE_CREATED,
                    {
                        "response": {
                            "id": response_id,
                            "object": "response",
                            "created": created,
                            "model": model,
                            "status": "in_progress",
                            "output": [],
                        }
                    },
                )

            elif chunk.type == "content_block_start":
                # Only text blocks become SSE events in v1.
                if chunk.block_kind == "tool_use":
                    # Drop — tool-call streaming deferred.
                    continue
                # Lazy-open behaviour: don't emit anything yet; the first
                # text_delta opens the output_text logical block on the wire.

            elif chunk.type == "text_delta":
                output_text_open = True
                text = chunk.text or ""
                if text:
                    accumulated.append(text)
                    yield render_sse_event(
                        EVT_OUTPUT_TEXT_DELTA,
                        {
                            "item_id": item_id,
                            "output_index": 0,
                            "content_index": 0,
                            "delta": text,
                        },
                    )

            elif chunk.type == "input_json_delta":
                # Tool-call argument streaming — dropped in v1.
                continue

            elif chunk.type == "content_block_stop":
                if output_text_open:
                    yield render_sse_event(
                        EVT_OUTPUT_TEXT_DONE,
                        {
                            "item_id": item_id,
                            "output_index": 0,
                            "content_index": 0,
                            "text": "".join(accumulated),
                        },
                    )
                    output_text_open = False

            elif chunk.type == "message_delta":
                if chunk.usage is not None:
                    usage_input = chunk.usage.input_tokens
                    usage_output = chunk.usage.output_tokens

            elif chunk.type == "message_stop":
                final_output: list[dict[str, Any]] = []
                if accumulated:
                    final_output.append(
                        {
                            "type": "message",
                            "id": item_id,
                            "role": "assistant",
                            "status": "completed",
                            "content": [
                                {"type": "output_text", "text": "".join(accumulated)}
                            ],
                        }
                    )
                yield render_sse_event(
                    EVT_RESPONSE_COMPLETED,
                    {
                        "response": {
                            "id": response_id,
                            "object": "response",
                            "created": created,
                            "model": model,
                            "status": "completed",
                            "output": final_output,
                            "usage": {
                                "input_tokens": usage_input,
                                "output_tokens": usage_output,
                                "total_tokens": usage_input + usage_output,
                            },
                        }
                    },
                )


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _flatten_response_content(content: Any) -> str:
    """Flatten a Responses-API content list into a single string.

    Accepts both ``input_text`` (incoming, user/developer turns) and
    ``output_text`` (incoming if the caller is replaying assistant turns).
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise AdapterParseError("input item 'content' must be a string or list")
    out: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            raise AdapterParseError("content list entries must be objects")
        ptype = part.get("type")
        if ptype in ("input_text", "output_text", "text"):
            text = part.get("text", "")
            if not isinstance(text, str):
                raise AdapterParseError("text part 'text' must be a string")
            if text:
                out.append(text)
        # Unknown / image / file parts — drop silently in v1.
    return "".join(out)


def _parse_tools(raw: Any) -> list[Tool]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise AdapterParseError("'tools' must be a list")
    out: list[Tool] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise AdapterParseError("each tool entry must be an object")
        # Responses-API function tool shape: {type, name, description, parameters}
        # (flat, NOT nested under a "function" sub-object like chat-completions).
        if entry.get("type") != "function":
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            raise AdapterParseError("tool requires a string 'name'")
        description = entry.get("description") or ""
        params = entry.get("parameters") or {}
        if not isinstance(params, dict):
            raise AdapterParseError("tool 'parameters' must be an object")
        out.append(Tool(name=name, description=description, input_schema=params))
    return out


def _parse_tool_choice(raw: Any) -> Any:
    if raw is None:
        return None
    if isinstance(raw, str):
        if raw in ("auto", "none"):
            return raw
        if raw == "required":
            return "any"
        raise AdapterParseError(f"unsupported tool_choice string: {raw!r}")
    if isinstance(raw, dict):
        if raw.get("type") == "function":
            name = raw.get("name")
            if not isinstance(name, str) or not name:
                raise AdapterParseError("tool_choice function requires 'name'")
            return {"type": "tool", "name": name}
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
    if isinstance(value, bool):
        raise AdapterParseError("numeric field must be a number")
    if isinstance(value, (int, float)):
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


__all__ = [
    "AdapterParseError",
    "CodexAdapter",
]
