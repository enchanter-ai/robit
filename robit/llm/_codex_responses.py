"""robit.llm._codex_responses — Responses API helpers (Codex CLI wire format).

Shared between :class:`robit.llm.chatgpt_client.ChatGptClient` (which posts
directly to ``chatgpt.com/backend-api/codex/responses``) and
:class:`robit.proxy.adapters.codex.CodexAdapter` (which parses / renders
the same wire format on the proxy frontend).

Reference: ``docs/architecture/audits/codex-protocol.md`` (Wave 16.0 audit).

Scope (v1)
----------
* Text-only request/response. ``input`` items carry one or more
  ``input_text`` content parts; the assistant reply is rendered as a single
  ``output_text`` part.
* Non-streaming (``parse_responses_completion``) and SSE chunk parsing
  (``parse_responses_chunk``) — text deltas only.
* Tool-call streaming (``response.function_call_arguments.delta``) is **not**
  parsed in v1; callers that send tools will degrade to no-tool-call
  responses. Documented limitation.

Out of scope
------------
* ``reasoning.encrypted_content`` + ``previous_response_id`` carry-forward —
  the proxy is stateless across turns for v1.
* Image / file input parts.
* WebSocket transport (``OpenAI-Beta: responses_websockets=...``) — HTTP-SSE
  fallback only.
* ``x-oai-attestation`` — deferred (Wave 16.0 flagged).
* ``x-codex-turn-state`` — pass-through verbatim at the HTTP layer; never
  synthesised here.
"""

from __future__ import annotations

import json
import secrets
import time
from typing import Any

from .types import CompletionRequest, CompletionResponse
from .types import Message as LlmMessage


# ---------------------------------------------------------------------------
# CompletionRequest → Responses-API JSON body
# ---------------------------------------------------------------------------


def build_responses_request(
    req: CompletionRequest | Any,
    *,
    stream: bool | None = None,
) -> dict[str, Any]:
    """Build a Responses-API JSON body from a :class:`CompletionRequest`.

    The shape matches Codex CLI's `ResponsesApiRequest` (per Wave 16.0):

    .. code-block:: json

        {
          "model": "...",
          "instructions": "<system>",
          "input": [
            {"type": "message", "role": "user",
             "content": [{"type": "input_text", "text": "..."}]}
          ],
          "tools": [],
          "tool_choice": "auto",
          "stream": false,
          "store": false
        }

    Notes
    -----
    * ``instructions`` is required by the upstream when present; we omit it
      when ``req.system`` is None to keep the body terse.
    * ``store: false`` matches Codex's default — no server-side history.
    * If ``stream`` is left ``None`` we honour ``getattr(req, "stream", False)``
      (CompletionRequest currently has no stream flag — defaults to False).
    """
    body: dict[str, Any] = {
        "model": req.model,
        "input": [_message_to_input_item(m) for m in req.messages],
        "store": False,
        "stream": bool(stream if stream is not None else getattr(req, "stream", False)),
    }
    if getattr(req, "system", None):
        body["instructions"] = req.system
    if getattr(req, "temperature", None) is not None:
        body["temperature"] = req.temperature
    if getattr(req, "max_tokens", None) is not None:
        body["max_output_tokens"] = req.max_tokens
    if getattr(req, "stop_sequences", None):
        body["stop"] = list(req.stop_sequences)
    tools = getattr(req, "tools", None)
    if tools:
        body["tools"] = list(tools)
        body["tool_choice"] = "auto"
    return body


def _message_to_input_item(msg: LlmMessage | Any) -> dict[str, Any]:
    """Convert one :class:`Message` into a Responses-API ``input`` entry."""
    role = msg.role
    # CompletionRequest only carries "user" / "assistant"; both are valid here.
    return {
        "type": "message",
        "role": role,
        "content": [{"type": "input_text", "text": msg.content}],
    }


# ---------------------------------------------------------------------------
# Responses-API JSON body → CompletionResponse (non-streaming)
# ---------------------------------------------------------------------------


def parse_responses_completion(
    body: dict[str, Any],
    *,
    requested_model: str | None = None,
) -> CompletionResponse:
    """Parse a non-streaming Responses-API response body into a CompletionResponse.

    Expected shape::

        {
          "id": "resp_...",
          "object": "response",
          "model": "gpt-5-codex",
          "status": "completed",
          "output": [
            {"type": "message", "role": "assistant",
             "content": [{"type": "output_text", "text": "..."}]}
          ],
          "usage": {"input_tokens": N, "output_tokens": N, "total_tokens": N}
        }
    """
    output = body.get("output") or []
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        itype = item.get("type")
        if itype == "message":
            for part in item.get("content") or []:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                if ptype in ("output_text", "text"):
                    text = part.get("text")
                    if isinstance(text, str):
                        text_parts.append(text)
        elif itype == "function_call":
            args_raw = item.get("arguments") or ""
            try:
                args = json.loads(args_raw) if args_raw else {}
            except (TypeError, ValueError):
                args = {"_raw": args_raw}
            tool_calls.append(
                {
                    "id": item.get("call_id") or item.get("id") or "",
                    "name": item.get("name") or "",
                    "input": args,
                }
            )

    usage = body.get("usage") or {}
    return CompletionResponse(
        text="".join(text_parts),
        model=body.get("model") or requested_model or "unknown",
        stop_reason=body.get("status") or "completed",
        input_tokens=_safe_int(usage.get("input_tokens")),
        output_tokens=_safe_int(usage.get("output_tokens")),
        tool_calls=tool_calls,
    )


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# Streaming SSE line → canonical-ish parsed event
# ---------------------------------------------------------------------------


# Canonical event names we emit downstream of `parse_responses_chunk`. Kept as
# constants so the adapter and tests can refer to them without duplicating
# string literals.
EVT_RESPONSE_CREATED = "response.created"
EVT_OUTPUT_TEXT_DELTA = "response.output_text.delta"
EVT_OUTPUT_TEXT_DONE = "response.output_text.done"
EVT_RESPONSE_COMPLETED = "response.completed"


def parse_responses_chunk(event: str, data: str) -> dict[str, Any] | None:
    """Parse one SSE ``(event, data)`` pair into a normalised dict.

    Returns ``None`` for events we do not handle in v1 (e.g.
    ``response.function_call_arguments.delta``,
    ``response.reasoning_summary_text.delta``).

    The return shape is intentionally simple; downstream callers map it to
    their own type (canonical chunk, CompletionResponse, etc.).
    """
    if not event:
        return None
    try:
        payload = json.loads(data) if data else {}
    except (TypeError, ValueError):
        return None
    if event == EVT_OUTPUT_TEXT_DELTA:
        return {"event": event, "delta": payload.get("delta", "")}
    if event == EVT_RESPONSE_COMPLETED:
        response = payload.get("response") or {}
        usage = response.get("usage") or {}
        return {
            "event": event,
            "usage": {
                "input_tokens": _safe_int(usage.get("input_tokens")),
                "output_tokens": _safe_int(usage.get("output_tokens")),
            },
            "model": response.get("model"),
        }
    if event == EVT_RESPONSE_CREATED:
        return {"event": event, "response": payload.get("response") or {}}
    if event == EVT_OUTPUT_TEXT_DONE:
        return {"event": event}
    return None


# ---------------------------------------------------------------------------
# SSE rendering helpers (used by CodexAdapter.render_stream)
# ---------------------------------------------------------------------------


def render_sse_event(event: str, payload: dict[str, Any]) -> bytes:
    """Render one SSE event as ``event: …\\ndata: …\\n\\n`` bytes."""
    line = f"event: {event}\ndata: {json.dumps(payload)}\n\n"
    return line.encode("utf-8")


def new_response_id() -> str:
    """Generate a Responses-API-style response id (``resp_<hex>``)."""
    return f"resp_{secrets.token_hex(12)}"


def new_message_id() -> str:
    """Generate a Responses-API output-item id (``msg_<hex>``)."""
    return f"msg_{secrets.token_hex(12)}"


def now_ts() -> int:
    return int(time.time())


__all__ = [
    "EVT_RESPONSE_CREATED",
    "EVT_OUTPUT_TEXT_DELTA",
    "EVT_OUTPUT_TEXT_DONE",
    "EVT_RESPONSE_COMPLETED",
    "build_responses_request",
    "new_message_id",
    "new_response_id",
    "now_ts",
    "parse_responses_chunk",
    "parse_responses_completion",
    "render_sse_event",
]
