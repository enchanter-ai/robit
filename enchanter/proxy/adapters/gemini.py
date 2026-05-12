"""enchanter.proxy.adapters.gemini — Google Gemini wire adapter.

Translates between Gemini's ``/v1beta/models/<model>:{generateContent,
streamGenerateContent}`` JSON / SSE wire format and the provider-neutral
:mod:`enchanter.proxy.canonical` dataclasses.

The adapter is **pure wire translation** — it does not call any upstream
provider SDK.  The proxy server is responsible for handing the parsed
:class:`~enchanter.proxy.canonical.CanonicalRequest` to
:func:`enchanter.proxy.upstream.call_upstream` (or ``stream_upstream``)
and then handing the result back to :meth:`GeminiAdapter.render_response`
(or ``render_stream``).

Wire-format references:

* https://ai.google.dev/api/rest/v1beta/models/generateContent
* https://ai.google.dev/api/rest/v1beta/models/streamGenerateContent

Gemini quirks (relevant to Wave 2):

* **No tool-call IDs on the wire.**  Gemini's ``functionCall`` /
  ``functionResponse`` parts pair by ``name`` + positional order rather
  than by id.  We synthesise ids for parsed ``functionCall`` parts and
  set ``tool_use_id=""`` (empty string) for ``functionResponse`` parts.
* **No ``[DONE]`` sentinel.**  ``streamGenerateContent`` simply closes
  the connection after the final event.
* **Tool-call args are not streamed incrementally.**  Each
  ``functionCall`` is emitted as a single SSE event with the complete
  ``args`` object; Gemini does not chunk the JSON.
"""

from __future__ import annotations

import json
import re
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


# /v1beta/models/<model>:<verb> — model id may contain dots and dashes
# (e.g. gemini-1.5-flash) but no slashes or colons.
_PATH_RE = re.compile(
    r"^/v1beta/models/(?P<model>[^/:]+):(?P<verb>generateContent|streamGenerateContent)$"
)


from .errors import AdapterParseError  # re-exported by adapters/__init__.py


# Canonical → Gemini stop_reason mapping.
_STOP_REASON_MAP = {
    "end_turn": "STOP",
    "max_tokens": "MAX_TOKENS",
    "stop_sequence": "STOP",
    "tool_use": "STOP",  # Gemini has no dedicated tool finish reason.
    None: "STOP",
}


class GeminiAdapter:
    """Wire adapter for Gemini's ``generateContent`` family of endpoints."""

    # Exposed for routing tables; the regex above is the actual matcher.
    path_pattern: ClassVar[str] = _PATH_RE.pattern

    # ------------------------------------------------------------------
    # Routing.
    # ------------------------------------------------------------------

    @staticmethod
    def matches(method: str, path: str) -> bool:
        if method != "POST":
            return False
        pure_path = path.split("?", 1)[0]
        return _PATH_RE.match(pure_path) is not None

    # ------------------------------------------------------------------
    # Request parsing.
    # ------------------------------------------------------------------

    @staticmethod
    def parse_request(
        body: bytes, path: str, headers: dict[str, str]
    ) -> CanonicalRequest:
        """Parse a Gemini ``generateContent`` body into a CanonicalRequest.

        The model id and ``stream`` flag come from the URL path; the body
        carries everything else.  Tool-call ids are synthesised because
        Gemini does not carry them on the wire (see module docstring).
        """
        pure_path = path.split("?", 1)[0]
        match = _PATH_RE.match(pure_path)
        if match is None:
            raise AdapterParseError(f"invalid Gemini path: {path!r}")
        model = match.group("model")
        verb = match.group("verb")
        stream = verb == "streamGenerateContent"

        try:
            payload = json.loads(body)
        except (ValueError, TypeError) as exc:
            raise AdapterParseError(f"invalid JSON body: {exc}") from exc

        if not isinstance(payload, dict):
            raise AdapterParseError("request body must be a JSON object")

        # ---- system instruction (top-level, not a message role) ----
        system = _parse_system_instruction(payload.get("systemInstruction"))

        # ---- contents → messages ----
        raw_contents = payload.get("contents", [])
        if not isinstance(raw_contents, list):
            raise AdapterParseError("'contents' must be a list")
        messages = tuple(_parse_content_entry(c) for c in raw_contents)

        # ---- tools (flatten functionDeclarations across blocks) ----
        tools = _parse_tools(payload.get("tools", []) or [])

        # ---- toolConfig.functionCallingConfig ----
        tool_choice = _parse_tool_config(payload.get("toolConfig"))

        # ---- generationConfig → scalar fields ----
        gen_cfg = payload.get("generationConfig") or {}
        if not isinstance(gen_cfg, dict):
            raise AdapterParseError("'generationConfig' must be an object")

        temperature = _opt_float(gen_cfg.get("temperature"), "temperature")
        top_p = _opt_float(gen_cfg.get("topP"), "topP")
        max_tokens = _opt_int(gen_cfg.get("maxOutputTokens"), "maxOutputTokens")

        stop_sequences_raw = gen_cfg.get("stopSequences", []) or []
        if not isinstance(stop_sequences_raw, list):
            raise AdapterParseError("'stopSequences' must be a list")
        stop_sequences = tuple(str(s) for s in stop_sequences_raw)

        return CanonicalRequest(
            model=model,
            messages=messages,
            system=system if system else None,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
            stream=stream,
            metadata={},
        )

    # ------------------------------------------------------------------
    # Response rendering — non-streaming.
    # ------------------------------------------------------------------

    @staticmethod
    def render_response(resp: CanonicalResponse) -> bytes:
        """Render a CanonicalResponse into Gemini ``GenerateContentResponse`` bytes."""
        parts = [_render_content_part(p) for p in resp.content]
        finish_reason = _STOP_REASON_MAP.get(resp.stop_reason, "STOP")
        body = {
            "candidates": [
                {
                    "content": {"role": "model", "parts": parts},
                    "finishReason": finish_reason,
                    "index": 0,
                    "safetyRatings": [],
                }
            ],
            "usageMetadata": {
                "promptTokenCount": resp.usage.input_tokens,
                "candidatesTokenCount": resp.usage.output_tokens,
                "totalTokenCount": resp.usage.input_tokens + resp.usage.output_tokens,
            },
            "modelVersion": resp.model,
        }
        return json.dumps(body).encode("utf-8")

    # ------------------------------------------------------------------
    # Response rendering — streaming SSE.
    # ------------------------------------------------------------------

    @staticmethod
    async def render_stream(
        stream: AsyncIterator[CanonicalChunk],
    ) -> AsyncIterator[bytes]:
        """Translate canonical chunks into Gemini SSE event bytes.

        Each yielded ``bytes`` is ``data: <json>\\n\\n`` UTF-8 encoded.
        Gemini does NOT emit a ``[DONE]`` sentinel; the stream simply
        closes after the final event.

        Block buffering strategy:

        * ``message_start`` / ``content_block_start`` emit no event.
        * ``text_delta`` emits one event with ``parts=[{"text": <delta>}]``.
        * ``input_json_delta`` is accumulated per index; on
          ``content_block_stop`` we emit one event with the complete
          ``functionCall``.  Gemini does not stream tool-call args
          incrementally — it always sends the whole thing in one shot.
        * ``message_delta`` is buffered; we attach its ``finishReason``
          and ``usage`` to the **next** event we emit, or — if no more
          events follow — synthesise a final empty event carrying just
          the finishReason and usageMetadata.
        * ``message_stop`` simply closes the stream.

        Tool-name gap: the canonical ``content_block_start`` does NOT
        carry the tool's name/id.  We fall back to two heuristics so the
        emitted ``functionCall`` is still well-formed:

        1. If the accumulated partial_json parses as a JSON object that
           includes a ``"name"`` key (Anthropic-style tool calls round-
           tripping through the substrate), use it.  Otherwise:
        2. Emit ``name=""`` and document the limitation here.  Wave 2
           upstream callers writing Gemini-bound tool streams should
           prefer to emit a single ``input_json_delta`` carrying the
           complete arguments JSON — incremental fragments would
           accumulate into invalid JSON anyway given Gemini's single-
           event constraint.

        This gap is REPORTED back to the Wave 1 caller; it does not
        block the adapter's basic operation.
        """
        # Per-index buffer of partial_json fragments for tool_use blocks.
        tool_buffers: dict[int, list[str]] = {}
        # Per-index authoritative tool name captured from
        # content_block_start (when the canonical chunk carries it).
        # If absent, we fall back to the self-describing-payload decode
        # in ``_decode_tool_args`` for backwards compat with legacy
        # producers.
        tool_names: dict[int, str] = {}
        # Pending message-level updates from a message_delta chunk that
        # has been seen but not yet flushed onto an outgoing event.
        pending_finish_reason: str | None = None
        pending_usage: dict[str, int] | None = None
        emitted_any = False
        # Track whether we've emitted a final event with finishReason.
        final_emitted = False

        def make_event(parts: list[dict[str, Any]]) -> bytes:
            nonlocal pending_finish_reason, pending_usage, emitted_any, final_emitted
            candidate: dict[str, Any] = {
                "content": {"role": "model", "parts": parts},
                "index": 0,
            }
            payload: dict[str, Any] = {"candidates": [candidate]}
            if pending_finish_reason is not None:
                candidate["finishReason"] = pending_finish_reason
                final_emitted = True
                pending_finish_reason = None
            if pending_usage is not None:
                payload["usageMetadata"] = pending_usage
                pending_usage = None
            emitted_any = True
            return b"data: " + json.dumps(payload).encode("utf-8") + b"\n\n"

        async for chunk in stream:
            if chunk.type == "message_start":
                continue
            if chunk.type == "content_block_start":
                # If the canonical chunk carries the tool name (the new
                # contract), capture it now so we can emit it on
                # content_block_stop without depending on a
                # self-describing payload.  Legacy producers leave
                # block_kind=None — we still defer in that case.
                if (
                    chunk.block_kind == "tool_use"
                    and chunk.tool_name is not None
                ):
                    tool_names[chunk.index] = chunk.tool_name
                # No SSE event emitted here either way (Gemini doesn't
                # have a block-open wire event; the block reveals itself
                # on the first delta).
                continue
            if chunk.type == "text_delta":
                if chunk.text is None or chunk.text == "":
                    continue
                yield make_event([{"text": chunk.text}])
            elif chunk.type == "input_json_delta":
                # Buffer; emit on content_block_stop.
                tool_buffers.setdefault(chunk.index, []).append(
                    chunk.partial_json or ""
                )
            elif chunk.type == "content_block_stop":
                fragments = tool_buffers.pop(chunk.index, None)
                if fragments is None:
                    # Text block stopped — nothing to flush.
                    continue
                accumulated = "".join(fragments)
                # Preferred path: tool name was captured from
                # content_block_start.  Accumulated bytes are the raw
                # args JSON.  Legacy path: no captured name — fall back
                # to the self-describing-payload decode.
                if chunk.index in tool_names:
                    name = tool_names.pop(chunk.index)
                    args = _decode_tool_args_raw(accumulated)
                else:
                    name, args = _decode_tool_args(accumulated)
                yield make_event(
                    [{"functionCall": {"name": name, "args": args}}]
                )
            elif chunk.type == "message_delta":
                # Buffer finish_reason + usage; flush onto next event or
                # synthesise a final one after the loop.
                if chunk.stop_reason is not None:
                    pending_finish_reason = _STOP_REASON_MAP.get(
                        chunk.stop_reason, "STOP"
                    )
                elif pending_finish_reason is None:
                    # message_delta with no stop_reason — leave as None
                    # so the final synthesiser still fills STOP.
                    pass
                if chunk.usage is not None:
                    pending_usage = {
                        "promptTokenCount": chunk.usage.input_tokens,
                        "candidatesTokenCount": chunk.usage.output_tokens,
                        "totalTokenCount": chunk.usage.input_tokens
                        + chunk.usage.output_tokens,
                    }
            elif chunk.type == "message_stop":
                # No event; the stream closing IS the signal.
                continue
            # Unknown chunk types silently dropped (forward compat).

        # Always emit a final event carrying finishReason (and usage if
        # known), even if no message_delta was observed.
        if not final_emitted:
            if pending_finish_reason is None:
                pending_finish_reason = "STOP"
            # Final event has an empty parts list (Gemini's wire shape
            # for finish-only events).
            yield make_event([])
        elif pending_usage is not None:
            # Edge case: usage arrived after finish_reason was already
            # flushed.  Emit a trailing event with just usage.
            payload: dict[str, Any] = {
                "candidates": [
                    {
                        "content": {"role": "model", "parts": []},
                        "index": 0,
                    }
                ],
                "usageMetadata": pending_usage,
            }
            yield b"data: " + json.dumps(payload).encode("utf-8") + b"\n\n"


# ---------------------------------------------------------------------------
# Parsing helpers.
# ---------------------------------------------------------------------------


def _parse_system_instruction(raw: Any) -> str | None:
    """Flatten ``systemInstruction.parts[].text`` into a single string."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise AdapterParseError("'systemInstruction' must be an object")
    parts = raw.get("parts", [])
    if not isinstance(parts, list):
        raise AdapterParseError("'systemInstruction.parts' must be a list")
    chunks: list[str] = []
    for part in parts:
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            chunks.append(part["text"])
    return "".join(chunks) if chunks else None


def _parse_content_entry(raw: Any) -> Message:
    """Map one Gemini ``contents[]`` entry to a canonical :class:`Message`.

    Role mapping:

    * ``"user"``   → ``"user"``
    * ``"model"``  → ``"assistant"``

    Gemini's wire format has no ``system`` or ``tool`` role inside
    ``contents``; function responses live as parts inside a ``user``
    message.
    """
    if not isinstance(raw, dict):
        raise AdapterParseError("each content entry must be an object")
    raw_role = raw.get("role", "user")
    if raw_role == "model":
        role: Any = "assistant"
    elif raw_role == "user":
        role = "user"
    else:
        raise AdapterParseError(f"invalid contents role: {raw_role!r}")
    parts_raw = raw.get("parts")
    if not isinstance(parts_raw, list):
        raise AdapterParseError("content 'parts' must be a list")
    content = tuple(_parse_part(p) for p in parts_raw)
    return Message(role=role, content=content)


def _parse_part(raw: Any) -> ContentPart:
    """Map one Gemini ``parts[]`` entry to a canonical :class:`ContentPart`.

    Gemini does NOT carry tool-call ids on the wire.  We synthesise an
    id per ``functionCall`` using ``secrets.token_hex(4)`` so downstream
    consumers can reference the call.  This means a round-trip through
    canonical → Gemini → canonical does NOT preserve the original ids,
    but the substrate's tool-call topology (which call produced which
    response) is preserved through positional ordering only.
    """
    if not isinstance(raw, dict):
        raise AdapterParseError("each part must be an object")

    if "text" in raw:
        text = raw["text"]
        if not isinstance(text, str):
            raise AdapterParseError("part 'text' must be a string")
        return TextPart(text=text)

    if "functionCall" in raw:
        fc = raw["functionCall"]
        if not isinstance(fc, dict):
            raise AdapterParseError("'functionCall' must be an object")
        name = fc.get("name")
        if not isinstance(name, str):
            raise AdapterParseError("functionCall 'name' must be a string")
        args = fc.get("args", {})
        if args is None:
            args = {}
        if not isinstance(args, dict):
            raise AdapterParseError("functionCall 'args' must be an object")
        # Synthesise an id — Gemini doesn't carry one on the wire.
        return ToolUsePart(id=f"tool_{secrets.token_hex(4)}", name=name, input=args)

    if "functionResponse" in raw:
        fr = raw["functionResponse"]
        if not isinstance(fr, dict):
            raise AdapterParseError("'functionResponse' must be an object")
        name = fr.get("name", "")
        response = fr.get("response", {})
        # Stringify the whole response payload — canonical
        # ToolResultPart.content is a string.  Embed the name so it can
        # be recovered on rendering (Gemini's wire requires the name).
        try:
            content = json.dumps({"name": name, "response": response})
        except (TypeError, ValueError) as exc:
            raise AdapterParseError(
                f"functionResponse not JSON-serialisable: {exc}"
            ) from exc
        # Gemini doesn't carry tool_use_ids on the wire — leave empty.
        return ToolResultPart(tool_use_id="", content=content, is_error=False)

    raise AdapterParseError(f"unknown part shape: {sorted(raw.keys())!r}")


def _parse_tools(raw: Any) -> tuple[Tool, ...]:
    """Flatten ``tools[].functionDeclarations[]`` into a tuple of Tools."""
    if not isinstance(raw, list):
        raise AdapterParseError("'tools' must be a list")
    out: list[Tool] = []
    for block in raw:
        if not isinstance(block, dict):
            raise AdapterParseError("each tool block must be an object")
        decls = block.get("functionDeclarations", []) or []
        if not isinstance(decls, list):
            raise AdapterParseError("'functionDeclarations' must be a list")
        for decl in decls:
            if not isinstance(decl, dict):
                raise AdapterParseError("each functionDeclaration must be an object")
            name = decl.get("name")
            if not isinstance(name, str):
                raise AdapterParseError("functionDeclaration 'name' must be a string")
            description = decl.get("description", "")
            if not isinstance(description, str):
                raise AdapterParseError(
                    "functionDeclaration 'description' must be a string"
                )
            parameters = decl.get("parameters", {}) or {}
            if not isinstance(parameters, dict):
                raise AdapterParseError(
                    "functionDeclaration 'parameters' must be an object"
                )
            out.append(
                Tool(name=name, description=description, input_schema=parameters)
            )
    return tuple(out)


def _parse_tool_config(raw: Any) -> Any:
    """Map ``toolConfig.functionCallingConfig`` to a canonical tool_choice."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise AdapterParseError("'toolConfig' must be an object")
    fcc = raw.get("functionCallingConfig")
    if fcc is None:
        return None
    if not isinstance(fcc, dict):
        raise AdapterParseError("'functionCallingConfig' must be an object")
    mode = fcc.get("mode")
    if mode is None:
        return None
    if not isinstance(mode, str):
        raise AdapterParseError("functionCallingConfig 'mode' must be a string")
    norm = mode.upper()
    if norm == "AUTO":
        return "auto"
    if norm == "NONE":
        return "none"
    if norm == "ANY":
        allowed = fcc.get("allowedFunctionNames") or []
        if (
            isinstance(allowed, list)
            and len(allowed) == 1
            and isinstance(allowed[0], str)
        ):
            return {"type": "tool", "name": allowed[0]}
        return "any"
    raise AdapterParseError(f"unknown functionCallingConfig mode: {mode!r}")


def _opt_float(raw: Any, field_name: str) -> float | None:
    if raw is None:
        return None
    if not isinstance(raw, (int, float)) or isinstance(raw, bool):
        raise AdapterParseError(f"'{field_name}' must be a number")
    return float(raw)


def _opt_int(raw: Any, field_name: str) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise AdapterParseError(f"'{field_name}' must be an integer")
    return int(raw)


# ---------------------------------------------------------------------------
# Rendering helpers.
# ---------------------------------------------------------------------------


def _render_content_part(part: ContentPart) -> dict[str, Any]:
    """Render a canonical content part as a Gemini ``parts[]`` entry."""
    if isinstance(part, TextPart):
        return {"text": part.text}
    if isinstance(part, ToolUsePart):
        return {"functionCall": {"name": part.name, "args": part.input}}
    if isinstance(part, ToolResultPart):
        # Recover the wrapper shape we wrote in parse_part if possible;
        # otherwise pass the raw string through as the response body.
        try:
            decoded = json.loads(part.content)
        except (ValueError, TypeError):
            decoded = {"content": part.content}
        if (
            isinstance(decoded, dict)
            and "name" in decoded
            and "response" in decoded
        ):
            return {
                "functionResponse": {
                    "name": decoded["name"],
                    "response": decoded["response"],
                }
            }
        return {
            "functionResponse": {
                "name": "",
                "response": decoded
                if isinstance(decoded, dict)
                else {"content": part.content},
            }
        }
    raise TypeError(f"unknown content part type: {type(part).__name__}")


def _decode_tool_args_raw(accumulated: str) -> dict[str, Any]:
    """Decode accumulated input_json_delta fragments as a raw args object.

    Used when the tool name is already known (captured from
    ``content_block_start.tool_name``) so the accumulated payload is
    expected to be the raw JSON arguments object — not a self-describing
    ``{"name": ..., "args": ...}`` wrapper.  Returns ``{}`` on empty or
    unparseable input rather than raising; a malformed args fragment
    should not crash the renderer.
    """
    if not accumulated.strip():
        return {}
    try:
        parsed = json.loads(accumulated)
    except (ValueError, TypeError):
        return {}
    if isinstance(parsed, dict):
        return parsed
    return {}


def _decode_tool_args(accumulated: str) -> tuple[str, dict[str, Any]]:
    """Decode an accumulated input_json_delta blob into (name, args).

    Canonical ``CanonicalChunk`` for ``content_block_start`` carries no
    tool name or id (Wave 0 gap).  We fall back to: if the blob is a JSON
    object containing a top-level ``"name"`` and ``"args"``, treat it as
    a self-describing payload; otherwise treat the whole blob as the
    args object and return ``name=""``.

    Returning ``name=""`` is documented at the call site; Wave 2 should
    prefer to emit tool calls as a single complete JSON blob so this
    decoder always has a clean parse to work with.
    """
    if not accumulated.strip():
        return "", {}
    try:
        parsed = json.loads(accumulated)
    except (ValueError, TypeError):
        # Could not decode — return empty args rather than raising; a
        # malformed tool-call stream should not crash the renderer.
        return "", {}
    if isinstance(parsed, dict) and "name" in parsed and "args" in parsed:
        name = parsed["name"] if isinstance(parsed["name"], str) else ""
        args = parsed["args"] if isinstance(parsed["args"], dict) else {}
        return name, args
    if isinstance(parsed, dict):
        return "", parsed
    return "", {}


__all__ = [
    "GeminiAdapter",
    "AdapterParseError",
]
