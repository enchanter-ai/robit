"""enchanter.proxy.upstream — LiteLLM bridge for canonical requests.

This module owns the *only* contact point between the proxy core and a
real provider SDK.  Every other module in :mod:`enchanter.proxy` should
treat upstream calls as opaque coroutines that produce
:class:`~.canonical.CanonicalResponse` / :class:`~.canonical.CanonicalChunk`.

Authentication
--------------
LiteLLM resolves provider credentials from environment variables on its
own — we do **not** re-implement auth here.  Set the appropriate
variable(s) before calling :func:`call_upstream` or
:func:`stream_upstream`:

* Anthropic: ``ANTHROPIC_API_KEY``
* OpenAI:    ``OPENAI_API_KEY``
* Gemini:    ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``)
* Others:    see https://docs.litellm.ai/docs/providers

Pass-through auth (Wave 16.1)
-----------------------------
When the proxy server is started with ``--passthrough-auth`` the inbound
HTTP auth header is captured into ``CanonicalRequest.metadata`` under
two reserved keys:

* ``_enchanter_passthrough_auth`` (bool) — flips the kwarg-building
  branch below.
* ``_enchanter_inbound_auth`` (dict) — ``{kind, header_name, value}``
  describing the captured credential.

Both keys are **scrubbed from the metadata bag** before LiteLLM sees the
request kwargs, so they never leak into the upstream JSON body.  The
mapping from ``kind`` to LiteLLM kwargs is:

================  ============================================================
kind              LiteLLM kwargs
================  ============================================================
anthropic-api-key ``api_key=<token>``
anthropic-oauth   ``api_key="sk-ant-placeholder"`` +
                  ``extra_headers={"Authorization": "Bearer <token>"}``
openai-bearer     ``api_key=<token>``
gemini-api-key    ``api_key=<token>``
================  ============================================================

Honesty note: LiteLLM (v1.x) accepts ``extra_headers`` and forwards the
mapping verbatim on the outbound HTTP call.  Anthropic's OAuth flow
nonetheless requires LiteLLM's own auth validator to see a value
matching the ``sk-ant-...`` shape — hence the placeholder ``api_key``
when forwarding an OAuth bearer.  The actual auth the upstream sees is
the ``Authorization: Bearer`` header from ``extra_headers``.

The model string carried on :class:`~.canonical.CanonicalRequest.model` is
forwarded verbatim; LiteLLM picks the provider from its prefix (e.g.
``anthropic/claude-3-5-sonnet-20241022``, ``gpt-4o-mini``,
``gemini/gemini-1.5-pro``).

Parameter drop-through
----------------------
``litellm.drop_params = True`` is set at module import time so that
provider-incompatible kwargs (e.g. ``top_p`` on a provider that doesn't
accept it, ``stop_sequences`` for an o-series model) are silently
dropped instead of raising — the proxy should never fail a request
because of a benign hint that the upstream can ignore.

Errors
------
All LiteLLM exceptions raised inside :func:`call_upstream` /
:func:`stream_upstream` are re-raised wrapped in :class:`UpstreamError`
so callers (Wave 2 server) have a single exception type to catch and a
stable shape to render back to the client.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

import litellm

from .canonical import (
    CanonicalChunk,
    CanonicalRequest,
    CanonicalResponse,
    CanonicalUsage,
    ContentPart,
    TextPart,
    ToolUsePart,
    ToolResultPart,
)

# Silently drop kwargs the upstream provider doesn't understand.
litellm.drop_params = True

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pass-through auth metadata keys (Wave 16.1).
#
# These keys ride on ``CanonicalRequest.metadata`` from the server to this
# module and are scrubbed from the metadata bag before LiteLLM sees the
# kwargs.  Never let either key reach litellm.acompletion.
# ---------------------------------------------------------------------------

_PASSTHROUGH_FLAG_KEY: str = "_enchanter_passthrough_auth"
_INBOUND_AUTH_KEY: str = "_enchanter_inbound_auth"
_INTERNAL_METADATA_KEYS: frozenset[str] = frozenset(
    {_PASSTHROUGH_FLAG_KEY, _INBOUND_AUTH_KEY}
)
_OAUTH_PLACEHOLDER_API_KEY: str = "sk-ant-placeholder"


# ---------------------------------------------------------------------------
# Errors.
# ---------------------------------------------------------------------------


class UpstreamError(Exception):
    """Wrap any LiteLLM/provider error in a stable shape.

    Attributes
    ----------
    provider:
        Best-effort provider identifier (``"anthropic"``, ``"openai"``,
        ``"gemini"``, ``"unknown"`` — derived from the model string).
    status:
        HTTP status code if the underlying error exposes one, else ``None``.
    message:
        Human-readable error message from the upstream.
    """

    def __init__(
        self,
        provider: str,
        status: int | None,
        message: str,
    ) -> None:
        self.provider = provider
        self.status = status
        self.message = message
        super().__init__(f"[{provider} status={status}] {message}")


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


async def call_upstream(req: CanonicalRequest) -> CanonicalResponse:
    """Run a non-streaming completion through LiteLLM.

    The request is converted to LiteLLM (OpenAI-shaped) kwargs, dispatched
    via ``litellm.acompletion``, and the response coerced back into a
    :class:`~.canonical.CanonicalResponse`.
    """
    kwargs = _build_litellm_kwargs(req, stream=False)
    try:
        resp = await litellm.acompletion(**kwargs)
    except Exception as exc:  # noqa: BLE001 — wrap-and-reraise pattern
        raise _wrap_error(req.model, exc) from exc

    return _coerce_response(resp, requested_model=req.model)


async def stream_upstream(req: CanonicalRequest) -> AsyncIterator[CanonicalChunk]:
    """Run a streaming completion and yield canonical events.

    LiteLLM yields OpenAI-shaped chunks::

        {choices: [{delta: {content: str, tool_calls: [...]}, finish_reason}]}

    We map these into Anthropic-style lifecycle events
    (``message_start`` → ``content_block_start`` → ``text_delta`` * →
    ``content_block_stop`` → ``message_stop``) so downstream adapters can
    render down to any provider's chunk format without information loss.
    """
    kwargs = _build_litellm_kwargs(req, stream=True)
    try:
        stream = await litellm.acompletion(**kwargs)
    except Exception as exc:  # noqa: BLE001
        raise _wrap_error(req.model, exc) from exc

    async for chunk in _translate_stream(stream, req.model):
        yield chunk


# ---------------------------------------------------------------------------
# Request building.
# ---------------------------------------------------------------------------


def _build_litellm_kwargs(req: CanonicalRequest, *, stream: bool) -> dict[str, Any]:
    """Translate a :class:`CanonicalRequest` into LiteLLM kwargs.

    Wave 16.1: when ``metadata["_enchanter_passthrough_auth"]`` is True
    and ``metadata["_enchanter_inbound_auth"]`` carries a recognised
    credential descriptor, the auth is forwarded via ``api_key`` /
    ``extra_headers`` kwargs.  Both internal metadata keys are stripped
    before the ``metadata`` bag is handed to LiteLLM, so the credential
    never reaches the upstream JSON body.
    """
    messages: list[dict[str, Any]] = []

    if req.system:
        messages.append({"role": "system", "content": req.system})

    for msg in req.messages:
        messages.extend(_message_to_litellm(msg))

    kwargs: dict[str, Any] = {
        "model": req.model,
        "messages": messages,
        "stream": stream,
    }

    if req.temperature is not None:
        kwargs["temperature"] = req.temperature
    if req.top_p is not None:
        kwargs["top_p"] = req.top_p
    if req.max_tokens is not None:
        kwargs["max_tokens"] = req.max_tokens
    if req.stop_sequences:
        kwargs["stop"] = list(req.stop_sequences)
    if req.tools:
        kwargs["tools"] = [_tool_to_litellm(t) for t in req.tools]
    if req.tool_choice is not None:
        kwargs["tool_choice"] = _tool_choice_to_litellm(req.tool_choice)

    # --- Wave 16.1: pass-through auth + metadata scrubbing -----------------
    passthrough_flag = bool(req.metadata.get(_PASSTHROUGH_FLAG_KEY)) if req.metadata else False
    inbound_auth = req.metadata.get(_INBOUND_AUTH_KEY) if req.metadata else None

    if passthrough_flag:
        auth_kwargs = _auth_kwargs_for_inbound(inbound_auth, model=req.model)
        kwargs.update(auth_kwargs)

    # Hand LiteLLM a *scrubbed* metadata bag — never let the internal keys
    # ride into the upstream request body.
    if req.metadata:
        public_metadata = {
            k: v for k, v in req.metadata.items() if k not in _INTERNAL_METADATA_KEYS
        }
        if public_metadata:
            kwargs["metadata"] = public_metadata

    return kwargs


def _auth_kwargs_for_inbound(
    inbound_auth: Any,
    *,
    model: str,
) -> dict[str, Any]:
    """Map an inbound-auth descriptor to LiteLLM kwargs.

    Returns ``{}`` (and logs a warning) when no recognisable descriptor
    is present — that falls back to the env-var auth path.
    """
    if not isinstance(inbound_auth, dict):
        _log.warning(
            "proxy passthrough_auth enabled but no inbound auth captured "
            "for model %r; falling back to env-var provider auth.",
            model,
        )
        return {}

    kind = inbound_auth.get("kind")
    value = inbound_auth.get("value")
    if not isinstance(kind, str) or not isinstance(value, str) or not value:
        _log.warning(
            "proxy passthrough_auth: malformed inbound auth descriptor "
            "(kind=%r, has-value=%s); falling back to env-var auth.",
            kind, bool(value),
        )
        return {}

    if kind == "anthropic-api-key":
        return {"api_key": value}
    if kind == "anthropic-oauth":
        # LiteLLM's Anthropic adapter rejects None api_key but forwards
        # extra_headers verbatim; the placeholder satisfies the validator
        # while the Authorization header carries the real OAuth bearer.
        return {
            "api_key": _OAUTH_PLACEHOLDER_API_KEY,
            "extra_headers": {"Authorization": f"Bearer {value}"},
        }
    if kind == "openai-bearer":
        return {"api_key": value}
    if kind == "gemini-api-key":
        return {"api_key": value}

    _log.warning(
        "proxy passthrough_auth: unknown inbound auth kind %r; "
        "falling back to env-var auth.",
        kind,
    )
    return {}


def _message_to_litellm(msg: Any) -> list[dict[str, Any]]:
    """Convert one canonical Message to one-or-more OpenAI-shaped messages.

    A single canonical assistant message that contains text + tool_use
    becomes one OpenAI message with ``content`` + ``tool_calls``.  A user
    message containing tool_result parts becomes one OpenAI ``tool``-role
    message per part (OpenAI's wire format requires that).
    """
    text_chunks: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    tool_results: list[ToolResultPart] = []

    for part in msg.content:
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
        elif isinstance(part, ToolResultPart):
            tool_results.append(part)
        else:  # pragma: no cover — exhaustive guard
            raise TypeError(f"Unknown content part type: {type(part).__name__}")

    out: list[dict[str, Any]] = []

    # Tool results expand to one tool-role message each.
    for tr in tool_results:
        out.append(
            {
                "role": "tool",
                "tool_call_id": tr.tool_use_id,
                "content": tr.content,
            }
        )

    # The remaining text + tool_calls form a single message in the original role.
    if text_chunks or tool_calls:
        entry: dict[str, Any] = {"role": msg.role}
        if text_chunks:
            entry["content"] = "".join(text_chunks)
        else:
            entry["content"] = None
        if tool_calls:
            entry["tool_calls"] = tool_calls
        out.append(entry)

    return out


def _tool_to_litellm(tool: Any) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        },
    }


def _tool_choice_to_litellm(choice: Any) -> Any:
    """Map canonical tool_choice verbs into OpenAI's wire format."""
    if isinstance(choice, str):
        # "auto" → "auto", "any" → "required", "none" → "none"
        if choice == "any":
            return "required"
        return choice
    # Dict — assume already shaped or close to it.  Map Anthropic-style
    # {"type":"tool","name":"foo"} into OpenAI's
    # {"type":"function","function":{"name":"foo"}}.
    if isinstance(choice, dict) and choice.get("type") == "tool" and "name" in choice:
        return {"type": "function", "function": {"name": choice["name"]}}
    return choice


# ---------------------------------------------------------------------------
# Response coercion.
# ---------------------------------------------------------------------------


def _coerce_response(resp: Any, *, requested_model: str) -> CanonicalResponse:
    """Coerce a LiteLLM ModelResponse into a CanonicalResponse."""
    choice = resp.choices[0]
    message = choice.message
    finish_reason = getattr(choice, "finish_reason", None)

    content: list[ContentPart] = []

    text = getattr(message, "content", None)
    if text:
        content.append(TextPart(text=text))

    tool_calls = getattr(message, "tool_calls", None) or []
    for tc in tool_calls:
        fn = tc.function
        try:
            args = json.loads(fn.arguments) if fn.arguments else {}
        except (TypeError, ValueError):
            args = {"_raw": fn.arguments}
        content.append(
            ToolUsePart(
                id=tc.id,
                name=fn.name,
                input=args,
            )
        )

    usage_obj = getattr(resp, "usage", None)
    usage = CanonicalUsage(
        input_tokens=_safe_int(getattr(usage_obj, "prompt_tokens", 0)),
        output_tokens=_safe_int(getattr(usage_obj, "completion_tokens", 0)),
    )

    model = getattr(resp, "model", None) or requested_model

    return CanonicalResponse(
        model=model,
        content=tuple(content),
        stop_reason=_map_finish_reason(finish_reason, has_tool_use=bool(tool_calls)),
        usage=usage,
    )


def _map_finish_reason(
    raw: str | None,
    *,
    has_tool_use: bool,
) -> str | None:
    """Map OpenAI finish_reason into Anthropic stop_reason vocabulary."""
    if raw is None:
        return None
    if raw == "stop":
        return "end_turn"
    if raw == "length":
        return "max_tokens"
    if raw in ("tool_calls", "function_call"):
        return "tool_use"
    if raw == "content_filter":
        return "end_turn"  # Best-effort; canonical vocab has no filter slot.
    # Anthropic providers may surface their own values directly.
    if raw in ("end_turn", "max_tokens", "stop_sequence", "tool_use"):
        return raw
    # Unknown — when there are tool calls, prefer tool_use.
    if has_tool_use:
        return "tool_use"
    return "end_turn"


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# Stream translation.
# ---------------------------------------------------------------------------


async def _translate_stream(
    stream: AsyncIterator[Any],
    requested_model: str,
) -> AsyncIterator[CanonicalChunk]:
    """Translate a LiteLLM async chunk iterator into canonical events.

    Strategy:

    * Emit ``message_start`` before the first chunk.
    * Open a text content block lazily the first time we see text content.
    * Open a tool_use content block per distinct tool_call ``index``
      (OpenAI's streamed tool_calls list carries an ``index`` field for
      multi-tool fan-out).
    * Emit ``content_block_stop`` for each opened block as soon as we see
      ``finish_reason`` on the same/later chunk.
    * Emit ``message_delta`` carrying stop_reason + usage (if present),
      then ``message_stop``.
    """
    yielded_start = False
    text_block_index: int | None = None
    text_block_opened = False
    # Map OpenAI tool_call streaming index → canonical block index.
    tool_block_indices: dict[int, int] = {}
    next_block_index = 0
    finish_reason: str | None = None
    usage_payload: CanonicalUsage | None = None
    has_tool_use = False

    try:
        async for chunk in stream:
            if not yielded_start:
                yielded_start = True
                yield CanonicalChunk(type="message_start")

            choices = getattr(chunk, "choices", None) or []
            if not choices:
                # Usage-only sentinel chunk (some providers send this last).
                usage_obj = getattr(chunk, "usage", None)
                if usage_obj is not None:
                    usage_payload = CanonicalUsage(
                        input_tokens=_safe_int(getattr(usage_obj, "prompt_tokens", 0)),
                        output_tokens=_safe_int(
                            getattr(usage_obj, "completion_tokens", 0)
                        ),
                    )
                continue

            choice = choices[0]
            delta = getattr(choice, "delta", None)
            choice_finish = getattr(choice, "finish_reason", None)
            if choice_finish:
                finish_reason = choice_finish

            if delta is not None:
                # --- text delta ---
                text = getattr(delta, "content", None)
                if text:
                    if not text_block_opened:
                        text_block_index = next_block_index
                        next_block_index += 1
                        text_block_opened = True
                        yield CanonicalChunk(
                            type="content_block_start",
                            index=text_block_index,
                            block_kind="text",
                        )
                    yield CanonicalChunk(
                        type="text_delta",
                        index=text_block_index or 0,
                        text=text,
                    )

                # --- tool_call deltas ---
                #
                # LiteLLM delivers tool id + function.name on the *first*
                # delta for a given index.  We need that metadata on the
                # ``content_block_start`` event so adapters don't have to
                # synthesise placeholders.  Strategy: open the block
                # lazily on the first delta we see for an index, reading
                # id/name from that same delta, and emit
                # ``content_block_start`` (with block_kind="tool_use",
                # tool_id, tool_name) BEFORE the corresponding
                # ``input_json_delta``.
                tcalls = getattr(delta, "tool_calls", None) or []
                for tcall in tcalls:
                    has_tool_use = True
                    src_idx = getattr(tcall, "index", 0) or 0
                    fn = getattr(tcall, "function", None)
                    args_fragment = getattr(fn, "arguments", None) if fn else None
                    if src_idx not in tool_block_indices:
                        block_idx = next_block_index
                        next_block_index += 1
                        tool_block_indices[src_idx] = block_idx
                        tcall_id = getattr(tcall, "id", None)
                        fn_name = getattr(fn, "name", None) if fn else None
                        yield CanonicalChunk(
                            type="content_block_start",
                            index=block_idx,
                            block_kind="tool_use",
                            tool_id=tcall_id,
                            tool_name=fn_name,
                        )
                    block_idx = tool_block_indices[src_idx]
                    if args_fragment:
                        yield CanonicalChunk(
                            type="input_json_delta",
                            index=block_idx,
                            partial_json=args_fragment,
                        )

            # Pull usage off the chunk if the provider attached it.
            usage_obj = getattr(chunk, "usage", None)
            if usage_obj is not None:
                usage_payload = CanonicalUsage(
                    input_tokens=_safe_int(getattr(usage_obj, "prompt_tokens", 0)),
                    output_tokens=_safe_int(
                        getattr(usage_obj, "completion_tokens", 0)
                    ),
                )
    except Exception as exc:  # noqa: BLE001
        raise _wrap_error(requested_model, exc) from exc

    # Close any open content blocks.
    if text_block_opened and text_block_index is not None:
        yield CanonicalChunk(type="content_block_stop", index=text_block_index)
    for block_idx in tool_block_indices.values():
        yield CanonicalChunk(type="content_block_stop", index=block_idx)

    stop_reason = _map_finish_reason(finish_reason, has_tool_use=has_tool_use)
    if stop_reason is not None or usage_payload is not None:
        yield CanonicalChunk(
            type="message_delta",
            stop_reason=stop_reason,
            usage=usage_payload,
        )

    yield CanonicalChunk(type="message_stop")


# ---------------------------------------------------------------------------
# Error wrapping.
# ---------------------------------------------------------------------------


def _wrap_error(model: str, exc: BaseException) -> UpstreamError:
    """Best-effort coercion of any provider/LiteLLM exception into UpstreamError."""
    if isinstance(exc, UpstreamError):
        return exc
    provider = _provider_from_model(model)
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(exc, "http_status", None)
    if status is None:
        response = getattr(exc, "response", None)
        if response is not None:
            status = getattr(response, "status_code", None)
    message = getattr(exc, "message", None) or str(exc)
    return UpstreamError(provider=provider, status=status, message=message)


def _provider_from_model(model: str) -> str:
    """Heuristic mapping from model string → provider id."""
    if "/" in model:
        prefix = model.split("/", 1)[0].lower()
        # LiteLLM's known prefixes.
        if prefix in {
            "anthropic",
            "openai",
            "gemini",
            "google",
            "vertex_ai",
            "azure",
            "bedrock",
            "groq",
            "mistral",
            "cohere",
            "ollama",
        }:
            return prefix
    lower = model.lower()
    if "claude" in lower:
        return "anthropic"
    if lower.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    if "gemini" in lower:
        return "gemini"
    return "unknown"


__all__ = [
    "UpstreamError",
    "call_upstream",
    "stream_upstream",
    # Wave 16.1 — exported so server.py can stamp them consistently.
    "PASSTHROUGH_FLAG_KEY",
    "INBOUND_AUTH_KEY",
]


# Public aliases for the metadata keys (server.py imports these to avoid
# string drift).  Wave 16.3 may also consume them.
PASSTHROUGH_FLAG_KEY = _PASSTHROUGH_FLAG_KEY
INBOUND_AUTH_KEY = _INBOUND_AUTH_KEY
