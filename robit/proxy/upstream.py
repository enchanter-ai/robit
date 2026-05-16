"""robit.proxy.upstream — LiteLLM bridge for canonical requests.

This module owns the *only* contact point between the proxy core and a
real provider SDK.  Every other module in :mod:`robit.proxy` should
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

import asyncio
import json
import urllib.error
import urllib.request
from types import SimpleNamespace
from typing import Any, AsyncIterator

import litellm

from ..llm._codex_responses import (
    build_responses_request,
    parse_responses_completion,
)
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

# Upstream endpoint for ChatGPT-login Codex mode. Hardcoded per Wave 16.0
# audit (`docs/architecture/audits/codex-protocol.md`) — the
# ``chatgpt.com/backend-api/codex`` base URL has no public override and
# LiteLLM has no provider entry for it.
CHATGPT_INTERNAL_URL = "https://chatgpt.com/backend-api/codex/responses"
CHATGPT_INTERNAL_USER_AGENT = "enchanter-agent/0.7 (proxy-chatgpt)"

# Silently drop kwargs the upstream provider doesn't understand.
litellm.drop_params = True


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

    ChatGPT-login routing
    ---------------------
    When the inbound carried a JWT-shaped Bearer (server.py marks it as
    ``kind="chatgpt-jwt"`` on the passthrough-auth metadata blob), we
    bypass LiteLLM entirely and POST to the ChatGPT-internal endpoint
    directly via stdlib HTTP. LiteLLM has no per-request base-URL
    override for the ``chatgpt.com/backend-api/codex/responses`` route.
    """
    auth = _passthrough_auth_dict(req)
    if auth is not None and auth.get("kind") == "chatgpt-jwt":
        return await _call_chatgpt_internal(req, auth)

    kwargs = _build_litellm_kwargs(req, stream=False)
    kwargs.update(_passthrough_auth_kwargs(req))
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

    ChatGPT-login streaming is **not** implemented in this wave — see
    :func:`_stream_chatgpt_internal`.
    """
    auth = _passthrough_auth_dict(req)
    if auth is not None and auth.get("kind") == "chatgpt-jwt":
        async for chunk in _stream_chatgpt_internal(req, auth):
            yield chunk
        return

    kwargs = _build_litellm_kwargs(req, stream=True)
    kwargs.update(_passthrough_auth_kwargs(req))
    try:
        stream = await litellm.acompletion(**kwargs)
    except Exception as exc:  # noqa: BLE001
        raise _wrap_error(req.model, exc) from exc

    async for chunk in _translate_stream(stream, req.model):
        yield chunk


def _passthrough_auth_dict(req: CanonicalRequest) -> dict[str, Any] | None:
    """Return the inbound-auth blob from request metadata, or None."""
    if not req.metadata:
        return None
    auth = req.metadata.get("_enchanter_passthrough_auth")
    if isinstance(auth, dict):
        return auth
    return None


# ---------------------------------------------------------------------------
# ChatGPT-internal (non-LiteLLM) upstream path.
# ---------------------------------------------------------------------------


def _canonical_to_completion_shim(req: CanonicalRequest) -> SimpleNamespace:
    """Build a duck-typed CompletionRequest for `build_responses_request`.

    ``build_responses_request`` only reads ``model``, ``messages``,
    ``system``, ``temperature``, ``max_tokens``, ``stop_sequences``, and
    ``tools`` via ``getattr``; ``messages`` need ``role`` and ``content``
    (string) per :func:`robit.llm._codex_responses._message_to_input_item`.

    We flatten each canonical message's TextParts into a single string and
    drop tool_use / tool_result parts (v1 limitation — Codex ChatGPT-mode
    is text-first; the audit notes tool streaming is deferred).
    """
    flat_messages: list[SimpleNamespace] = []
    for m in req.messages:
        if m.role not in ("user", "assistant"):
            # The Responses API also accepts these via the canonical path,
            # but we keep the shim minimal — only user/assistant survive.
            continue
        text_parts: list[str] = []
        for part in m.content:
            if isinstance(part, TextPart):
                text_parts.append(part.text)
            elif isinstance(part, ToolResultPart):
                # Inline tool results as text for the chatgpt-internal path
                # (v1 — Codex ChatGPT mode does not yet round-trip
                # canonical tool calls through this client).
                text_parts.append(part.content)
            # ToolUsePart from a prior assistant turn: dropped in v1.
        flat = "".join(text_parts)
        flat_messages.append(SimpleNamespace(role=m.role, content=flat))

    return SimpleNamespace(
        model=req.model,
        messages=flat_messages,
        system=req.system,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        stop_sequences=req.stop_sequences,
        tools=None,  # Responses-API tools are dropped in v1 — see docstring.
    )


def _coerce_completion_response_to_canonical(
    completion: Any, *, requested_model: str
) -> CanonicalResponse:
    """Translate :class:`CompletionResponse` → :class:`CanonicalResponse`.

    The Responses-API JSON only carries a single text output in v1 (one
    ``output_text`` part). Tool calls parsed by
    :func:`parse_responses_completion` come back on
    ``completion.tool_calls``; we surface them as :class:`ToolUsePart`s.
    """
    content: list[ContentPart] = []
    if completion.text:
        content.append(TextPart(text=completion.text))
    for tc in completion.tool_calls or []:
        content.append(
            ToolUsePart(
                id=tc.get("id") or "",
                name=tc.get("name") or "",
                input=tc.get("input") or {},
            )
        )

    has_tool_use = bool(completion.tool_calls)
    stop_reason = _map_finish_reason(
        completion.stop_reason, has_tool_use=has_tool_use
    )

    return CanonicalResponse(
        model=completion.model or requested_model,
        content=tuple(content),
        stop_reason=stop_reason,
        usage=CanonicalUsage(
            input_tokens=_safe_int(completion.input_tokens),
            output_tokens=_safe_int(completion.output_tokens),
        ),
    )


def _post_chatgpt_internal_sync(
    body: bytes, token: str, account_id: str | None
) -> dict[str, Any]:
    """Sync POST to the ChatGPT-internal Responses endpoint.

    Raises :class:`urllib.error.HTTPError` on non-2xx so the async caller
    can map status codes (401, 5xx, …) into :class:`UpstreamError`.

    Honesty note: the Authorization header carries the raw JWT. We never
    log the request object, never include the headers dict in any
    exception we re-raise, and the urllib stack does not log bodies by
    default. The only path that could leak the token is operator-supplied
    handlers attached to the root logger — outside this module's scope.
    """
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": CHATGPT_INTERNAL_USER_AGENT,
    }
    if account_id:
        headers["ChatGPT-Account-ID"] = account_id

    request = urllib.request.Request(  # noqa: S310 — fixed https URL
        CHATGPT_INTERNAL_URL,
        data=body,
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as resp:  # noqa: S310
        raw = resp.read()
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


async def _call_chatgpt_internal(
    req: CanonicalRequest,
    auth: dict,
) -> CanonicalResponse:
    """Non-streaming POST to ``chatgpt.com/backend-api/codex/responses``.

    Bypasses LiteLLM. Builds the Responses-API body via
    :func:`robit.llm._codex_responses.build_responses_request`, POSTs
    through stdlib ``urllib`` on a worker thread, and translates the
    response into a :class:`CanonicalResponse`.

    Errors are surfaced as :class:`UpstreamError`. On HTTP 401 we attach
    a hint that the JWT should be refreshed via ``codex login`` — the
    proxy itself does not perform token refresh (the host agent owns the
    credential lifecycle).
    """
    body_dict = build_responses_request(
        _canonical_to_completion_shim(req), stream=False
    )
    body_bytes = json.dumps(body_dict).encode("utf-8")
    token = str(auth.get("value") or "")
    account_id = auth.get("account_id")

    try:
        response_json = await asyncio.to_thread(
            _post_chatgpt_internal_sync, body_bytes, token, account_id
        )
    except urllib.error.HTTPError as exc:
        snippet = _safe_error_snippet(exc)
        if exc.code == 401:
            raise UpstreamError(
                provider="chatgpt",
                status=401,
                message=(
                    "ChatGPT subscription token rejected — refresh the "
                    "cached JWT (`codex login`) and retry."
                ),
            ) from exc
        raise UpstreamError(
            provider="chatgpt",
            status=exc.code,
            message=f"upstream {exc.code}: {snippet}",
        ) from exc
    except urllib.error.URLError as exc:
        raise UpstreamError(
            provider="chatgpt",
            status=None,
            message=f"network error: {exc.reason}",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise UpstreamError(
            provider="chatgpt",
            status=None,
            message=f"unexpected upstream error: {exc}",
        ) from exc

    completion = parse_responses_completion(
        response_json, requested_model=req.model
    )
    return _coerce_completion_response_to_canonical(
        completion, requested_model=req.model
    )


async def _stream_chatgpt_internal(
    req: CanonicalRequest,  # noqa: ARG001 — kept for signature parity
    auth: dict,  # noqa: ARG001
) -> AsyncIterator[CanonicalChunk]:
    """Streaming variant. **Deferred to Wave 18+.**

    Stdlib ``urllib`` does not expose an async chunked-read interface,
    and a worker-thread pump driving an ``asyncio.Queue`` is enough
    additional surface area to deserve its own wave. The honest scope
    cut is to raise ``NotImplementedError`` now and route streaming
    Codex-ChatGPT requests to a clear error rather than silently falling
    back to LiteLLM (which would 404 against ``chatgpt.com``).
    """
    if False:  # pragma: no cover — keep this an async generator
        yield  # type: ignore[unreachable]
    raise NotImplementedError(
        "ChatGPT-login streaming via the proxy is not implemented in "
        "Wave 17.2 — non-streaming requests work. See enchanter/proxy/"
        "upstream.py::_stream_chatgpt_internal."
    )


def _safe_error_snippet(exc: urllib.error.HTTPError) -> str:
    """Read up to 512 chars of an HTTPError body for diagnostics.

    Honesty note: this snippet is included in the surfaced
    :class:`UpstreamError` message. We deliberately *do not* read
    request headers from ``exc`` and we *do not* include the failing URL
    (which is constant — see ``CHATGPT_INTERNAL_URL`` — and therefore
    carries no incremental info). The response body from ChatGPT does
    not echo the inbound auth header.
    """
    try:
        raw = exc.read() if hasattr(exc, "read") else b""
    except Exception:  # noqa: BLE001
        return ""
    try:
        text = raw.decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return ""
    return text[:512]


# ---------------------------------------------------------------------------
# Request building.
# ---------------------------------------------------------------------------


def _passthrough_auth_kwargs(req: CanonicalRequest) -> dict[str, Any]:
    """Translate the server's stashed inbound auth into LiteLLM kwargs.

    Reads ``req.metadata["_enchanter_passthrough_auth"]`` (set by
    :mod:`robit.proxy.server` when ``passthrough_auth=True``) and
    returns kwargs to merge into the LiteLLM call: ``api_key`` for
    API-key-style auth, plus ``extra_headers`` for Anthropic OAuth
    bearer tokens.

    Honesty note: the returned dict carries the raw credential. Never
    log it. We do not include it in any error envelope.

    Anthropic-OAuth quirk: LiteLLM (≤ 1.50.x) does forward
    ``extra_headers`` for the Anthropic provider, but the ``api_key``
    kwarg is still required (else LiteLLM raises before constructing
    the request). We supply a placeholder so the OAuth header wins on
    the wire. # TODO: verify LiteLLM extra_headers acceptance across
    versions before flipping this path on by default.
    """
    if not req.metadata:
        return {}
    auth = req.metadata.get("_enchanter_passthrough_auth")
    if not isinstance(auth, dict):
        return {}
    kind = auth.get("kind")
    value = auth.get("value", "")
    if kind in ("anthropic-api-key", "openai-bearer", "gemini-api-key"):
        return {"api_key": value}
    if kind == "anthropic-oauth":
        return {
            "api_key": "sk-ant-placeholder",
            "extra_headers": {"Authorization": f"Bearer {value}"},
        }
    return {}


def _build_litellm_kwargs(req: CanonicalRequest, *, stream: bool) -> dict[str, Any]:
    """Translate a :class:`CanonicalRequest` into LiteLLM kwargs."""
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
    if req.metadata:
        # LiteLLM accepts a metadata bag for routing/logging hints. Strip
        # the internal pass-through-auth sentinel so the credential never
        # rides on the metadata bag (which LiteLLM may log).
        safe_meta = {
            k: v
            for k, v in req.metadata.items()
            if k != "_enchanter_passthrough_auth"
        }
        if safe_meta:
            kwargs["metadata"] = safe_meta

    return kwargs


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
]
