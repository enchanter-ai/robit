"""enchanter.proxy.pipeline — orchestrator wrapper around the upstream call.

This is the single integration surface between the proxy's request/response
shape (:mod:`enchanter.proxy.canonical`) and the agent's 7-phase lifecycle
(:mod:`enchanter.core.lifecycle`).  Every proxy request — streaming or not —
flows through one of two entry points:

* :func:`run`   — non-streaming;  returns a fully materialised response.
* :func:`stream` — streaming;     returns an async iterator of chunks.

Both run the full lifecycle around the upstream LiteLLM call, including:

  1. **Conduct injection.**   Per-skill conduct XML is prepended to the
     system prompt before anything else.  Controlled by
     :attr:`PipelineOptions.conduct`.
  2. **Trust-gate.**          A ``mcp.tool.call.requested`` event is published
     at the ``trust-gate`` phase, carrying a representative ``tool``/``args``
     payload synthesised from the request.  ``destructive-op-gate`` and
     ``cve-pattern-gate`` both subscribe to this topic and scan the payload
     for W5 / CVE patterns; either may veto.  A second ``llm.proxy.request``
     event is also emitted for future general listeners; the engines that
     actually decide the verdict key off ``mcp.tool.call.requested``.
  3. **Dispatch.**            The dispatch phase invokes the upstream — via
     :func:`call_upstream` for ``run`` or :func:`stream_upstream` for
     ``stream`` — wrapped in the orchestrator's ``dispatch`` callback so
     no other code path can talk to LiteLLM.
  4. **Post-response.**       A ``mcp.tool.result.received`` event with the
     full response text is published at the ``post-response`` phase so
     ``secret-mask`` can scan for leaked keys/tokens.  A parallel
     ``llm.proxy.response`` event is also emitted.

A fresh :class:`InProcessBus` and :class:`Orchestrator` are constructed per
request — sessions are isolated by :func:`create_request_context`'s
correlation_id.  Sharing state across requests would let a slow plugin's
ack from request *N* satisfy a different request's wait — that's a hard
no.

Topic choices
-------------

The wave-2 brief asks for ``llm.proxy.request`` / ``llm.proxy.response``
as the canonical topic names.  The actual security engines
(``destructive-op-gate``, ``cve-pattern-gate``, ``secret-mask``) subscribe
to ``mcp.tool.call.requested`` / ``mcp.tool.result.received`` instead.  To
satisfy both the spec's branding and the engines' wiring, the pipeline
publishes BOTH topics at each gate.  Agent E's HTTP layer can hang headers
off either name.

Known Limitations
-----------------

#1  **Streaming secret-mask is post-stream-only.**  ``stream`` ships each
    chunk to the caller immediately and only feeds the accumulator with a
    copy.  After the stream ends, the accumulated text is fed to
    post-response; if secret-mask matches, the proxy emits a
    ``BusObservation`` and Agent E can surface the match in headers — but
    the chunks the caller already received are NOT retroactively redacted.
    A targeted leak that arrives mid-stream still escapes.  Mitigations
    (chunk-level rolling redaction) are deferred to wave 3.

#2  **8 MiB accumulator cap.**  Streams that emit more than 8 MiB of
    text+tool-args stop accumulating after the cap; an ``llm.proxy.
    accumulator-truncated`` bus event is emitted exactly once.  The
    post-response secret scan then runs on whatever fit, so very long
    streams may have unscanned tail content.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, replace
from typing import AsyncIterator, Union

from enchanter.core import (
    InProcessBus,
    Orchestrator,
    OrchestratorConfig,
    SecurityVetoError,
    create_request_context,
)
from enchanter.core.bus import build_event
from enchanter.core.events import EnchantedEvent
from enchanter.loader import load_engine_registry

from .canonical import (
    CanonicalChunk,
    CanonicalRequest,
    CanonicalResponse,
    TextPart,
)
from .conduct import DEFAULT_PROXY_RULES, apply_conduct_to_request
from .events import EmitContext, EmitPhase, EventEmitter, load_emitters
from .streaming import SecretSanitizingStream, StreamAccumulator, tee_stream
from .upstream import call_upstream, stream_upstream


_log = logging.getLogger(__name__)


# Maximum prompt summary length forwarded into trust-gate payloads.  Keeps
# the bus event small (the engines run regex over this string).
_PROMPT_SUMMARY_LIMIT = 1024


# ---------------------------------------------------------------------------
# Public dataclasses.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineOptions:
    """Per-request knobs for :func:`run` and :func:`stream`.

    Attributes
    ----------
    conduct:
        When True (default), conduct XML is prepended to the request's system
        prompt before dispatch.  When False, the request is forwarded as-is.
    conduct_rules:
        Which conduct rules to inject when ``conduct=True``.  ``None`` →
        :data:`DEFAULT_PROXY_RULES`.
    """

    conduct: bool = True
    conduct_rules: frozenset[str] | None = None


@dataclass(frozen=True)
class BusObservation:
    """A tiny summary of a bus event that crossed an enforcement boundary.

    By design the payload summary contains only pattern identifiers, match
    counts, and similar small scalars — never raw content.  Agent E uses
    these to populate response headers (``X-Enchanter-Veto``,
    ``X-Enchanter-Mask-Matched``, ...) without risking content leakage.
    """

    topic: str
    source: str
    payload_summary: dict


@dataclass(frozen=True)
class VetoResult:
    """Returned by :func:`run` / :func:`stream` when a lifecycle gate vetoed."""

    phase: str
    plugin: str
    reason: str
    pattern_id: str | None = None
    pattern_name: str | None = None


@dataclass(frozen=True)
class PipelineResult:
    """Non-veto outcome of :func:`run`."""

    response: CanonicalResponse
    fired: tuple[BusObservation, ...]


# ---------------------------------------------------------------------------
# Internal helpers.
# ---------------------------------------------------------------------------


def _first_user_text(req: CanonicalRequest) -> str:
    """Best-effort first-user-message text snippet for trust-gate payloads."""
    for msg in req.messages:
        if msg.role != "user":
            continue
        parts: list[str] = []
        for part in msg.content:
            if isinstance(part, TextPart):
                parts.append(part.text)
        if parts:
            return "".join(parts)
    return ""


def _prompt_summary(req: CanonicalRequest) -> str:
    """First 1 KiB of joined user text, suitable for the trust-gate payload."""
    text = _first_user_text(req)
    if len(text) > _PROMPT_SUMMARY_LIMIT:
        return text[:_PROMPT_SUMMARY_LIMIT]
    return text


def _trust_gate_payload(req: CanonicalRequest) -> dict:
    """Build a tool/args-shaped payload the trust-gate engines can scan.

    The engines (``destructive-op-gate``, ``cve-pattern-gate``) look for two
    fields:

    * ``tool``  — a string identifying the operation; reconstructed command
      line is ``"<tool> <args.join(' ')>"``.
    * ``args``  — list of strings concatenated into the command line.

    We pick ``llm.completion`` as a stable synthetic tool name and pack the
    model + prompt snippet into ``args``.  The engines also scan a
    JSON-stringified view of the full payload, so any destructive text in
    the prompt is reachable through either corpus view.
    """
    summary = _prompt_summary(req)
    return {
        "tool": "llm.completion",
        "args": [req.model, summary],
        "model": req.model,
        "prompt_summary": summary,
    }


def _post_response_payload(
    *,
    result_text: str,
    input_tokens: int,
    output_tokens: int,
    model: str,
) -> dict:
    return {
        "result": result_text,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "model": model,
    }


def _joined_response_text(resp: CanonicalResponse) -> str:
    """Concatenate every text part in the response for the post-response scan."""
    parts: list[str] = []
    for part in resp.content:
        if isinstance(part, TextPart):
            parts.append(part.text)
    return "".join(parts)


class _BusRecorder:
    """Subscribes to ``*`` and snapshots enforcement-gate activity.

    Captures only events from the proxy's known enforcement engines so the
    observation list stays small (no flood from internal lifecycle topics).
    """

    # Topics worth surfacing to Agent E's response headers.
    _INTERESTING_SUFFIXES = (
        ".veto",
        ".warn",
        ".matched",
        "accumulator-truncated",
    )
    # Sources we treat as enforcement-relevant.
    _INTERESTING_SOURCES = frozenset(
        {
            "destructive-op-gate",
            "cve-pattern-gate",
            "secret-mask",
            "tool-poisoning-scan",
            "trust-scorer",
            "proxy-pipeline",
        }
    )

    def __init__(self) -> None:
        self.observations: list[BusObservation] = []

    def is_interesting(self, event: EnchantedEvent) -> bool:
        if event.source in self._INTERESTING_SOURCES:
            return True
        for suffix in self._INTERESTING_SUFFIXES:
            if event.topic.endswith(suffix):
                return True
        return False

    def record(self, event: EnchantedEvent) -> None:
        if not self.is_interesting(event):
            return
        # Strip raw content from the payload summary — Agent E surfaces this
        # in response headers, so it must be safe to render verbatim.
        payload_summary = self._summarise_payload(dict(event.payload or {}))
        self.observations.append(
            BusObservation(
                topic=event.topic,
                source=event.source,
                payload_summary=payload_summary,
            )
        )

    @staticmethod
    def _summarise_payload(payload: dict) -> dict:
        """Keep only small scalar/list-of-string fields; drop the rest."""
        safe_keys = {
            "pattern_id",
            "pattern_name",
            "matched_patterns",
            "redacted_length",
            "severity",
            "score",
            "reason",
            # Mid-stream redactions captured by SecretSanitizingStream.
            # Listed here so Agent E can surface them in response headers.
            "mid_stream_redactions",
        }
        out: dict = {}
        for key, value in payload.items():
            if key not in safe_keys:
                continue
            if isinstance(value, (int, float, bool, str)):
                out[key] = value
            elif isinstance(value, (list, tuple)) and all(
                isinstance(v, (int, float, bool, str)) for v in value
            ):
                out[key] = list(value)
        return out


async def _run_emitters(
    emitters: tuple[EventEmitter, ...],
    phase: str,
    ctx: EmitContext,
) -> None:
    """Run every emitter that registered for ``phase`` in discovery order.

    Fire-and-forget contract: an emitter that raises is logged and skipped;
    the chain continues.  This protects the pipeline from a buggy Wave 13.1
    plugin taking down a proxy request.
    """
    for em in emitters:
        if phase not in em.phases:
            continue
        try:
            await em.emit(phase, ctx)
        except Exception:
            _log.exception(
                "emitter %r raised during phase %s; continuing", em.name, phase
            )


async def _publish_trust_gate(
    bus: InProcessBus,
    ctx,
    req: CanonicalRequest,
) -> None:
    """Publish the trust-gate events the proxy is responsible for.

    Two events are published, both at ``phase="trust-gate"``:

    * ``mcp.tool.call.requested``  — what the security engines subscribe to.
    * ``llm.proxy.request``        — what wave-2 server-side header logic
      can match on for free, future-proofing the pipeline against a renamed
      engine topic later.
    """
    payload = _trust_gate_payload(req)

    event_a = build_event(
        correlation_id=ctx.correlation_id,
        session_id=ctx.session_id,
        phase="trust-gate",
        topic="mcp.tool.call.requested",
        source="proxy-pipeline",
        budget_tier=ctx.budget_tier,
        payload=payload,
    )
    await bus.publish(event_a.topic, event_a)

    event_b = build_event(
        correlation_id=ctx.correlation_id,
        session_id=ctx.session_id,
        phase="trust-gate",
        topic="llm.proxy.request",
        source="proxy-pipeline",
        budget_tier=ctx.budget_tier,
        payload={
            "model": req.model,
            "prompt_summary": payload["prompt_summary"],
        },
    )
    await bus.publish(event_b.topic, event_b)


async def _publish_post_response(
    bus: InProcessBus,
    ctx,
    *,
    result_text: str,
    input_tokens: int,
    output_tokens: int,
    model: str,
) -> None:
    """Publish the post-response events: secret-mask scan + proxy header hook."""
    payload = _post_response_payload(
        result_text=result_text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model=model,
    )

    event_a = build_event(
        correlation_id=ctx.correlation_id,
        session_id=ctx.session_id,
        phase="post-response",
        topic="mcp.tool.result.received",
        source="proxy-pipeline",
        budget_tier=ctx.budget_tier,
        payload=payload,
    )
    await bus.publish(event_a.topic, event_a)

    event_b = build_event(
        correlation_id=ctx.correlation_id,
        session_id=ctx.session_id,
        phase="post-response",
        topic="llm.proxy.response",
        source="proxy-pipeline",
        budget_tier=ctx.budget_tier,
        payload={
            "result_length": len(result_text),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "model": model,
        },
    )
    await bus.publish(event_b.topic, event_b)


def _veto_from_error(exc: SecurityVetoError) -> VetoResult:
    """Translate a :class:`SecurityVetoError` into a :class:`VetoResult`.

    Tries to extract pattern_id / pattern_name from the conventional
    ``"<plugin>:<pattern_id>"`` reason string used by destructive-op-gate
    and cve-pattern-gate.  Best-effort only; fields default to ``None`` if
    the reason doesn't fit the convention.
    """
    reason = exc.reason or ""
    pattern_id: str | None = None
    if ":" in reason:
        # Reason shape: "<plugin>:<pattern_id>" or "<plugin>:<id> (advisory)".
        _, _, rest = reason.partition(":")
        token = rest.strip().split(" ", 1)[0]
        if token:
            pattern_id = token
    return VetoResult(
        phase=str(exc.phase),
        plugin=exc.plugin,
        reason=reason,
        pattern_id=pattern_id,
        pattern_name=None,
    )


def _build_orchestrator() -> tuple[InProcessBus, Orchestrator]:
    """Fresh bus + orchestrator wired with the full engine registry.

    Failure to load the registry is fatal — every proxy request needs the
    enforcement pipeline, and silently running without it would violate the
    enforcement contract.
    """
    registry = load_engine_registry()
    bus = InProcessBus()
    orch = Orchestrator(OrchestratorConfig(registry=registry, bus=bus))
    return bus, orch


# ---------------------------------------------------------------------------
# Non-streaming entry point.
# ---------------------------------------------------------------------------


async def run(
    req: CanonicalRequest,
    opts: PipelineOptions = PipelineOptions(),
) -> Union[PipelineResult, VetoResult]:
    """Run the proxy pipeline end-to-end and return the response.

    On a security veto the upstream is NEVER called and a :class:`VetoResult`
    is returned synchronously (no exception).  On any other upstream error,
    :class:`~enchanter.proxy.upstream.UpstreamError` propagates — Agent E's
    HTTP layer maps it to a 502.
    """
    # 1. Conduct injection (or pass-through).
    effective_req = (
        apply_conduct_to_request(req, opts.conduct_rules)
        if opts.conduct
        else req
    )

    # 2. Fresh bus + orchestrator, per-request isolation.
    bus, orch = _build_orchestrator()
    recorder = _BusRecorder()

    async def _record(event: EnchantedEvent):
        recorder.record(event)
        return None

    bus.subscribe("*", _record)

    # 3. Build the request context.
    ctx = create_request_context(user_prompt=_prompt_summary(effective_req))

    # 4. Build the emitter chain and the per-request EmitContext.
    emitters = tuple(load_emitters())
    emit_ctx = EmitContext(
        req=effective_req,
        bus=bus,
        correlation_id=ctx.correlation_id,
        session_id=ctx.session_id,
    )
    emit_ctx.scratch["budget_tier"] = ctx.budget_tier

    # 5. Fire PRE_DISPATCH emitters.  The built-in emitter publishes the
    #    trust-gate events here, so destructive-op-gate / cve-pattern-gate
    #    evaluate the prompt BEFORE the orchestrator's lifecycle.trust-gate.
    await _run_emitters(emitters, EmitPhase.PRE_DISPATCH, emit_ctx)
    emit_ctx = replace(emit_ctx, pre_dispatch_done=True)

    # 6. Dispatch closure — the only place call_upstream is reachable.
    captured: dict = {}

    async def dispatch(ctx) -> CanonicalResponse:
        resp = await call_upstream(effective_req)
        captured["resp"] = resp
        # Inject the post-response payload BEFORE the orchestrator advances
        # to the post-response phase so secret-mask sees real content.
        # We do this by firing POST_SESSION emitters here (NOT after orch.run
        # returns) — secret-mask is wired to the lifecycle post-response
        # phase and must observe the result before the orchestrator's phase
        # ack tracker waits for it.
        nonlocal emit_ctx
        emit_ctx = replace(
            emit_ctx,
            response=resp,
            accumulated_text=_joined_response_text(resp),
        )
        await _run_emitters(emitters, EmitPhase.POST_SESSION, emit_ctx)
        return resp

    # 7. Run the 7-phase lifecycle; trap veto, propagate everything else.
    try:
        await orch.run(ctx, dispatch)
    except SecurityVetoError as exc:
        return _veto_from_error(exc)

    resp: CanonicalResponse = captured["resp"]
    return PipelineResult(response=resp, fired=tuple(recorder.observations))


# ---------------------------------------------------------------------------
# Streaming entry point.
# ---------------------------------------------------------------------------


async def stream(
    req: CanonicalRequest,
    opts: PipelineOptions = PipelineOptions(),
) -> Union[AsyncIterator[CanonicalChunk], VetoResult]:
    """Run the proxy pipeline in streaming mode.

    Two-phase shape:

    * **Synchronous gate.**  Builds the request context, publishes the
      trust-gate events, and inspects ack state for a veto.  If any gate
      vetoed, returns a :class:`VetoResult` synchronously (NOT an async
      iterator).
    * **Async generator.**   Otherwise returns an async generator that
      yields :class:`CanonicalChunk` events from the upstream, in order,
      with zero added latency.  Post-stream the accumulated text is fed to
      a ``mcp.tool.result.received`` event so secret-mask can scan; the
      observation (if any) is silently appended to the bus tap — Agent E
      reads it via the bus, not via the iterator.

    Caller pattern::

        result = await stream(req)
        if isinstance(result, VetoResult):
            return http_451(result)
        async for chunk in result:
            send_sse(chunk)
    """
    effective_req = (
        apply_conduct_to_request(req, opts.conduct_rules)
        if opts.conduct
        else req
    )

    bus, orch = _build_orchestrator()
    recorder = _BusRecorder()

    async def _record(event: EnchantedEvent):
        recorder.record(event)
        return None

    bus.subscribe("*", _record)

    ctx = create_request_context(user_prompt=_prompt_summary(effective_req))

    # Build the emitter chain + EmitContext for this request.
    emitters = tuple(load_emitters())
    emit_ctx = EmitContext(
        req=effective_req,
        bus=bus,
        correlation_id=ctx.correlation_id,
        session_id=ctx.session_id,
    )
    emit_ctx.scratch["budget_tier"] = ctx.budget_tier

    # Synchronous trust-gate check — drive enough of the orchestrator to learn
    # whether anyone vetoes BEFORE we open the upstream stream.  We can't run
    # the full lifecycle here because dispatch needs to feed chunks back to
    # the caller; instead we fire PRE_DISPATCH emitters (which publish the
    # trust-gate events) and ask the bus's ack tracker whether the required
    # gates voted veto.
    await _run_emitters(emitters, EmitPhase.PRE_DISPATCH, emit_ctx)
    emit_ctx = replace(emit_ctx, pre_dispatch_done=True)

    # Inspect ack state for each required trust-gate plugin.
    veto = _check_trust_gate_veto(bus, ctx, orch)
    if veto is not None:
        return veto

    return _stream_body(bus, orch, ctx, effective_req, recorder, emitters, emit_ctx)


def _check_trust_gate_veto(
    bus: InProcessBus,
    ctx,
    orch: Orchestrator,
) -> VetoResult | None:
    """Synchronously check the bus ack tracker for a trust-gate veto.

    Required plugins for ``trust-gate`` may have already acked from the
    pre-published events.  If any acked ``veto``, return a :class:`VetoResult`;
    if any acked anything else (including ``ack`` or ``error``), the request
    is cleared to proceed.  Plugins that haven't acked yet — typically
    because they didn't subscribe to ``mcp.tool.call.requested`` — will be
    polled by the orchestrator's own ``lifecycle.trust-gate`` publish inside
    :func:`stream_body`.
    """
    # Access the orchestrator's internal registry view via its _subscribers.
    # We re-derive it instead of poking private state to keep the contract
    # auditable: the orchestrator knows which plugins gate trust.
    required_plugins = tuple(
        p.name
        for p in orch._registry.values()  # type: ignore[attr-defined]
        if "trust-gate" in p.phases and p.required
    )

    for plugin in required_plugins:
        key = bus.acks._key(ctx.correlation_id, "trust-gate", plugin)  # type: ignore[attr-defined]
        ack = bus.acks._acks.get(key)  # type: ignore[attr-defined]
        if ack is not None and ack.status == "veto":
            reason = ack.reason or "veto"
            pattern_id: str | None = None
            if ":" in reason:
                _, _, rest = reason.partition(":")
                token = rest.strip().split(" ", 1)[0]
                if token:
                    pattern_id = token
            return VetoResult(
                phase="trust-gate",
                plugin=plugin,
                reason=reason,
                pattern_id=pattern_id,
                pattern_name=None,
            )
    return None


async def _stream_body(
    bus: InProcessBus,
    orch: Orchestrator,
    ctx,
    req: CanonicalRequest,
    recorder: _BusRecorder,
    emitters: tuple[EventEmitter, ...] = (),
    emit_ctx: EmitContext | None = None,
) -> AsyncIterator[CanonicalChunk]:
    """The async-generator half of :func:`stream`.

    Runs the orchestrator's 7-phase lifecycle, but the dispatch callback
    opens an upstream stream and tees its chunks to (a) the consumer via
    a shared queue / sentinel and (b) the accumulator for post-response.

    Implementation note: because :meth:`Orchestrator.run` returns once the
    dispatch coroutine completes — and dispatch needs to *yield chunks*
    through this generator — we drive the lifecycle inline rather than
    calling ``orch.run``.  This mirrors the sequence ``orch.run`` performs
    but skips the orchestrator's own publishing for phases we've already
    published.
    """
    accumulator = StreamAccumulator()
    sanitizer = SecretSanitizingStream()
    truncation_signalled = False

    # We bypass orch.run() and instead drive the lifecycle phases manually so
    # chunks can be yielded mid-dispatch.  The bus subscriptions wired by
    # Orchestrator.__init__ are still active; publishing each phase's
    # lifecycle event still triggers the plugin handlers.
    from enchanter.core.context import LIFECYCLE_PHASES

    try:
        for phase in LIFECYCLE_PHASES:
            ctx.phase = phase

            # Publish the orchestrator's lifecycle.<phase> event (matches
            # what Orchestrator.run does verbatim).
            phase_event = orch._build_phase_event(ctx, phase)  # type: ignore[attr-defined]
            await bus.publish(phase_event.topic, phase_event)

            subscribers = [
                p
                for p in orch._registry.values()  # type: ignore[attr-defined]
                if phase in p.phases
            ]
            required = tuple(p.name for p in subscribers if p.required)
            advisory = tuple(p.name for p in subscribers if not p.required)
            all_names = required + advisory

            # Wave 13.3 — drive the two-bucket plugin dispatch directly. Prior
            # to Wave 13.3 the bus publish above triggered plugin handlers via
            # ``lifecycle.<phase>`` subscriptions; the orchestrator now owns
            # that dispatch, so we delegate to ``_dispatch_phase``. The
            # ``dispatch`` phase is special-cased below (we tee the upstream
            # stream there), so skip the direct dispatch for that phase.
            if phase != "dispatch":
                await orch._dispatch_phase(phase, phase_event, subscribers)  # type: ignore[attr-defined]

            if phase == "dispatch":
                # Wrap the upstream with the SecretSanitizingStream (mid-
                # stream redactor) BEFORE the tee.  The sanitizer holds a
                # rolling K-byte window per content-block index, flushes
                # safe prefixes after a regex sweep, and surfaces matched
                # pattern IDs in ``sanitizer.redactions`` once exhausted.
                # The tee_stream then feeds the SANITISED chunks to the
                # accumulator (so post-response scans see redacted text)
                # and yields them to the client.
                src = stream_upstream(req)
                sanitised = sanitizer.wrap(src)
                async for chunk in tee_stream(sanitised, accumulator):
                    # Emit a one-shot truncation bus event the first time
                    # the cap is hit.
                    if accumulator.truncated and not truncation_signalled:
                        truncation_signalled = True
                        trunc_event = build_event(
                            correlation_id=ctx.correlation_id,
                            session_id=ctx.session_id,
                            phase="dispatch",
                            topic="llm.proxy.accumulator-truncated",
                            source="proxy-pipeline",
                            budget_tier=ctx.budget_tier,
                            payload={},
                        )
                        await bus.publish(trunc_event.topic, trunc_event)
                    yield chunk

                # Once the stream finishes we know the full accumulated text
                # and can fire POST_SESSION emitters so secret-mask scans
                # before the orchestrator hits the post-response phase.
                if emit_ctx is not None:
                    emit_ctx = replace(
                        emit_ctx,
                        accumulated_text=accumulator.text,
                        redactions=tuple(sanitizer.redactions),
                    )
                    await _run_emitters(emitters, EmitPhase.POST_SESSION, emit_ctx)
                else:
                    # Backwards-compat path for older test callers.
                    await _publish_post_response(
                        bus,
                        ctx,
                        result_text=accumulator.text,
                        input_tokens=0,
                        output_tokens=0,
                        model=req.model,
                    )
                continue

            if not all_names:
                continue

            # Wait for acks on the current phase.  Required plugins must ack.
            acks = await bus.acks.wait_for_acks(
                ctx.correlation_id,
                phase,
                all_names,
                orch._timeouts[phase],  # type: ignore[attr-defined]
            )

            missing_required = tuple(p for p in required if p not in acks)
            if missing_required:
                # Phase timeout from a required plugin is fatal; surface as
                # a synthetic veto so the caller's consumer (Agent E) can
                # close the stream cleanly.  Wave 3 may differentiate.
                return

            # Veto check.
            for p in required:
                a = acks.get(p)
                if a is not None and a.status == "veto":
                    # Mid-stream veto is not possible at trust-gate because
                    # we've already cleared it; could only happen for a
                    # required plugin on a later phase.  Close the stream.
                    return
    finally:
        # Nothing to clean up — the bus and orchestrator are GC-owned.
        pass


def get_observations(result_or_iter: object) -> tuple[BusObservation, ...]:
    """Convenience accessor for tests and Agent E.

    The streaming entry point returns an async iterator; observations are
    collected on the bus recorder and surfaced through a side channel.
    For non-streaming :class:`PipelineResult` this is just ``.fired``.
    """
    if isinstance(result_or_iter, PipelineResult):
        return result_or_iter.fired
    return ()


__all__ = [
    "BusObservation",
    "PipelineOptions",
    "PipelineResult",
    "VetoResult",
    "run",
    "stream",
    "get_observations",
]
