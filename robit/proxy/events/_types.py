"""robit.proxy.events._types — emitter contract types.

A proxy *event emitter* is a small, stateless-across-requests object that
publishes 0..N events to the in-process bus at one or more lifecycle
phases.  The pipeline drives emitters at well-defined points and never
waits on their results — emitters are fire-and-forget; downstream engines
react to the events via their normal subscriptions.

Layering note: this module defines only the types.  The discovery glue
(:func:`robit.proxy.events.load_emitters`) and the built-in emitter
that ports the pre-refactor publishes live in sibling modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from robit.core import InProcessBus, RequestScratchpad, ScratchCompatMapping

from ..canonical import CanonicalRequest, CanonicalResponse


class EmitPhase:
    """String constants — the four canonical points an emitter can hook.

    Why not :class:`enum.Enum`?  Plain strings flow through the bus payload
    and dict keys without an extra ``.value`` everywhere; ``Protocol``-typed
    ``phases`` tuples then read naturally.  These constants compare equal to
    the literal strings, so tests can write either form.
    """

    PRE_DISPATCH = "pre-dispatch"
    """Before the upstream call.  Vetos are still possible via the engines
    that subscribe to events published here (destructive-op-gate, cve-
    pattern-gate)."""

    POST_DISPATCH = "post-dispatch"
    """After upstream call returns, before yielding to the client.  For
    streaming this is the *first* chunk boundary, NOT the end of stream.
    Wave 13.1 cost-ledger uses this to stamp request-start usage."""

    POST_SESSION = "post-session"
    """After the full response is materialised (unary) or after the stream
    iterator has been exhausted (streaming).  This is where secret-mask and
    other post-response engines see their event."""

    CROSS_SESSION = "cross-session"
    """Outside any single request — fired at proxy startup/shutdown.  For
    long-lived emitters (rate-limiters, cost rollups) that need a heartbeat
    independent of request traffic.  Reserved for future use; the pipeline
    does not currently drive this phase."""


@dataclass(frozen=True)
class EmitContext:
    """Everything an emitter needs from the request to do its job.

    The pipeline constructs one of these per-request and re-builds (via
    :func:`dataclasses.replace`) when the response and accumulated text
    become available.  Emitters MUST treat the dataclass as read-only —
    mutation would smear state across emitters in the same chain.

    Attributes
    ----------
    req:
        The canonical request as it stands AFTER conduct injection.  Wave
        13.1 emitters should NOT assume this matches the original wire
        request — that's available via the adapter layer.
    bus:
        The per-request :class:`InProcessBus`.  Emitters publish to this
        bus and the orchestrator-wired plugins handle the events.  Do NOT
        retain a reference past the emit call — the bus is GC-owned and
        scoped to the request.
    correlation_id, session_id:
        Identifiers stamped on every event the emitter publishes.  Sourced
        from :func:`robit.core.create_request_context`.
    response:
        Populated on POST_DISPATCH and POST_SESSION for unary requests.
        ``None`` for pre-dispatch and for streaming (use ``accumulated_text``
        instead).
    accumulated_text:
        The full joined text output.  For unary, this is the response text;
        for streaming, the text observed mid-stream by
        :class:`~robit.proxy.streaming.SecretSanitizingStream`.
        ``None`` before the response/stream has been seen.
    redactions:
        Pattern IDs emitted by :class:`SecretSanitizingStream` mid-stream.
        Empty for unary requests; populated for streaming on POST_SESSION.
        Read by emitters that surface "this stream had a leak" telemetry
        (cost-ledger, trust-scorer).
    pre_dispatch_done:
        Flips True after the PRE_DISPATCH chain has fired.  Lets a single
        emitter that subscribes to multiple phases know which slot it's
        currently in without inspecting ``phase`` everywhere.
    scratchpad:
        The typed per-request scratch surface
        (:class:`robit.core.RequestScratchpad`).  Each emitter gets its OWN
        isolated bucket via ``scratchpad.for_emitter(name)`` — the namespace is
        enforced by structure, so two emitters can never collide on a key.
        Cross-cutting scalars (``budget_tier``, an observed ``veto``) live in
        ``scratchpad.shared``.  New emitters should prefer this over ``scratch``.
    scratch:
        Deprecated dict-shaped compatibility view over ``scratchpad`` (see
        :class:`robit.core.ScratchCompatMapping`).  Emitters MAY still stash
        private state here between their own phases — keys equal to a known
        emitter name route to that emitter's isolated bucket; any other key
        routes to ``scratchpad.shared``.  Retained for one release so existing
        emitters keep working; new code should use ``scratchpad`` directly.
    """

    req: CanonicalRequest
    bus: InProcessBus
    correlation_id: str
    session_id: str
    response: CanonicalResponse | None = None
    accumulated_text: str | None = None
    redactions: tuple[str, ...] = ()
    pre_dispatch_done: bool = False
    scratchpad: RequestScratchpad = field(default_factory=RequestScratchpad)
    # ``scratch`` is the deprecated dict-shaped compatibility view bound to
    # ``scratchpad`` (resolved in ``__post_init__``).  It accepts a plain dict
    # at construction (legacy callers seed cross-cutting keys that way) or a
    # ready :class:`ScratchCompatMapping` (carried verbatim by
    # :func:`dataclasses.replace`).  ``None`` (the default) means "bind a fresh
    # view over ``scratchpad``".
    scratch: ScratchCompatMapping = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        current = self.scratch
        if isinstance(current, ScratchCompatMapping):
            # Carried over by dataclasses.replace — already bound to a
            # scratchpad; re-bind defensively to THIS instance's scratchpad so
            # the view and the typed field never diverge.
            object.__setattr__(self, "scratch", ScratchCompatMapping(self.scratchpad))
            return
        view = ScratchCompatMapping(self.scratchpad)
        if isinstance(current, dict):
            # Legacy seed: cross-cutting keys (budget_tier, veto, ...) and/or
            # per-emitter dicts.  Replay them through the view so routing
            # (namespace bucket vs shared) is applied consistently.
            for key, value in current.items():
                view[key] = value
        object.__setattr__(self, "scratch", view)


class EventEmitter(Protocol):
    """An emitter publishes 0-N bus events at one or more lifecycle phases.

    Implementations MUST be stateless across requests (use ``ctx.scratch`` for
    per-request state).  MUST NOT mutate ``ctx.req`` or ``ctx.response``.
    MAY publish events to ``ctx.bus``.  The pipeline does NOT wait for engine
    ACKs — emitters fire-and-forget; downstream engines pick up via their
    normal subscriptions.

    Discovery contract: a module under ``robit.proxy.events`` is treated
    as an emitter module iff it defines a module-level ``emitter`` attribute
    that satisfies this Protocol.  The discovery order is alphabetical by
    module name; emitters within the chain fire in that order at each phase.

    A single emitter MAY register for multiple phases (see
    :class:`robit.proxy.events.builtin.BuiltinEmitter`).  The pipeline
    calls :meth:`emit` once per (emitter, phase) tuple in the order the
    phases are reached.
    """

    name: str
    phases: tuple[str, ...]

    async def emit(self, phase: str, ctx: EmitContext) -> None: ...


__all__ = [
    "EmitPhase",
    "EmitContext",
    "EventEmitter",
]
