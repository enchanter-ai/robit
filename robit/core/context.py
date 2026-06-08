"""Request context — port of `src/orchestration/request-context.ts`.

Every request carries a correlation_id stamped at orchestrator entry;
every bus emission propagates it. Hybrid orchestrator + bus (ADR-001).
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator, MutableMapping
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict


LifecyclePhase = Literal[
    "anchor",
    "trust-gate",
    "pre-dispatch",
    "dispatch",
    "post-response",
    "post-session",
    "cross-session",
]

BudgetTier = Literal["HIGH", "MED", "LOW", "CRITICAL"]

LIFECYCLE_PHASES: tuple[LifecyclePhase, ...] = (
    "anchor",
    "trust-gate",
    "pre-dispatch",
    "dispatch",
    "post-response",
    "post-session",
    "cross-session",
)


@dataclass
class DegradedFinding:
    plugin: str
    reason: str


# ---------------------------------------------------------------------------
# Per-request scratch surface (bus-contract-hardening, audit §5 Q4).
#
# Historically there were three overlapping per-request scratch buckets:
#   * ``EmitContext.scratch: dict[str, Any]``     — namespace-by-convention,
#     emitters were *asked* to key by their own name to avoid collisions, but
#     nothing enforced it, so two emitters could clobber the same key.
#   * ``RequestContext.degraded_findings``        — the degraded-finding list.
#   * ``BusObservation.payload_summary``          — recorder-side, untouched here.
#
# ``RequestScratchpad`` consolidates the first two into one typed structure:
# every emitter gets its *own* sub-dict, created up front, so the namespace is
# enforced by STRUCTURE (you cannot reach into another emitter's bucket by
# accident) rather than by docstring.  ``EmitContext.scratch`` is kept as a
# thin compatibility view over this structure so existing emitters that still
# write ``ctx.scratch[...]`` keep working with no regression.
# ---------------------------------------------------------------------------


@dataclass
class EmitterScratch:
    """Typed per-emitter scratch bucket.

    Every known emitter gets one of these up front.  Cross-cutting fields that
    used to be smuggled through magic dict keys get a real attribute here:

    * ``cents`` — realised request cost in integer US cents, written by the
      cost-ledger emitter so sibling emitters (rate-limiter, trust-scorer) can
      read it without re-deriving.  This replaces the old ``score``-key abuse
      where ``cents`` rode under a ``payload["score"]`` whitelist slot.
    * ``model`` — the upstream model id the cost was computed against.

    ``data`` keeps a free-form dict so emitters with bespoke, non-shared state
    (e.g. inference-substrate's ``briefing`` / ``last_artifact``) keep a home
    without each one needing a typed field.
    """

    cents: int | None = None
    model: str | None = None
    data: dict[str, Any] = field(default_factory=dict)


class _EmitterScratchView(MutableMapping):
    """Dict-shaped view over a single :class:`EmitterScratch`.

    Lets legacy ``ctx.scratch["cost-ledger"]["cents"] = n`` keep working: the
    typed attributes (``cents``, ``model``) are surfaced as mapping keys, and
    any other key falls through to the free-form ``data`` dict.
    """

    _TYPED = ("cents", "model")

    def __init__(self, scratch: "EmitterScratch") -> None:
        self._s = scratch

    def __getitem__(self, key: str) -> Any:
        if key in self._TYPED:
            val = getattr(self._s, key)
            if val is None:
                raise KeyError(key)
            return val
        return self._s.data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if key in self._TYPED:
            setattr(self._s, key, value)
        else:
            self._s.data[key] = value

    def __delitem__(self, key: str) -> None:
        if key in self._TYPED:
            setattr(self._s, key, None)
        else:
            del self._s.data[key]

    def __iter__(self) -> Iterator[str]:
        for key in self._TYPED:
            if getattr(self._s, key) is not None:
                yield key
        yield from self._s.data

    def __len__(self) -> int:
        n = sum(1 for k in self._TYPED if getattr(self._s, k) is not None)
        return n + len(self._s.data)


@dataclass
class RequestScratchpad:
    """One typed scratch surface per request.

    ``per_emitter`` maps an emitter name to its own :class:`EmitterScratch`
    bucket.  Buckets are created lazily-but-isolated via :meth:`for_emitter`,
    so two emitters can never collide on a key — they simply do not share a
    dict.  ``shared`` holds genuinely cross-cutting scalars that are not owned
    by any single emitter (e.g. ``budget_tier``, an observed ``veto``).
    ``findings`` folds in the former ``degraded_findings`` list.
    """

    per_emitter: dict[str, EmitterScratch] = field(default_factory=dict)
    shared: dict[str, Any] = field(default_factory=dict)
    findings: list[DegradedFinding] = field(default_factory=list)

    def for_emitter(self, name: str) -> EmitterScratch:
        """Return (creating if needed) the isolated bucket for *name*."""
        bucket = self.per_emitter.get(name)
        if bucket is None:
            bucket = EmitterScratch()
            self.per_emitter[name] = bucket
        return bucket


class ScratchCompatMapping(MutableMapping):
    """Backwards-compatible ``dict``-shaped view over a :class:`RequestScratchpad`.

    Deprecated shim: new code should reach the scratchpad directly via
    ``EmitContext.scratchpad`` (``scratchpad.for_emitter("name")`` for an
    isolated bucket, ``scratchpad.shared`` for cross-cutting scalars).  This
    view exists only so the existing emitters that still index ``ctx.scratch``
    keep working with no regression:

    * Known emitter names (see :data:`_EMITTER_NAMESPACES`) route to that
      emitter's isolated :class:`EmitterScratch` bucket — so the historical
      ``ctx.scratch.setdefault("cost-ledger", {})["cents"] = n`` lands in the
      structurally-isolated bucket, not a shared free-for-all dict.
    * Every other key routes to ``scratchpad.shared`` (``budget_tier``,
      ``veto``, ...).
    """

    def __init__(self, scratchpad: "RequestScratchpad") -> None:
        self._pad = scratchpad

    def _is_namespace(self, key: str) -> bool:
        return key in _EMITTER_NAMESPACES or key in self._pad.per_emitter

    def __getitem__(self, key: str) -> Any:
        if self._is_namespace(key):
            return _EmitterScratchView(self._pad.for_emitter(key))
        return self._pad.shared[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if self._is_namespace(key):
            bucket = self._pad.for_emitter(key)
            view = _EmitterScratchView(bucket)
            if isinstance(value, dict):
                # Replace semantics: clear then copy so re-assigning the bucket
                # dict (rare, but legal in the old API) behaves like a dict.
                bucket.cents = None
                bucket.model = None
                bucket.data.clear()
                for k, v in value.items():
                    view[k] = v
            else:  # pragma: no cover - defensive; old API only ever set dicts
                raise TypeError(
                    f"emitter-namespace scratch key {key!r} must map to a dict"
                )
        else:
            self._pad.shared[key] = value

    def __delitem__(self, key: str) -> None:
        if self._is_namespace(key):
            self._pad.per_emitter.pop(key, None)
        else:
            del self._pad.shared[key]

    def setdefault(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        if self._is_namespace(key):
            # Always return the isolated, dict-shaped view; ignore *default*
            # (it was historically an empty dict literal).
            return _EmitterScratchView(self._pad.for_emitter(key))
        return self._pad.shared.setdefault(key, default)

    def __contains__(self, key: object) -> bool:
        if isinstance(key, str) and self._is_namespace(key):
            return key in self._pad.per_emitter
        return key in self._pad.shared

    def __iter__(self) -> Iterator[str]:
        yield from self._pad.per_emitter
        yield from self._pad.shared

    def __len__(self) -> int:
        return len(self._pad.per_emitter) + len(self._pad.shared)


# Emitter names whose scratch lives in an isolated per-emitter bucket.  Sourced
# from the emitters' declared ``name`` (proxy/events/*).  Listed explicitly so a
# typo'd key (e.g. ``cost_ledger`` vs ``cost-ledger``) routes to ``shared``
# rather than silently minting a fake bucket.  New emitters add their name here.
_EMITTER_NAMESPACES: frozenset[str] = frozenset(
    {
        "cost-ledger",
        "inference-substrate",
        "trust-scorer",
        "rate-limiter",
        "tool-poisoning-scan",
        "builtin",
    }
)


@dataclass
class RequestContext:
    correlation_id: str
    session_id: str
    phase: LifecyclePhase
    budget_tier: BudgetTier
    sampling_depth: int
    deadline_ms: int
    started_ms: int
    degraded_findings: list[DegradedFinding] = field(default_factory=list)
    user_prompt: str | None = None
    mcp_server_id: str | None = None
    tool_call_id: str | None = None
    # Operator's additive prompt overlay for agent-shaped engines (audit §8 /
    # F3). Threaded from PipelineOptions.prompt_overlay through the orchestrator
    # into every plugin's on_phase ctx — WITHOUT changing on_phase's signature.
    # Precedence: framework < engine-author prompt < operator overlay. ``None``
    # (the default) means no overlay; deterministic engines ignore it.
    prompt_overlay: str | None = None


def _now_ms() -> int:
    return int(time.time() * 1000)


def create_request_context(
    *,
    session_id: str | None = None,
    budget_tier: BudgetTier = "HIGH",
    user_prompt: str | None = None,
    mcp_server_id: str | None = None,
    tool_call_id: str | None = None,
    deadline_ms: int = 30_000,
    prompt_overlay: str | None = None,
) -> RequestContext:
    return RequestContext(
        correlation_id=str(uuid.uuid4()),
        session_id=session_id or str(uuid.uuid4()),
        phase="anchor",
        budget_tier=budget_tier,
        sampling_depth=0,
        deadline_ms=deadline_ms,
        started_ms=_now_ms(),
        user_prompt=user_prompt,
        mcp_server_id=mcp_server_id,
        tool_call_id=tool_call_id,
        prompt_overlay=prompt_overlay,
    )


class PhaseTimeoutMap(TypedDict):
    anchor: int
    trust_gate: int
    pre_dispatch: int
    dispatch: int
    post_response: int
    post_session: int
    cross_session: int


# Phase-name keys use hyphens in the TS source. Python dict keys can't easily
# carry hyphens for TypedDict, so we mirror the TS shape via a regular dict
# keyed by the canonical hyphenated phase names. Consumers index by phase string.
DEFAULT_PHASE_TIMEOUTS_MS: dict[LifecyclePhase, int] = {
    "anchor": 200,
    "trust-gate": 500,
    "pre-dispatch": 200,
    "dispatch": 10_000,
    "post-response": 1_000,
    "post-session": 300,
    "cross-session": 500,
}
