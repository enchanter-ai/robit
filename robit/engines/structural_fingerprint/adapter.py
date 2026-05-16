"""StructuralFingerprint engine — Python port of naga.adapter.ts.

Implements triple-axis structural fingerprinting (N1 shape hash, N2 TF token
signature, N3 naming-convention) for schema-drift detection at trust-gate.

Phase:      trust-gate
Required:   True  — fail-closed on N1/N3 structural drift;
                    N2-only drift → degraded ack (fail-open)
Topics sub: mcp.tools.list.received
Topics emit:
  structural-fingerprint.pattern.fingerprinted   (first registration of a tool)
  structural-fingerprint.schema.drift.detected   (drift on a known tool)

Drift logic (faithful to TS):
  N1 drift: shape hash mismatch          → structural → veto
  N2 drift: Jaccard(stored, current) < 0.6 (JACCARD_THRESHOLD)
            → non-structural → degraded ack
  N3 drift: naming-convention mismatch   → structural → veto

Any N1 or N3 drift (or both) → veto with reason list.
N2-only drift → degraded ack, no veto.
No drift → clean ack.

N2 token set: top-20 terms by raw TF (no IDF, single-document),
              lowercased, stop-words dropped, min length > 1.
              Matches computeN2() in the TS adapter exactly.

The StructuralFingerprintStore is also available for corpus-level cosine
similarity queries via the adapter's .store property (multi-algorithm path).
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Literal

from robit.core import EnchantedEvent, PluginAck, RequestContext
from robit.core.plugin import PluginTopics
from robit.core.bus import new_event_id

from .tfidf import tokenize
from .store import StructuralFingerprintStore


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

JACCARD_THRESHOLD = 0.6   # [author judgment — mirrors TS]
N2_TOP_N = 20             # top-N terms by TF — mirrors TS computeN2()


# ---------------------------------------------------------------------------
# Naming convention detection — port of TS detectConvention()
# ---------------------------------------------------------------------------

import re as _re

_CAMEL_RE  = _re.compile(r"^[a-z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9]*$")
_SNAKE_RE  = _re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)+$")
_PASCAL_RE = _re.compile(r"^[A-Z][a-zA-Z0-9]+$")
_KEBAB_RE  = _re.compile(r"^[a-z][a-z0-9]*(-[a-z0-9]+)+$")

NamingConvention = Literal["camel", "snake", "pascal", "kebab", "mixed"]


def _detect_convention(name: str) -> NamingConvention:
    if _CAMEL_RE.match(name):  return "camel"
    if _SNAKE_RE.match(name):  return "snake"
    if _PASCAL_RE.match(name): return "pascal"
    if _KEBAB_RE.match(name):  return "kebab"
    return "mixed"


def _compute_n3(properties: dict[str, object]) -> NamingConvention:
    """Majority naming convention across parameter names."""
    names = list(properties.keys())
    if not names:
        return "mixed"

    counts: dict[NamingConvention, int] = {
        "camel": 0, "snake": 0, "pascal": 0, "kebab": 0, "mixed": 0,
    }
    for n in names:
        counts[_detect_convention(n)] += 1

    best: NamingConvention = "mixed"
    best_count = 0
    for conv, cnt in counts.items():
        if cnt > best_count:
            best_count = cnt
            best = conv  # type: ignore[assignment]
    return best


# ---------------------------------------------------------------------------
# N1 — shape hash (simplified: SHA-1 of param_count, param_types_sorted, has_output)
# ---------------------------------------------------------------------------

def _compute_n1(tool: dict[str, object]) -> str:
    input_schema = tool.get("inputSchema") or {}
    if not isinstance(input_schema, dict):
        input_schema = {}

    props: dict[str, object] = input_schema.get("properties") or {}  # type: ignore[assignment]
    if not isinstance(props, dict):
        props = {}

    param_count = len(props)
    param_types_sorted = sorted(
        (p.get("type", "unknown") if isinstance(p, dict) else "unknown")
        for p in props.values()
    )
    has_output_schema = "outputSchema" in tool and tool["outputSchema"] is not None

    import json
    repr_str = json.dumps(
        {"paramCount": param_count, "paramTypesSorted": param_types_sorted,
         "hasOutputSchema": has_output_schema},
        separators=(",", ":"),
    )
    return hashlib.sha1(repr_str.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# N2 — TF token signature (top-N by raw TF, stop-words dropped)
# ---------------------------------------------------------------------------

def _compute_n2(description: str) -> list[str]:
    """Top-N tokens by raw term frequency — mirrors TS computeN2()."""
    tokens = tokenize(description)
    tf: dict[str, int] = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1

    return [
        term
        for term, _ in sorted(tf.items(), key=lambda kv: kv[1], reverse=True)[:N2_TOP_N]
    ]


# ---------------------------------------------------------------------------
# Jaccard similarity (N2 drift detection)
# ---------------------------------------------------------------------------

def _jaccard(a: list[str], b: list[str]) -> float:
    """Jaccard similarity between two token lists treated as sets."""
    if not a and not b:
        return 1.0
    set_a, set_b = set(a), set(b)
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union else 1.0


# ---------------------------------------------------------------------------
# Fingerprint types
# ---------------------------------------------------------------------------

@dataclass
class TripleAxisFingerprint:
    n1: str                  # shape hash (16-char hex)
    n2: list[str]            # top-N token list
    n3: NamingConvention     # majority naming convention


@dataclass
class FingerprintEntry:
    qualified_name: str
    fingerprint: TripleAxisFingerprint


# ---------------------------------------------------------------------------
# Per-instance fingerprint store
# ---------------------------------------------------------------------------

@dataclass
class _FingerprintRegistry:
    """Keyed by qualified_name (server_id.tool_name)."""
    _entries: dict[str, FingerprintEntry] = field(default_factory=dict, init=False)

    def get(self, key: str) -> FingerprintEntry | None:
        return self._entries.get(key)

    def set(self, key: str, entry: FingerprintEntry) -> None:
        self._entries[key] = entry

    def reset(self) -> None:
        self._entries.clear()


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------

DriftKind = Literal["n1", "n2", "n3"]


@dataclass
class DriftResult:
    has_drift: bool
    axes: list[DriftKind]
    structural: bool   # True → veto; False → degraded only


def _detect_drift(
    stored: TripleAxisFingerprint,
    current: TripleAxisFingerprint,
) -> DriftResult:
    axes: list[DriftKind] = []

    if stored.n1 != current.n1:
        axes.append("n1")

    jaccard = _jaccard(stored.n2, current.n2)
    if jaccard < JACCARD_THRESHOLD:
        axes.append("n2")

    if stored.n3 != current.n3:
        axes.append("n3")

    structural = "n1" in axes or "n3" in axes
    return DriftResult(has_drift=bool(axes), axes=axes, structural=structural)


def _now_ms() -> int:
    return int(time.time() * 1000)


# ---------------------------------------------------------------------------
# StructuralFingerprint engine
# ---------------------------------------------------------------------------

class StructuralFingerprint:
    """Triple-axis fingerprint engine — port of naga.adapter.ts.

    Instance-isolated: each StructuralFingerprint() holds its own
    _FingerprintRegistry (N1/N3 drift) and StructuralFingerprintStore
    (N2 TF-IDF corpus similarity, multi-algorithm path).
    """

    name = "structural-fingerprint"
    phases = ("trust-gate",)
    required = True   # fail-closed on N1/N3 structural drift
    topics = PluginTopics(
        subscribes=("mcp.tools.list.received",),
        emits=(
            "structural-fingerprint.pattern.fingerprinted",
            "structural-fingerprint.schema.drift.detected",
        ),
    )
    budget_tier = "always"

    def __init__(self) -> None:
        self._registry = _FingerprintRegistry()
        # Corpus-level TF-IDF store (multi-algorithm; exposed via .store)
        self._store = StructuralFingerprintStore()

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def store(self) -> StructuralFingerprintStore:
        """TF-IDF corpus store — supports cosine similarity queries."""
        return self._store

    def reset(self) -> None:
        """Clear all per-instance state — test teardown helper."""
        self._registry.reset()
        self._store.reset()

    # ------------------------------------------------------------------
    # PluginAdapter protocol
    # ------------------------------------------------------------------

    async def on_phase(self, event: EnchantedEvent, ctx: RequestContext) -> PluginAck:
        if event.phase != "trust-gate":
            return PluginAck(status="ack")
        if event.topic != "mcp.tools.list.received":
            return PluginAck(status="ack")

        payload = event.payload or {}
        server_id: str = payload.get("server_id") or "unknown"  # type: ignore[assignment]
        tools = payload.get("tools")
        if not isinstance(tools, list):
            return PluginAck(status="ack")

        drift_events: list[EnchantedEvent] = []
        fingerprinted_events: list[EnchantedEvent] = []
        should_veto = False
        veto_reasons: list[str] = []
        ts_now = _now_ms()

        for tool in tools:
            if not isinstance(tool, dict):
                continue
            tool_name = tool.get("name")
            if not isinstance(tool_name, str):
                continue

            qualified = f"{server_id}.{tool_name}"
            description: str = tool.get("description") or ""
            input_schema: dict[str, object] = tool.get("inputSchema") or {}  # type: ignore[assignment]
            if not isinstance(input_schema, dict):
                input_schema = {}
            props: dict[str, object] = input_schema.get("properties") or {}  # type: ignore[assignment]
            if not isinstance(props, dict):
                props = {}

            current = TripleAxisFingerprint(
                n1=_compute_n1(tool),
                n2=_compute_n2(description),
                n3=_compute_n3(props),
            )

            existing = self._registry.get(qualified)
            if existing is None:
                # First registration.
                self._registry.set(qualified, FingerprintEntry(
                    qualified_name=qualified,
                    fingerprint=current,
                ))
                # Feed description into the TF-IDF corpus store.
                self._store.add_document(qualified, description)
                fingerprinted_events.append(EnchantedEvent(
                    id=new_event_id(),
                    correlation_id=event.correlation_id,
                    session_id=event.session_id,
                    phase=event.phase,
                    topic="structural-fingerprint.pattern.fingerprinted",
                    source=self.name,
                    budget_tier=event.budget_tier,
                    ts=ts_now,
                    payload={
                        "qualified_name": qualified,
                        "n1": current.n1,
                        "n2": current.n2,
                        "n3": current.n3,
                    },
                ))
                continue

            # Known tool — check for drift.
            drift = _detect_drift(existing.fingerprint, current)
            if not drift.has_drift:
                continue

            jaccard_score = _jaccard(existing.fingerprint.n2, current.n2)
            drift_events.append(EnchantedEvent(
                id=new_event_id(),
                correlation_id=event.correlation_id,
                session_id=event.session_id,
                phase=event.phase,
                topic="structural-fingerprint.schema.drift.detected",
                source=self.name,
                budget_tier=event.budget_tier,
                ts=ts_now,
                payload={
                    "qualified_name": qualified,
                    "axes": drift.axes,
                    "structural": drift.structural,
                    "n1_match": "n1" not in drift.axes,
                    "n3_match": "n3" not in drift.axes,
                    "jaccard": jaccard_score,
                },
            ))

            if drift.structural:
                should_veto = True
                veto_reasons.append(
                    f"structural-fingerprint-drift-veto:{qualified} axes=[{','.join(drift.axes)}]"
                )

        all_derived = fingerprinted_events + drift_events

        if should_veto:
            return PluginAck(
                status="veto",
                reason="; ".join(veto_reasons),
                derived_events=all_derived,
            )

        if drift_events:
            # N2-only drift: degraded ack, not veto.
            return PluginAck(
                status="ack",
                degraded=True,
                reason=f"structural-fingerprint-drift-n2: {len(drift_events)} tool(s) show token-set drift",
                derived_events=all_derived,
            )

        return PluginAck(status="ack", derived_events=all_derived)


adapter = StructuralFingerprint()
