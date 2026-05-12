"""ToolPoisoningScan engine — port of lich.adapter.ts (M1 static scan + M6 FP tracking).

Fires at `post-response` phase.  Required plugin (fail-closed).

Algorithm:
  1. M1 static scan — examines tool_schema fields (description, parameters,
     errorTemplates, name, displayName) against the 5 SUSPICION_PATTERNS.
  2. M6 simplified — EMA FP-rate per pattern_id; FP rate > 0.5 → 50% effective
     severity downweight.
  3. Aggregate suspicion_score = sum of effective severities.
     score >= VETO_THRESHOLD (3) → veto + tool-poisoning-scan.suspicion.flagged derived events
     0 < score < threshold        → ack degraded=True + tool-poisoning-scan.suspicion.flagged events
     score == 0                   → clean ack
  4. ReplayCache keyed on SHA-256 of the stable-serialised schema.  Cache hit
     returns the cached verdict without re-running M1.

Sandbox confirmation (SandboxConfirmation):
  - Off by default.  Enable via engine.enable_sandbox = True.
  - When on and M1 produces a warn-level ack: sandbox runs a second static pass.
    Sandbox failures are advisory (degraded=True) — they do not override the
    M1 decision.  If M1 already vetoed, sandbox is skipped.

Phase: post-response (matches the TS lichAdapter.phases).
Topics:
  subscribes: mcp.tool.registered, filesystem.write.completed
  emits:      tool-poisoning-scan.suspicion.flagged, tool-poisoning-scan.sandbox.executed, tool-poisoning-scan.rubric.verdict
"""

from __future__ import annotations

import hashlib
import json
import time

from enchanter.core import EnchantedEvent, PluginAck, RequestContext
from enchanter.core.plugin import PluginTopics
from enchanter.core.bus import new_event_id

from .patterns import SUSPICION_PATTERNS, VETO_THRESHOLD, SuspicionPattern
from .replay_cache import ReplayCache, ScanVerdict
from .sandbox import SandboxConfirmation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_ms() -> int:
    return int(time.time() * 1000)


def _stable_json(obj: object) -> str:
    """Deterministically serialise *obj* — dict keys sorted at every level.

    Mirrors the TS stableStringify used for schema digesting.
    """
    return json.dumps(obj, sort_keys=True, default=str)


def _schema_signature(schema: dict[str, object]) -> str:
    """Return a SHA-256 hex digest of the stable-serialised schema."""
    return hashlib.sha256(_stable_json(schema).encode()).hexdigest()


# ---------------------------------------------------------------------------
# M6 simplified: EMA false-positive tracking (per-instance)
# ---------------------------------------------------------------------------

_FP_DECAY = 0.9
_FP_DOWNWEIGHT_THRESHOLD = 0.5


class _PatternFPState:
    """Tracks EMA false-positive rate for a single pattern."""

    __slots__ = ("fp_rate",)

    def __init__(self) -> None:
        self.fp_rate: float = 0.0

    def record_false_positive(self) -> None:
        """EMA update: new_fp = decay * current + (1 - decay) * 1."""
        self.fp_rate = _FP_DECAY * self.fp_rate + (1 - _FP_DECAY)


# ---------------------------------------------------------------------------
# ToolPoisoningScan
# ---------------------------------------------------------------------------

class ToolPoisoningScan:
    """Required at post-response.  Fail-closed veto on tool-poisoning patterns.

    Per-instance state:
      _fp_states     — EMA FP rates keyed by pattern_id
      _replay_cache  — LRU(1000) schema-signature → ScanVerdict
      _sandbox       — SandboxConfirmation instance (static-only v0)
      enable_sandbox — False by default; set True to enable the second pass
    """

    name = "tool-poisoning-scan"
    phases: tuple[str, ...] = ("post-response",)
    required = True
    topics = PluginTopics(
        subscribes=("mcp.tool.registered", "filesystem.write.completed"),
        emits=("tool-poisoning-scan.suspicion.flagged", "tool-poisoning-scan.sandbox.executed", "tool-poisoning-scan.rubric.verdict"),
    )
    budget_tier = "always"  # security plugin — never silenced

    def __init__(
        self,
        *,
        cache_capacity: int = 1_000,
        enable_sandbox: bool = False,
    ) -> None:
        self._fp_states: dict[str, _PatternFPState] = {
            p.id: _PatternFPState() for p in SUSPICION_PATTERNS
        }
        self._replay_cache = ReplayCache(capacity=cache_capacity)
        self._sandbox = SandboxConfirmation()
        self.enable_sandbox = enable_sandbox

    # ------------------------------------------------------------------
    # Public: mark a false positive for M6 downweighting
    # ------------------------------------------------------------------

    def mark_false_positive(self, pattern_id: str) -> None:
        """Feed a false-positive signal for *pattern_id* (M6 EMA update)."""
        state = self._fp_states.get(pattern_id)
        if state is not None:
            state.record_false_positive()

    # ------------------------------------------------------------------
    # PluginAdapter protocol
    # ------------------------------------------------------------------

    async def on_phase(self, event: EnchantedEvent, ctx: RequestContext) -> PluginAck:
        if event.phase != "post-response":
            return PluginAck(status="ack")

        raw_schema = (dict(event.payload) if event.payload else {}).get("tool_schema")
        if raw_schema is None or not isinstance(raw_schema, dict):
            # No tool schema in payload — nothing to scan.
            return PluginAck(status="ack")

        schema: dict[str, object] = raw_schema  # type: ignore[assignment]

        # --- Cache lookup ---
        sig = _schema_signature(schema)
        cached = self._replay_cache.get(sig)
        if cached is not None:
            return self._ack_from_verdict(cached, event, from_cache=True)

        # --- M1 static scan ---
        verdict = self._run_m1_scan(schema, sig)
        self._replay_cache.set(sig, verdict)

        ack = self._ack_from_verdict(verdict, event, from_cache=False)

        # --- Sandbox confirmation (optional, warn-only path) ---
        if self.enable_sandbox and ack.status != "veto":
            ack = self._augment_with_sandbox(ack, schema, event)

        return ack

    # ------------------------------------------------------------------
    # Internal: M1 static scan
    # ------------------------------------------------------------------

    def _effective_severity(self, pattern: SuspicionPattern) -> float:
        state = self._fp_states.get(pattern.id)
        if state is not None and state.fp_rate > _FP_DOWNWEIGHT_THRESHOLD:
            return pattern.severity * 0.5
        return float(pattern.severity)

    def _run_m1_scan(self, schema: dict[str, object], sig: str) -> ScanVerdict:
        """Scan *schema* with M1 patterns; return a ScanVerdict (not yet cached)."""
        corpora = _extract_corpora(schema)

        seen_ids: set[str] = set()
        matched_ids: list[str] = []
        suspicion_score: float = 0.0

        for pattern in SUSPICION_PATTERNS:
            if pattern.id in seen_ids:
                continue
            if any(pattern.match.search(corpus) for corpus in corpora):
                seen_ids.add(pattern.id)
                matched_ids.append(pattern.id)
                suspicion_score += self._effective_severity(pattern)

        if not matched_ids:
            return ScanVerdict(
                status="clean",
                suspicion_score=0.0,
                pattern_ids=(),
                reason=None,
            )

        pattern_list = ",".join(matched_ids)

        if suspicion_score >= VETO_THRESHOLD:
            return ScanVerdict(
                status="veto",
                suspicion_score=suspicion_score,
                pattern_ids=tuple(matched_ids),
                reason=f"lich-tool-poisoning:{pattern_list}",
            )

        return ScanVerdict(
            status="warn",
            suspicion_score=suspicion_score,
            pattern_ids=tuple(matched_ids),
            reason=f"lich-suspicion-below-threshold:score={suspicion_score}",
        )

    # ------------------------------------------------------------------
    # Internal: build PluginAck from verdict
    # ------------------------------------------------------------------

    def _ack_from_verdict(
        self,
        verdict: ScanVerdict,
        event: EnchantedEvent,
        *,
        from_cache: bool,
    ) -> PluginAck:
        if verdict.status == "clean":
            return PluginAck(status="ack")

        # Build tool-poisoning-scan.suspicion.flagged derived events — one per matched pattern.
        derived: list[EnchantedEvent] = []
        for i, pid in enumerate(verdict.pattern_ids):
            pattern = next((p for p in SUSPICION_PATTERNS if p.id == pid), None)
            eff_sev = (
                self._effective_severity(pattern) if pattern is not None
                else 0.0
            )
            derived.append(
                EnchantedEvent(
                    id=f"{event.correlation_id}::lich-flag-{i}",
                    correlation_id=event.correlation_id,
                    session_id=event.session_id,
                    phase=event.phase,
                    topic="tool-poisoning-scan.suspicion.flagged",
                    source=self.name,
                    budget_tier=event.budget_tier,
                    ts=_now_ms(),
                    payload={
                        "pattern_id": pid,
                        "severity": eff_sev,
                        "from_cache": from_cache,
                    },
                )
            )

        if verdict.status == "veto":
            return PluginAck(
                status="veto",
                reason=verdict.reason,
                derived_events=derived,
            )

        # warn
        return PluginAck(
            status="ack",
            degraded=True,
            reason=verdict.reason,
            derived_events=derived,
        )

    # ------------------------------------------------------------------
    # Internal: sandbox augmentation
    # ------------------------------------------------------------------

    def _augment_with_sandbox(
        self,
        base_ack: PluginAck,
        schema: dict[str, object],
        event: EnchantedEvent,
    ) -> PluginAck:
        """Run SandboxConfirmation; merge findings into *base_ack* (advisory)."""
        sb_verdict = self._sandbox.confirm(schema)

        sandbox_event = EnchantedEvent(
            id=f"{event.correlation_id}::tool-poisoning-scan-sandbox",
            correlation_id=event.correlation_id,
            session_id=event.session_id,
            phase=event.phase,
            topic="tool-poisoning-scan.sandbox.executed",
            source=self.name,
            budget_tier=event.budget_tier,
            ts=_now_ms(),
            payload={
                "status": sb_verdict.status,
                "suspicion_score": sb_verdict.suspicion_score,
                "pattern_ids": list(sb_verdict.pattern_ids),
                "detail": sb_verdict.detail,
            },
        )

        derived = list(base_ack.derived_events) + [sandbox_event]

        if sb_verdict.status == "error":
            return PluginAck(
                status=base_ack.status,
                degraded=True,
                reason=(
                    f"{base_ack.reason}; {sb_verdict.detail}"
                    if base_ack.reason
                    else sb_verdict.detail
                ),
                derived_events=derived,
            )

        if sb_verdict.status == "veto":
            # Sandbox says veto but M1 was warn — escalate to veto.
            reason = f"lich-sandbox-veto:{','.join(sb_verdict.pattern_ids)}"
            if base_ack.reason:
                reason = f"{base_ack.reason}; {reason}"
            return PluginAck(
                status="veto",
                reason=reason,
                derived_events=derived,
            )

        # sandbox clean or warn — advisory, keep M1 ack as-is.
        return PluginAck(
            status=base_ack.status,
            degraded=base_ack.degraded,
            reason=base_ack.reason,
            derived_events=derived,
        )


# ---------------------------------------------------------------------------
# Schema text extraction (faithful to TS scanSchema)
# ---------------------------------------------------------------------------

def _extract_corpora(schema: dict[str, object]) -> list[str]:
    """Return all text strings from a tool schema dict for pattern matching.

    Covers: description, parameter descriptions (both conventions),
    errorTemplates, name, displayName.
    """
    texts: list[str] = []

    desc = schema.get("description")
    if isinstance(desc, str):
        texts.append(desc)

    # Parameters — dict convention or inputSchema.properties convention.
    props: dict[str, object] = {}
    params = schema.get("parameters")
    if isinstance(params, dict):
        props = params
    else:
        input_schema = schema.get("inputSchema")
        if isinstance(input_schema, dict):
            maybe_props = input_schema.get("properties")
            if isinstance(maybe_props, dict):
                props = maybe_props

    for _key, val in props.items():
        if isinstance(val, dict):
            param_desc = val.get("description")
            if isinstance(param_desc, str):
                texts.append(param_desc)
        elif isinstance(val, str):
            texts.append(val)

    # Error templates.
    err = schema.get("errorTemplates")
    if err is not None:
        if isinstance(err, str):
            texts.append(err)
        else:
            texts.append(json.dumps(err, default=str))

    # Name fields (hidden-unicode check).
    for name_field in ("name", "displayName"):
        val = schema.get(name_field)
        if isinstance(val, str):
            texts.append(val)

    return texts


# ---------------------------------------------------------------------------
# Module-level singleton (mirrors TS lichAdapter export)
# ---------------------------------------------------------------------------

adapter = ToolPoisoningScan()
