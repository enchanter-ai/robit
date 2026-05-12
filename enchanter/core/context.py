"""Request context — port of `src/orchestration/request-context.ts`.

Every request carries a correlation_id stamped at orchestrator entry;
every bus emission propagates it. Hybrid orchestrator + bus (ADR-001).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Literal, TypedDict


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
