"""tool-poisoning-scan — port of lich.adapter.ts (M1 static scan + M6 FP tracking).

Required plugin at `post-response`. Scans tool_schema fields (description,
parameter descriptions, errorTemplates, name, displayName) against 5 suspicion
patterns (P1–P5). On a veto-threshold breach: returns veto with
lich.suspicion.flagged derived events; the orchestrator surfaces this as
SecurityVetoError. Below threshold: ack with degraded=True + flagged events.
Clean: ack.

ReplayCache (LRU-1000) skips the scan for repeated identical schemas.
SandboxConfirmation (optional, v0 static-only — no subprocess) runs a second
pass on warn-level schemas when enable_sandbox=True on the engine instance.
"""

from .adapter import ToolPoisoningScan, adapter
from .patterns import SUSPICION_PATTERNS, VETO_THRESHOLD, SuspicionPattern
from .replay_cache import ReplayCache, ScanVerdict
from .sandbox import SandboxConfirmation, SandboxVerdict

__all__ = [
    "SUSPICION_PATTERNS",
    "VETO_THRESHOLD",
    "SuspicionPattern",
    "ScanVerdict",
    "SandboxVerdict",
    "ReplayCache",
    "SandboxConfirmation",
    "ToolPoisoningScan",
    "adapter",
]
