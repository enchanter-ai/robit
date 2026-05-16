"""cve-pattern-gate — port of hydra.adapter.ts CVE pattern trust-gate layer.

Required plugin at `trust-gate`. Scans the tool call payload against the
CVE_PATTERNS table (5 patterns across critical/high tiers). On a critical
match: returns veto with a derived cve-pattern-gate.veto event; the
orchestrator surfaces this as SecurityVetoError and short-circuits dispatch.
On a high/medium match: returns ack with degraded=True and a
cve-pattern-gate.warn derived event.
"""

from .adapter import CvePatternGate, adapter
from .patterns import CVE_PATTERNS, CvePattern

__all__ = [
    "CVE_PATTERNS",
    "CvePattern",
    "CvePatternGate",
    "adapter",
]
