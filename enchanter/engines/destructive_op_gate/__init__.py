"""destructive-op-gate — port of sylph W5 trust-gate fail-closed guard.

Required plugin at `trust-gate`. Scans the tool call payload against a
pattern table of irrecoverable / dangerous operations (force-push, reset
--hard, branch -D, rm -rf). On match: returns veto with a derived event;
the orchestrator surfaces this as SecurityVetoError and short-circuits
dispatch.
"""

from .adapter import DestructiveOpGate, adapter
from .patterns import DESTRUCTIVE_OP_PATTERNS, DestructiveOpPattern

__all__ = [
    "DESTRUCTIVE_OP_PATTERNS",
    "DestructiveOpGate",
    "DestructiveOpPattern",
    "adapter",
]
