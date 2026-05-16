"""ReplayCache — LRU cache for tool-schema scan verdicts.

Port of lich/replay-cache.ts ReplayCache.  Maps a tool signature string
(SHA-256 hex of the stable-serialised schema) to a cached ScanVerdict so
that subsequent calls with identical tool schemas skip the full M1 scan.

Capacity: 1000 entries (matches the task spec; the TS default was 256 — we
match the spec's explicit instruction for the Python port).  Eviction policy:
LRU via OrderedDict, identical to the TS Map-based pattern.

Thread-safety: single-threaded asyncio; no locking needed.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Literal


# ---------------------------------------------------------------------------
# Verdict type
# ---------------------------------------------------------------------------

VerdictStatus = Literal["clean", "warn", "veto"]


@dataclass(frozen=True)
class ScanVerdict:
    """Cached result of a tool-schema scan.

    status: 'clean' | 'warn' | 'veto'
    suspicion_score: sum of effective severities of matched patterns
    pattern_ids: patterns that fired (empty for clean)
    reason: reason string for warn/veto; None for clean
    """

    status: VerdictStatus
    suspicion_score: float
    pattern_ids: tuple[str, ...]
    reason: str | None


# ---------------------------------------------------------------------------
# ReplayCache
# ---------------------------------------------------------------------------

_DEFAULT_CAPACITY = 1_000


class ReplayCache:
    """LRU cache: signature → ScanVerdict.

    get(signature) → ScanVerdict | None   (hit promotes to MRU)
    set(signature, verdict)               (evicts LRU when over capacity)
    size() → int
    clear()
    """

    def __init__(self, capacity: int = _DEFAULT_CAPACITY) -> None:
        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("ReplayCache capacity must be a positive integer")
        self._capacity = capacity
        # OrderedDict preserves insertion order; we re-insert on get() to
        # maintain MRU at the end, LRU at the front — same pattern as the TS.
        self._entries: OrderedDict[str, ScanVerdict] = OrderedDict()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get(self, signature: str) -> ScanVerdict | None:
        """Return cached verdict and promote to MRU, or None on miss."""
        verdict = self._entries.get(signature)
        if verdict is None:
            return None
        # Promote to MRU: delete then re-insert.
        self._entries.move_to_end(signature)
        return verdict

    def set(self, signature: str, verdict: ScanVerdict) -> None:
        """Store verdict.  Evicts LRU entry when capacity is exceeded."""
        if signature in self._entries:
            # Re-insert at MRU end.
            self._entries.move_to_end(signature)
        self._entries[signature] = verdict
        if len(self._entries) > self._capacity:
            # popitem(last=False) removes the LRU (front) entry.
            self._entries.popitem(last=False)

    def size(self) -> int:
        """Current number of cached entries."""
        return len(self._entries)

    def clear(self) -> None:
        """Drop all cached entries (test helper / runtime reset)."""
        self._entries.clear()
