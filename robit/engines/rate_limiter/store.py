"""Token-bucket rate-limiter store.

Per-instance state: dict of (vendor, session_id) → bucket.
Each bucket tracks current token count and the last refill timestamp.

Token bucket algorithm (from pech rate-shield / rate-check):
  tokens_new = min(capacity, tokens_old + elapsed * refill_rate)
  if tokens_new >= 1 → consume 1, return True (allowed)
  else → return False (exhausted)

Clock is injected via the ``clock`` constructor argument (defaults to
``time.monotonic``) so tests can monkeypatch time without relying on
module-level patches.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Tuple

_Key = Tuple[str, str]  # (vendor, session_id)


@dataclass
class _Bucket:
    tokens: float
    last_refill: float  # time.monotonic() timestamp
    capacity: float
    refill_rate: float  # tokens / second


@dataclass
class RateLimiterStore:
    """Per-instance token-bucket store.

    Args:
        capacity:     Maximum tokens per bucket (default 60).
        refill_rate:  Tokens added per second (default 1.0).
        clock:        Callable returning current time in fractional seconds.
                      Defaults to ``time.monotonic``.  Inject a fake clock in
                      tests to control refill behaviour deterministically.
    """

    capacity: float = 60.0
    refill_rate: float = 1.0
    clock: Callable[[], float] = field(default=time.monotonic, repr=False)
    _buckets: dict[_Key, _Bucket] = field(default_factory=dict, init=False, repr=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create(self, key: _Key) -> _Bucket:
        if key not in self._buckets:
            self._buckets[key] = _Bucket(
                tokens=self.capacity,
                last_refill=self.clock(),
                capacity=self.capacity,
                refill_rate=self.refill_rate,
            )
        return self._buckets[key]

    def _refill(self, bucket: _Bucket) -> None:
        """Apply elapsed-time refill in-place (lazy, called before consume)."""
        now = self.clock()
        elapsed = max(0.0, now - bucket.last_refill)
        bucket.tokens = min(bucket.capacity, bucket.tokens + elapsed * bucket.refill_rate)
        bucket.last_refill = now

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def try_consume(self, vendor: str, session_id: str) -> bool:
        """Attempt to consume one token from the (vendor, session_id) bucket.

        Returns:
            True  — token available; bucket decremented.
            False — bucket exhausted; no change to token count.
        """
        key = (vendor, session_id)
        bucket = self._get_or_create(key)
        self._refill(bucket)

        if bucket.tokens >= 1.0:
            bucket.tokens -= 1.0
            return True
        return False

    def tokens_available(self, vendor: str, session_id: str) -> float:
        """Return current token count after applying any pending refill.

        Creates the bucket at full capacity if it has never been seen.
        Does NOT consume a token.
        """
        key = (vendor, session_id)
        bucket = self._get_or_create(key)
        self._refill(bucket)
        return bucket.tokens

    def reset(self) -> None:
        """Clear all bucket state — used in test teardown."""
        self._buckets.clear()
