"""Per-instance cost-ledger store.

Tracks cumulative token totals per (session_id, vendor) and per session.
Detects tier boundary crossings against a configurable thresholds list.

Port of pech.adapter.ts (TS module-level state) and pech/ledger-store.ts
(JSONL file backing).  Python divergence: per-instance state rather than
module-level singletons so each CostLedger() is fully isolated — safe for
concurrent tests and multi-session orchestrators.

JSONL persistence is opt-in (ledger_path).  Default: in-memory only.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Default tier thresholds — remaining-budget fractions (0–1).
# Mirrors pech.adapter.ts _thresholds defaults.
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS: list[float] = [0.7, 0.3, 0.1]
"""Three remaining-fraction waypoints: HIGH→MED at 0.7, MED→LOW at 0.3,
LOW→CRITICAL at 0.1, CRITICAL→EXHAUSTED at 0.0."""


# ---------------------------------------------------------------------------
# Internal data shapes
# ---------------------------------------------------------------------------


@dataclass
class _VendorBudget:
    limit_tokens: int
    used: int = 0


@dataclass
class LedgerEntry:
    """One recorded token-cost observation."""

    ts: int
    session_id: str
    correlation_id: str
    plugin: str
    model: str
    vendor: str
    input_tokens: int
    output_tokens: int
    tool_call_cost: Optional[float] = None


def _now_ms() -> int:
    return int(time.time() * 1000)


# ---------------------------------------------------------------------------
# Tier helpers (faithful to TS computeTierLabel)
# ---------------------------------------------------------------------------

_TIER_LABELS = ("HIGH", "MED", "LOW", "CRITICAL")


def compute_tier_label(remaining_pct: float, thresholds: list[float]) -> str:
    """Return tier label for *remaining_pct* against *thresholds*.

    thresholds must be a list of three descending fractions corresponding to
    the HIGH/MED/LOW waypoints.  Anything below thresholds[2] is CRITICAL.
    """
    high, med, low = thresholds[0], thresholds[1], thresholds[2]
    if remaining_pct >= high:
        return "HIGH"
    if remaining_pct >= med:
        return "MED"
    if remaining_pct >= low:
        return "LOW"
    return "CRITICAL"


# ---------------------------------------------------------------------------
# JSONL store helper (opt-in)
# ---------------------------------------------------------------------------


class FileLedgerStore:
    """Append-only JSONL-backed ledger file.

    Best-effort: I/O errors are captured and returned as strings, never
    raised — the in-memory store is the source of truth at runtime.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        # Ensure parent directory exists.
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    def append(self, entry: LedgerEntry) -> Optional[str]:
        """Append one entry as a JSONL line. Returns None on success or an error message."""
        try:
            line = json.dumps(entry.__dict__) + "\n"
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line)
            return None
        except Exception as exc:  # noqa: BLE001
            return str(exc)

    def replay(self) -> list[LedgerEntry]:
        """Read all valid entries from the JSONL file.

        Tolerates missing files, empty files, and truncated/malformed lines —
        the file is observability, not a transactional store.
        """
        if not os.path.exists(self.path):
            return []
        try:
            raw = Path(self.path).read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001
            return []
        if not raw:
            return []

        out: list[LedgerEntry] = []
        for line in raw.splitlines():
            if not line.strip():
                continue
            try:
                d = json.loads(line)
                # Minimal shape check — same invariant as TS ledger-store.ts.
                if (
                    isinstance(d.get("ts"), (int, float))
                    and isinstance(d.get("correlation_id"), str)
                    and isinstance(d.get("input_tokens"), int)
                    and isinstance(d.get("output_tokens"), int)
                ):
                    out.append(
                        LedgerEntry(
                            ts=int(d["ts"]),
                            session_id=str(d.get("session_id", "")),
                            correlation_id=d["correlation_id"],
                            plugin=str(d.get("plugin", "unknown")),
                            model=str(d.get("model", "unknown")),
                            vendor=str(d.get("vendor", "unknown")),
                            input_tokens=int(d["input_tokens"]),
                            output_tokens=int(d["output_tokens"]),
                            tool_call_cost=(
                                float(d["tool_call_cost"])
                                if isinstance(d.get("tool_call_cost"), (int, float))
                                else None
                            ),
                        )
                    )
            except Exception:  # noqa: BLE001
                # skip malformed line
                pass
        return out


# ---------------------------------------------------------------------------
# CostLedgerStore — per-instance state
# ---------------------------------------------------------------------------


@dataclass
class CostLedgerStore:
    """Per-instance cost-ledger state.

    Args:
        thresholds: Three descending remaining-fraction waypoints.
                    Default: [0.7, 0.3, 0.1] (HIGH, MED, LOW boundaries).
        ledger_path: Optional path to a JSONL file for durable persistence.
                     Absent → pure in-memory mode (default for tests).
    """

    thresholds: list[float] = field(default_factory=lambda: list(DEFAULT_THRESHOLDS))
    ledger_path: Optional[str] = None

    # Internal — not part of the constructor signature
    _entries: list[LedgerEntry] = field(default_factory=list, init=False, repr=False)
    _session_totals: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _vendor_totals: dict[tuple[str, str], int] = field(
        default_factory=dict, init=False, repr=False
    )
    _vendor_budgets: dict[str, _VendorBudget] = field(
        default_factory=dict, init=False, repr=False
    )
    _last_tier: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _file_store: Optional[FileLedgerStore] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.ledger_path is not None:
            self._file_store = FileLedgerStore(self.ledger_path)
            # Replay pre-existing entries into the in-memory mirror.
            for entry in self._file_store.replay():
                self._apply(entry, persist=False)

    # ------------------------------------------------------------------
    # Budget configuration
    # ------------------------------------------------------------------

    def set_budget(self, vendor: str, limit_tokens: int) -> None:
        """Register or update the token budget for *vendor*."""
        existing = self._vendor_budgets.get(vendor)
        self._vendor_budgets[vendor] = _VendorBudget(
            limit_tokens=limit_tokens,
            used=existing.used if existing else 0,
        )

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        *,
        session_id: str,
        correlation_id: str,
        plugin: str,
        model: str,
        vendor: str,
        input_tokens: int,
        output_tokens: int,
        tool_call_cost: Optional[float] = None,
    ) -> Optional[str]:
        """Append one observation. Returns None on success or an error string if
        JSONL persistence fails (in-memory update always succeeds).
        """
        entry = LedgerEntry(
            ts=_now_ms(),
            session_id=session_id,
            correlation_id=correlation_id,
            plugin=plugin,
            model=model,
            vendor=vendor,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_call_cost=tool_call_cost,
        )
        return self._apply(entry, persist=True)

    def _apply(self, entry: LedgerEntry, *, persist: bool) -> Optional[str]:
        """Apply *entry* to in-memory state; optionally persist to JSONL."""
        self._entries.append(entry)

        total = entry.input_tokens + entry.output_tokens
        self._session_totals[entry.session_id] = (
            self._session_totals.get(entry.session_id, 0) + total
        )

        key: tuple[str, str] = (entry.session_id, entry.vendor)
        self._vendor_totals[key] = self._vendor_totals.get(key, 0) + total

        # Update vendor budget used counter.
        budget = self._vendor_budgets.get(entry.vendor)
        if budget is not None:
            budget.used += total

        store_error: Optional[str] = None
        if persist and self._file_store is not None:
            store_error = self._file_store.append(entry)

        return store_error

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def total(self, session_id: str) -> int:
        """Total tokens used in *session_id* (all vendors)."""
        return self._session_totals.get(session_id, 0)

    def vendor_total(self, session_id: str, vendor: str) -> int:
        """Total tokens used by *vendor* in *session_id*."""
        return self._vendor_totals.get((session_id, vendor), 0)

    def remaining(self, vendor: str) -> Optional[int]:
        """Tokens remaining for *vendor*, or None if no budget is registered."""
        budget = self._vendor_budgets.get(vendor)
        if budget is None:
            return None
        return max(0, budget.limit_tokens - budget.used)

    def entries(self) -> list[LedgerEntry]:
        """Read-only snapshot of all recorded entries."""
        return list(self._entries)

    # ------------------------------------------------------------------
    # Threshold crossing detection
    # ------------------------------------------------------------------

    def check_threshold_crossed(self, vendor: str) -> Optional[dict]:
        """Check whether a tier boundary was crossed for *vendor* after the
        most recent ``record()`` call.

        Returns a dict with ``{old_tier, new_tier, remaining_pct}`` when a
        crossing occurs, or None when no budget is registered, the vendor is
        not yet exhausted, or the tier is unchanged.

        Side-effect: updates ``_last_tier[vendor]`` to the new tier label.
        """
        budget = self._vendor_budgets.get(vendor)
        if budget is None:
            return None

        remaining_tokens = max(0, budget.limit_tokens - budget.used)
        remaining_pct = (
            remaining_tokens / budget.limit_tokens if budget.limit_tokens > 0 else 0.0
        )

        if remaining_pct <= 0.0:
            old_tier = self._last_tier.get(vendor, "HIGH")
            self._last_tier[vendor] = "EXHAUSTED"
            if old_tier != "EXHAUSTED":
                return {
                    "old_tier": old_tier,
                    "new_tier": "EXHAUSTED",
                    "remaining_pct": 0.0,
                }
            return None

        new_tier = compute_tier_label(remaining_pct, self.thresholds)
        # Cold-start: treat vendor as starting at HIGH (remaining=100%).
        old_tier = self._last_tier.get(vendor, compute_tier_label(1.0, self.thresholds))

        if new_tier != old_tier:
            self._last_tier[vendor] = new_tier
            return {
                "old_tier": old_tier,
                "new_tier": new_tier,
                "remaining_pct": remaining_pct,
            }

        self._last_tier[vendor] = new_tier
        return None

    def reset(self) -> None:
        """Clear all state — used in test teardown."""
        self._entries.clear()
        self._session_totals.clear()
        self._vendor_totals.clear()
        self._vendor_budgets.clear()
        self._last_tier.clear()
