"""cost-ledger — per-request token ledger, vendor budget tracking, tier-boundary events.

Port of pech.adapter.ts (v0.3) + pech/ledger-store.ts.  Required plugin at
post-response; emits cost-ledger.appended on every call and
cost-ledger.threshold.crossed / cost-ledger.vendor.exhausted when vendor
budgets cross tier waypoints.

Token-key conventions: canonical tokens.input/tokens.output and legacy
flat input_tokens/output_tokens are both supported.
"""

from .adapter import CostLedger, adapter
from .store import CostLedgerStore, FileLedgerStore, LedgerEntry, compute_tier_label

__all__ = [
    "CostLedger",
    "CostLedgerStore",
    "FileLedgerStore",
    "LedgerEntry",
    "adapter",
    "compute_tier_label",
]
