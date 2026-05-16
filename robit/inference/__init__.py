"""robit.inference — cross-session evidence accumulation substrate.

Port of ``wixie/shared/scripts/inference-engine.py``.  Algorithms U1-U6
preserved verbatim; env var and path conventions adapted for enchanter-agent.

Public exports (importable without the CLI layer):
    InferenceSubstrate  — thin facade over the core math functions
    emit                — append one artifact to artifacts.jsonl
    reconcile           — run Wald SPRT + Beta-Binomial + EMA; write catalog
    render_briefing     — render a per-plugin briefing markdown
    query               — search catalog by code / tag / pattern_id
    status              — return a status summary dict
    fingerprint         — U1: SHA-1 fingerprint for a record
    sprt_update         — U2: update running LLR
    sprt_verdict        — U2: elevation/retirement verdict
    beta_update         — U3: update Beta-Binomial parameters
    beta_mean           — U3: posterior mean
    beta_ci             — U3: 95% credible interval
    ema_weight          — U5: EMA decay weight
    reservoir_add       — U6: Vitter reservoir sampling
    env_enabled         — opt-in gate check
Path helpers:
    DEFAULT_STATE_DIR
    CATALOG_PATH
    ARTIFACTS_PATH
    BRIEFINGS_DIR
    resolve_state_dir
"""

from robit.inference.engine import (
    LAMBDA,
    LLR_ELEVATE,
    LLR_RETIRE,
    beta_ci,
    beta_mean,
    beta_update,
    emit,
    emit_unconditional,
    ema_weight,
    env_enabled,
    fingerprint,
    load_catalog,
    query,
    reconcile,
    render_briefing,
    reservoir_add,
    sprt_update,
    sprt_verdict,
    status,
)
from robit.inference.paths import (
    ARTIFACTS_PATH,
    BRIEFINGS_DIR,
    CATALOG_PATH,
    DEFAULT_STATE_DIR,
    resolve_state_dir,
)


class InferenceSubstrate:
    """Thin facade wrapping the substrate functions with an optional state_dir binding.

    Construct with a specific *state_dir* to run in an isolated sandbox (useful
    for tests and multi-tenant sessions).  Without arguments it picks up the
    env-var override or the default production path.
    """

    def __init__(self, state_dir=None) -> None:
        from pathlib import Path as _Path

        self._state_dir = _Path(state_dir) if state_dir is not None else None

    def emit(self, record: dict) -> bool:
        return emit(record, self._state_dir)

    def emit_unconditional(self, record: dict) -> None:
        emit_unconditional(record, self._state_dir)

    def reconcile(self) -> dict:
        return reconcile(self._state_dir)

    def render_briefing(self, plugin: str):
        return render_briefing(plugin, self._state_dir)

    def query(self, term: str) -> list:
        return query(term, self._state_dir)

    def status(self) -> dict:
        return status(self._state_dir)


__all__ = [
    "InferenceSubstrate",
    # math primitives
    "fingerprint",
    "sprt_update",
    "sprt_verdict",
    "beta_update",
    "beta_mean",
    "beta_ci",
    "ema_weight",
    "reservoir_add",
    # constants
    "LAMBDA",
    "LLR_ELEVATE",
    "LLR_RETIRE",
    # high-level API
    "emit",
    "emit_unconditional",
    "reconcile",
    "render_briefing",
    "query",
    "status",
    "env_enabled",
    "load_catalog",
    # path helpers
    "DEFAULT_STATE_DIR",
    "CATALOG_PATH",
    "ARTIFACTS_PATH",
    "BRIEFINGS_DIR",
    "resolve_state_dir",
]
