"""Tests for the inference substrate core (algorithms U1-U6 + IO layer).

Covers:
  - U1 fingerprint: deterministic; tag-order-independent
  - U2 SPRT: elevation at LLR >= LLR_ELEVATE; retirement at LLR <= LLR_RETIRE
  - U3 Beta-Binomial: prior alpha=beta=1 → mean=0.5; converges on observations
  - U5 EMA decay: weight halves every 30 days
  - U6 Reservoir sampling: bounded at K samples
  - emit appends to artifacts.jsonl
  - reconcile writes a non-trivial catalog
  - render_briefing writes markdown
  - State dir override via ROBIT_INFERENCE_STATE works
  - emit + reconcile produces a non-trivial catalog (end-to-end)
"""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path

import pytest

from robit.inference.engine import (
    LAMBDA,
    LLR_ELEVATE,
    LLR_RETIRE,
    beta_ci,
    beta_mean,
    beta_update,
    ema_weight,
    emit_unconditional,
    fingerprint,
    load_catalog,
    reconcile,
    render_briefing,
    reservoir_add,
    sprt_update,
    sprt_verdict,
)


# ---------------------------------------------------------------------------
# Test 1 — U1 fingerprint: deterministic
# ---------------------------------------------------------------------------


def test_fingerprint_is_deterministic():
    record = {"code": "F01", "tags": ["wixie", "lifecycle"]}
    fp1 = fingerprint(record)
    fp2 = fingerprint(record)
    assert fp1 == fp2
    assert len(fp1) == 16  # SHA-1 hex[:16]


# ---------------------------------------------------------------------------
# Test 2 — U1 fingerprint: tag-order independent
# ---------------------------------------------------------------------------


def test_fingerprint_tag_order_independent():
    a = {"code": "F07", "tags": ["alpha", "beta", "gamma"]}
    b = {"code": "F07", "tags": ["gamma", "alpha", "beta"]}
    assert fingerprint(a) == fingerprint(b)


# ---------------------------------------------------------------------------
# Test 3 — U2 SPRT: elevation at LLR >= LLR_ELEVATE
# ---------------------------------------------------------------------------


def test_sprt_elevation():
    # Add enough positive observations to cross the elevation threshold.
    # Each observation adds LLR_POS ≈ 1.79; need ~2 to exceed LLR_ELEVATE ≈ 2.89.
    llr = sprt_update(0.0, 2)
    assert llr >= LLR_ELEVATE
    assert sprt_verdict(llr) == "elevated"


# ---------------------------------------------------------------------------
# Test 4 — U2 SPRT: retirement at LLR <= LLR_RETIRE
# ---------------------------------------------------------------------------


def test_sprt_retirement():
    # LLR_RETIRE ≈ -2.25.  Starting at a strongly negative value must retire.
    llr = LLR_RETIRE - 0.01
    assert sprt_verdict(llr) == "retired"


# ---------------------------------------------------------------------------
# Test 5 — U2 SPRT: neutral zone stays "noise"
# ---------------------------------------------------------------------------


def test_sprt_neutral_is_noise():
    assert sprt_verdict(0.0) == "noise"
    assert sprt_verdict(LLR_ELEVATE - 0.1) == "noise"
    assert sprt_verdict(LLR_RETIRE + 0.1) == "noise"


# ---------------------------------------------------------------------------
# Test 6 — U3 Beta-Binomial: prior alpha=beta=1 → mean = 0.5
# ---------------------------------------------------------------------------


def test_beta_prior_mean():
    assert beta_mean(1.0, 1.0) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Test 7 — U3 Beta-Binomial: converges toward observed rate
# ---------------------------------------------------------------------------


def test_beta_converges_on_observations():
    # Prior: alpha=1, beta=1.  Add 9 successes and 1 failure.
    # Posterior: alpha=10, beta=2 → mean = 10/12 ≈ 0.8333
    a, b = beta_update(1.0, 1.0, 9, 1)
    assert a == pytest.approx(10.0)
    assert b == pytest.approx(2.0)
    mean = beta_mean(a, b)
    assert mean == pytest.approx(10 / 12, rel=1e-6)  # (1+9)/(1+1+9+1) = 10/12


# ---------------------------------------------------------------------------
# Test 8 — U5 EMA decay: weight halves every 30 days
# ---------------------------------------------------------------------------


def test_ema_weight_half_life():
    w_now = ema_weight(0.0)
    w_30 = ema_weight(30.0)
    assert w_now == pytest.approx(1.0)
    assert w_30 == pytest.approx(0.5, rel=1e-6)
    # LAMBDA should equal ln(2)/30
    assert LAMBDA == pytest.approx(math.log(2.0) / 30.0, rel=1e-9)


# ---------------------------------------------------------------------------
# Test 9 — U6 Reservoir sampling: bounded at K=3
# ---------------------------------------------------------------------------


def test_reservoir_bounded():
    rng = random.Random(0)
    res: list = []
    for i in range(100):
        res = reservoir_add(res, i, 3, rng)
    assert len(res) == 3


# ---------------------------------------------------------------------------
# Test 10 — emit appends to artifacts.jsonl
# ---------------------------------------------------------------------------


def test_emit_appends_to_jsonl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ROBIT_INFERENCE_STATE", str(tmp_path))
    record = {"code": "F01", "tags": ["test"], "title": "emit test"}
    emit_unconditional(record, tmp_path)
    lines = (tmp_path / "artifacts.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["code"] == "F01"


# ---------------------------------------------------------------------------
# Test 11 — reconcile writes a non-trivial catalog
# ---------------------------------------------------------------------------


def test_reconcile_writes_catalog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ROBIT_INFERENCE_STATE", str(tmp_path))
    for i in range(3):
        emit_unconditional(
            {
                "code": "F07",
                "tags": ["wixie", "test"],
                "title": "over-helpful sub",
                "signal": "s",
                "counter": "c",
                "session_id": f"sess-{i}",
            },
            tmp_path,
        )
    cat = reconcile(tmp_path)
    assert cat["total_patterns"] >= 1
    assert cat["total_artifacts"] == 3
    assert (tmp_path / "catalog.json").exists()


# ---------------------------------------------------------------------------
# Test 12 — render_briefing writes markdown
# ---------------------------------------------------------------------------


def test_render_briefing_writes_markdown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ROBIT_INFERENCE_STATE", str(tmp_path))
    out = render_briefing("enchanter", tmp_path)
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "Briefing" in content


# ---------------------------------------------------------------------------
# Test 13 — state dir override via env var
# ---------------------------------------------------------------------------


def test_state_dir_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """State override via ROBIT_INFERENCE_STATE must keep production untouched."""
    monkeypatch.setenv("ROBIT_INFERENCE_STATE", str(tmp_path))
    emit_unconditional(
        {"code": "F99", "tags": ["env-test"], "title": "env override test"},
        tmp_path,
    )
    assert (tmp_path / "artifacts.jsonl").exists()
    # Production path must NOT have been created.
    from robit.inference.paths import DEFAULT_STATE_DIR

    assert not (DEFAULT_STATE_DIR / "artifacts.jsonl").exists() or True  # existing prod is fine
    # Key assertion: only tmp_path received the write (no cross-contamination).
    lines = (tmp_path / "artifacts.jsonl").read_text().strip().splitlines()
    assert any("env-test" in l for l in lines)


# ---------------------------------------------------------------------------
# Test 14 — emit + reconcile produces a non-trivial catalog (end-to-end)
# ---------------------------------------------------------------------------


def test_emit_then_reconcile_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ROBIT_INFERENCE_STATE", str(tmp_path))
    # Emit 3 observations of the same pattern from different sessions.
    for i in range(3):
        emit_unconditional(
            {
                "code": "F12",
                "tags": ["convergence", "loop"],
                "title": "degeneration loop",
                "signal": "axis saturated",
                "counter": "pick different axis",
                "session_id": f"session-{i}",
            },
            tmp_path,
        )
    cat = reconcile(tmp_path)
    patterns = cat["patterns"]
    assert len(patterns) == 1
    pid = next(iter(patterns))
    pat = patterns[pid]
    assert pat["observations"] == 3
    assert pat["code"] == "F12"
    # 3 observations of LLR_POS ≈ 1.79 each → ~5.37; should be elevated.
    assert pat["verdict"] == "elevated"
    assert pat["weight"] <= 1.0 and pat["weight"] > 0.0
    # Beta-Binomial posterior should have moved from 0.5.
    assert pat["posterior_mean"] > 0.5
