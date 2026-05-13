"""Tests for enchanter.insights_cli — calls main() directly without subprocess.

All tests use capsys to capture stdout/stderr and inspect return codes.

Coverage:
    T01  enchanter-insights version          → 0, prints version string
    T02  enchanter-insights (no args)        → 1, prints help
    T03  enchanter-insights engines list     → 0, lists 14 engines
    T04  enchanter-insights engines list --json → 0, valid JSON list of 14
    T05  enchanter-insights engines show destructive-op-gate → 0, shows manifest fields
    T06  enchanter-insights engines show nonexistent → 1, clear error message
    T07  enchanter-insights conduct list     → 0, at least 10 conduct modules
    T08  enchanter-insights conduct list --json → 0, valid JSON list
    T09  enchanter-insights conduct show discipline → 0, prints body text
    T10  enchanter-insights conduct show nonexistent → 1, clear error
    T11  enchanter-insights tier route orchestrator → 0, prints claude-opus-4-7
    T12  enchanter-insights tier route bogus_class → 1
    T13  enchanter-insights status           → 0, non-empty aggregate output
    T14  enchanter-insights status --json    → 0, valid JSON with expected keys
    T15  enchanter-insights inference status → 0, prints summary
    T16  enchanter-insights inference reconcile → 0 (handles empty state dir gracefully)
"""

from __future__ import annotations

import json

import pytest

from enchanter.insights_cli import main


# ─── T01: version ─────────────────────────────────────────────────────────────

def test_version_exit_code(capsys):
    rc = main(["version"])
    assert rc == 0


def test_version_output_contains_version_string(capsys):
    main(["version"])
    out = capsys.readouterr().out
    assert "enchanter-agent" in out
    from enchanter import __version__
    assert __version__ in out


# ─── T02: no args → help + exit 1 ─────────────────────────────────────────────

def test_no_args_exits_1(capsys):
    rc = main([])
    assert rc == 1


def test_no_args_prints_help(capsys):
    main([])
    # argparse help goes to stdout
    out = capsys.readouterr().out
    assert "enchanter" in out.lower() or "usage" in out.lower()


# ─── T03: engines list ────────────────────────────────────────────────────────

def test_engines_list_exit_0(capsys):
    rc = main(["engines", "list"])
    assert rc == 0


def test_engines_list_shows_14_engines(capsys):
    main(["engines", "list"])
    out = capsys.readouterr().out
    # Each engine occupies one row in the table.  Count non-header, non-separator
    # lines that look like data rows (contain at least two "  " separators).
    lines = [l for l in out.splitlines() if "  " in l and not l.startswith("-")]
    # Remove header line.
    data_lines = [l for l in lines[1:] if l.strip()]
    assert len(data_lines) == 14, (
        f"Expected 14 engine rows, got {len(data_lines)}. Lines:\n{out}"
    )


# ─── T04: engines list --json ─────────────────────────────────────────────────

def test_engines_list_json_exit_0(capsys):
    rc = main(["engines", "list", "--json"])
    assert rc == 0


def test_engines_list_json_is_valid_list_of_14(capsys):
    main(["engines", "list", "--json"])
    out = capsys.readouterr().out
    data = json.loads(out)
    assert isinstance(data, list)
    assert len(data) == 14, f"Expected 14 entries, got {len(data)}"
    # Each entry has at minimum a 'name' key.
    for entry in data:
        assert "name" in entry
        assert "description" in entry
        assert "phases" in entry


# ─── T05: engines show (known) ────────────────────────────────────────────────

def test_engines_show_known_exit_0(capsys):
    rc = main(["engines", "show", "destructive-op-gate"])
    assert rc == 0


def test_engines_show_known_contains_manifest_fields(capsys):
    main(["engines", "show", "destructive-op-gate"])
    out = capsys.readouterr().out
    assert "destructive-op-gate" in out
    assert "budget_tier" in out
    assert "adapter" in out


# ─── T06: engines show (unknown) ─────────────────────────────────────────────

def test_engines_show_unknown_exit_1(capsys):
    rc = main(["engines", "show", "nonexistent-engine-xyz"])
    assert rc == 1


def test_engines_show_unknown_error_message(capsys):
    main(["engines", "show", "nonexistent-engine-xyz"])
    err = capsys.readouterr().err
    assert "not found" in err.lower() or "nonexistent" in err.lower()


# ─── T07: conduct list ────────────────────────────────────────────────────────

def test_conduct_list_exit_0(capsys):
    rc = main(["conduct", "list"])
    assert rc == 0


def test_conduct_list_shows_at_least_10_modules(capsys):
    main(["conduct", "list"])
    out = capsys.readouterr().out
    # Count data rows (skip header and separator).
    lines = out.splitlines()
    data_lines = [
        l for l in lines[2:]  # skip header + separator
        if l.strip() and not l.startswith("-")
    ]
    assert len(data_lines) >= 10, (
        f"Expected at least 10 conduct modules, got {len(data_lines)}.\n{out}"
    )


# ─── T08: conduct list --json ─────────────────────────────────────────────────

def test_conduct_list_json_is_valid(capsys):
    rc = main(["conduct", "list", "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert isinstance(data, list)
    assert len(data) >= 10
    for item in data:
        assert "name" in item
        assert "package" in item
        assert "enforcement" in item


# ─── T09: conduct show (known) ────────────────────────────────────────────────

def test_conduct_show_known_exit_0(capsys):
    rc = main(["conduct", "show", "discipline"])
    assert rc == 0


def test_conduct_show_known_prints_body(capsys):
    main(["conduct", "show", "discipline"])
    out = capsys.readouterr().out
    # The discipline module body should contain some non-trivial content.
    assert len(out) > 100


# ─── T10: conduct show (unknown) ─────────────────────────────────────────────

def test_conduct_show_unknown_exit_1(capsys):
    rc = main(["conduct", "show", "nonexistent-module-xyz"])
    assert rc == 1


def test_conduct_show_unknown_error_message(capsys):
    main(["conduct", "show", "nonexistent-module-xyz"])
    err = capsys.readouterr().err
    assert "not found" in err.lower() or "nonexistent" in err.lower()


# ─── T11: tier route orchestrator ────────────────────────────────────────────

def test_tier_route_orchestrator_exit_0(capsys):
    rc = main(["tier", "route", "orchestrator"])
    assert rc == 0


def test_tier_route_orchestrator_output(capsys):
    main(["tier", "route", "orchestrator"])
    out = capsys.readouterr().out.strip()
    assert out == "claude-opus-4-7", f"Expected claude-opus-4-7, got {out!r}"


# ─── T12: tier route bogus class ─────────────────────────────────────────────

def test_tier_route_bogus_class_exit_1(capsys):
    rc = main(["tier", "route", "bogus_class_xyz"])
    assert rc == 1


# ─── T13: status text ────────────────────────────────────────────────────────

def test_status_exit_0(capsys):
    rc = main(["status"])
    assert rc == 0


def test_status_output_non_empty(capsys):
    main(["status"])
    out = capsys.readouterr().out
    assert len(out) > 10
    assert "enchanter-agent" in out


# ─── T14: status --json ───────────────────────────────────────────────────────

def test_status_json_exit_0(capsys):
    rc = main(["status", "--json"])
    assert rc == 0


def test_status_json_has_expected_keys(capsys):
    main(["status", "--json"])
    out = capsys.readouterr().out
    data = json.loads(out)
    assert "version" in data
    assert "engine_count" in data
    assert "conduct_count" in data
    assert "inference" in data
    assert "tier_defaults" in data
    assert data["engine_count"] == 14
    assert data["conduct_count"] >= 10


# ─── T15: inference status ───────────────────────────────────────────────────

def test_inference_status_exit_0(capsys):
    rc = main(["inference", "status"])
    assert rc == 0


def test_inference_status_output(capsys):
    main(["inference", "status"])
    out = capsys.readouterr().out
    # Should print enabled: flag and state_dir at minimum.
    assert "enabled" in out
    assert "state_dir" in out


# ─── T16: inference reconcile (empty state dir) ──────────────────────────────

def test_inference_reconcile_empty_state_does_not_crash(capsys, tmp_path, monkeypatch):
    """Reconcile on empty state dir should exit 0 (no artifacts yet)."""
    monkeypatch.setenv("ENCHANTER_INFERENCE_STATE", str(tmp_path))
    rc = main(["inference", "reconcile"])
    # Should not crash — either 0 or a graceful message.
    assert rc == 0


# ─── Extra: noun without verb prints help ────────────────────────────────────

def test_engines_without_verb_exits_1(capsys):
    rc = main(["engines"])
    assert rc == 1


def test_conduct_without_verb_exits_1(capsys):
    rc = main(["conduct"])
    assert rc == 1
