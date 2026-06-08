"""G9 — codegen drift guard + emit verification.

These tests turn ``tools/codegen/generate.py`` into a CI gate:

  * ``test_check_passes_against_current_code`` goes RED the day someone changes a
    dataclass in robit/core/events.py or robit/core/verdict.py (or the ErrorCode
    enum) without updating schema/contracts.json. This is the anti-drift contract.
  * the emit tests assert ``--emit-python`` / ``--emit-ts`` produce non-empty,
    parseable output naming every shared contract.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GEN_PATH = REPO_ROOT / "tools" / "codegen" / "generate.py"

CONTRACTS = ["EnchantedEvent", "PluginAck", "Verdict"]
ALIASES = ["LifecyclePhase", "BudgetTier", "PluginAckStatus"]
ERROR_CODES = [
    "PARSE_ERROR",
    "INVALID_REQUEST",
    "METHOD_NOT_FOUND",
    "INVALID_PARAMS",
    "INTERNAL_ERROR",
    "SECURITY_VETO",
    "VENDOR_UNAVAILABLE",
    "SAMPLING_BOUND_EXCEEDED",
    "TOOL_NAME_COLLISION",
    "BUDGET_FLOOR_REFUSAL",
]


def _load_generate():
    """Import tools/codegen/generate.py as a module (not on the package path)."""
    spec = importlib.util.spec_from_file_location("g9_generate", GEN_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gen():
    return _load_generate()


@pytest.fixture(scope="module")
def schema(gen):
    return gen.load_schema()


# ---------------------------------------------------------------------------
# Drift guard
# ---------------------------------------------------------------------------

def test_check_passes_against_current_code(gen, schema):
    """The schema must match the live dataclasses + ErrorCode enum exactly."""
    problems = gen.check(schema)
    assert problems == [], "contract drift detected:\n" + "\n".join(problems)


def test_check_exit_code_zero(gen):
    """The CLI --check path returns 0 when there is no drift."""
    assert gen.main(["--check"]) == 0


def test_check_verifies_every_field_of_each_contract(gen, schema):
    """Guard against a hollow check: every live field is actually compared."""
    import dataclasses

    for contract, (module, cls_name) in gen.OBJECT_CONTRACTS.items():
        mod = importlib.import_module(module)
        cls = getattr(mod, cls_name)
        live = {f.name for f in dataclasses.fields(cls)}
        schema_side = set(gen.schema_fields(contract, schema).keys())
        assert live == schema_side, (
            f"{contract}: schema fields {schema_side} != live fields {live}"
        )


def test_check_detects_injected_drift(gen, schema):
    """A mutated schema copy must be reported as drift (the guard has teeth)."""
    import copy

    mutated = copy.deepcopy(schema)
    # Remove a Wave-1 field from the schema -> code now has an extra field.
    del mutated["$defs"]["EnchantedEvent"]["properties"]["hop_count"]
    problems = gen.check(mutated)
    assert any("hop_count" in p for p in problems), problems

    mutated2 = copy.deepcopy(schema)
    # Retype a field -> type drift.
    mutated2["$defs"]["Verdict"]["properties"]["plugin"]["type"] = "integer"
    problems2 = gen.check(mutated2)
    assert any("Verdict.plugin" in p and "type drift" in p for p in problems2), problems2

    mutated3 = copy.deepcopy(schema)
    # Change an error-code value -> enum value drift.
    mutated3["$defs"]["JsonRpcErrorCode"]["oneOf"][5]["const"] = -1
    problems3 = gen.check(mutated3)
    assert any("SECURITY_VETO" in p for p in problems3), problems3


# ---------------------------------------------------------------------------
# Emit: Python
# ---------------------------------------------------------------------------

def test_emit_python_nonempty_and_parses(gen, schema):
    src = gen.emit_python(schema)
    assert src.strip(), "emitted Python is empty"
    # Parseable.
    tree = ast.parse(src)
    class_names = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    for contract in CONTRACTS + ["ErrorCode"]:
        assert contract in class_names, f"{contract} missing from emitted Python"
    for alias in ALIASES:
        assert alias in src, f"{alias} alias missing from emitted Python"


def test_emit_python_executes(gen, schema, tmp_path):
    """The generated module imports + instantiates cleanly."""
    src = gen.emit_python(schema)
    p = tmp_path / "_generated_contracts.py"
    p.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("g9_emitted", p)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so dataclasses can resolve the module's namespace
    # for the deferred (PEP 563) string annotations.
    sys.modules["g9_emitted"] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop("g9_emitted", None)
    v = mod.Verdict(plugin="x", phase="anchor", reason="r")
    assert v.severity == "veto"
    ev = mod.EnchantedEvent(
        id="1", correlation_id="c", session_id="s", phase="anchor",
        topic="t", source="orchestrator", budget_tier="HIGH", ts=0,
    )
    assert ev.hop_count == 0 and ev.schema_version == 1
    ack = mod.PluginAck(status="ack")
    assert ack.verdict is None and ack.derived_events == []
    assert int(mod.ErrorCode.SECURITY_VETO) == -32099


# ---------------------------------------------------------------------------
# Emit: TypeScript
# ---------------------------------------------------------------------------

def test_emit_ts_nonempty_and_wellformed(gen, schema):
    ts = gen.emit_ts(schema)
    assert ts.strip(), "emitted TS is empty"
    # Balanced braces — a coarse parseability proxy without a TS compiler.
    assert ts.count("{") == ts.count("}"), "unbalanced braces in emitted TS"
    for contract in CONTRACTS:
        assert f"export interface {contract} {{" in ts, f"{contract} interface missing"
    for alias in ALIASES:
        assert f"export type {alias} =" in ts, f"{alias} type alias missing"
    assert "export enum ErrorCode {" in ts
    for code in ERROR_CODES:
        assert code in ts, f"{code} missing from emitted TS enum"
    # Custom-range sentinel value present.
    assert "-32099" in ts


def test_emit_cli_writes_files(gen, tmp_path):
    py = tmp_path / "out.py"
    ts = tmp_path / "out.ts"
    rc = gen.main(["--emit-python", str(py), "--emit-ts", str(ts)])
    assert rc == 0
    assert py.read_text(encoding="utf-8").strip()
    assert ts.read_text(encoding="utf-8").strip()
