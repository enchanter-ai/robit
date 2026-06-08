#!/usr/bin/env python3
"""tools/codegen/generate.py — codegen + drift guard for the shared wire contracts (G9).

Single language-neutral source of truth lives in ``schema/contracts.json`` (JSON
Schema draft 2020-12). This stdlib-only tool can:

  --emit-python <path>   write Python dataclass definitions matching the schema.
  --emit-ts <path>       write TypeScript interface/type definitions matching the schema.
  --check                introspect the live hand-written dataclasses in
                         robit/core/events.py + robit/core/verdict.py (and the
                         ErrorCode enum in robit/protocol/jsonrpc.py) via
                         dataclasses.fields and fail (exit 2) with a clear diff if
                         the schema and the code have drifted.

This wave codegen is *additive verification* — the emitted Python module is NOT
wired into imports; only ``--check`` runs in the test suite as the drift guard.

stdlib only: json, dataclasses, argparse, typing, enum, importlib, pathlib, sys.
No third-party deps (no jsonschema lib) — the field comparison is hand-rolled.
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import importlib
import json
import sys
from pathlib import Path
from typing import Any

# Repo root = three levels up from this file (tools/codegen/generate.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "schema" / "contracts.json"

# Object contracts in the schema that map to live Python dataclasses, paired
# with their (module, class) source of truth.
OBJECT_CONTRACTS = {
    "EnchantedEvent": ("robit.core.events", "EnchantedEvent"),
    "PluginAck": ("robit.core.events", "PluginAck"),
    "Verdict": ("robit.core.verdict", "Verdict"),
}
# The JSON-RPC error-code enum contract.
ENUM_CONTRACT = ("JsonRpcErrorCode", "robit.protocol.jsonrpc", "ErrorCode")


# ---------------------------------------------------------------------------
# Schema loading
# ---------------------------------------------------------------------------

def load_schema(path: Path = SCHEMA_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _defs(schema: dict[str, Any]) -> dict[str, Any]:
    return schema["$defs"]


def _ref_name(ref: str) -> str:
    """'#/$defs/Verdict' -> 'Verdict'."""
    return ref.rsplit("/", 1)[-1]


# ---------------------------------------------------------------------------
# Canonical type tokens — the comparison currency between schema and code
# ---------------------------------------------------------------------------
#
# Both the schema side and the live-dataclass side are reduced to a small set of
# canonical, normalised type tokens so they can be compared without depending on
# import-time type resolution (the dataclasses use ``from __future__ import
# annotations`` so field types arrive as strings).


def _canon_schema_type(prop: dict[str, Any], defs: dict[str, Any]) -> str:
    """Reduce a JSON-Schema property node to a canonical type token."""
    if "$ref" in prop:
        return _ref_name(prop["$ref"])

    if "oneOf" in prop:
        parts = [_canon_schema_type(sub, defs) for sub in prop["oneOf"]]
        return _normalise_optional(parts)

    t = prop.get("type")
    if isinstance(t, list):
        parts = [_canon_scalar(x) for x in t]
        return _normalise_optional(parts)
    if isinstance(t, str):
        if t == "array":
            item = prop.get("items", {})
            return f"list[{_canon_schema_type(item, defs)}]"
        return _canon_scalar(t)
    raise ValueError(f"cannot canonicalise schema property: {prop!r}")


def _canon_scalar(json_type: str) -> str:
    return {
        "string": "str",
        "integer": "int",
        "number": "float",
        "boolean": "bool",
        "object": "Mapping",
        "null": "None",
        "array": "list",
    }[json_type]


def _normalise_optional(parts: list[str]) -> str:
    """Sort + dedupe a union's parts into a stable canonical token.

    'str | None' and 'None | str' both canonicalise to 'None|str'.
    """
    uniq = sorted(set(parts))
    return "|".join(uniq)


def _canon_py_type(type_repr: str) -> str:
    """Reduce a dataclass field's annotation (a string under PEP 563) to a token.

    Handles the exact shapes used in the live contracts:
      'str', 'int', 'bool'
      'str | None'
      'LifecyclePhase', 'BudgetTier', 'PluginAckStatus'   (Literal aliases)
      'Verdict | None'
      'Mapping[str, object]'                               (payload)
      'list[EnchantedEvent]'                               (derived_events)
    """
    s = type_repr.strip()

    # Union via '|'
    if "|" in s and not s.startswith("list[") and not s.startswith("Mapping["):
        parts = [_canon_py_type(p) for p in s.split("|")]
        return _normalise_optional(parts)

    # list[...]
    if s.startswith("list[") and s.endswith("]"):
        inner = s[len("list["):-1]
        return f"list[{_canon_py_type(inner)}]"

    # Mapping[...] / dict[...] -> the schema 'object' canon is 'Mapping'
    if s.startswith(("Mapping[", "dict[", "Dict[")):
        return "Mapping"
    if s in ("Mapping", "dict", "Dict"):
        return "Mapping"

    return {
        "str": "str",
        "int": "int",
        "float": "float",
        "bool": "bool",
        "None": "None",
        "NoneType": "None",
        # Literal type aliases pass through by name (schema uses matching $def names)
    }.get(s, s)


# ---------------------------------------------------------------------------
# Schema -> normalised field model (name -> (canon_type, required))
# ---------------------------------------------------------------------------

def schema_fields(contract: str, schema: dict[str, Any]) -> dict[str, tuple[str, bool]]:
    defs = _defs(schema)
    node = defs[contract]
    required = set(node.get("required", []))
    out: dict[str, tuple[str, bool]] = {}
    for name, prop in node["properties"].items():
        out[name] = (_canon_schema_type(prop, defs), name in required)
    return out


def code_fields(module: str, cls_name: str) -> dict[str, tuple[str, bool]]:
    """Live dataclass -> name -> (canon_type, required).

    'required' == the field has NO default and NO default_factory.
    """
    mod = importlib.import_module(module)
    cls = getattr(mod, cls_name)
    out: dict[str, tuple[str, bool]] = {}
    for f in dataclasses.fields(cls):
        has_default = (
            f.default is not dataclasses.MISSING
            or f.default_factory is not dataclasses.MISSING  # type: ignore[misc]
        )
        out[f.name] = (_canon_py_type(_type_to_str(f.type)), not has_default)
    return out


def _type_to_str(t: Any) -> str:
    """Field types are strings under PEP 563, but be robust if a real type leaks in."""
    if isinstance(t, str):
        return t
    return getattr(t, "__name__", str(t))


# ---------------------------------------------------------------------------
# --check: drift detection
# ---------------------------------------------------------------------------

def diff_contract(contract: str, schema: dict[str, Any]) -> list[str]:
    module, cls_name = OBJECT_CONTRACTS[contract]
    want = schema_fields(contract, schema)
    have = code_fields(module, cls_name)
    problems: list[str] = []

    for name in want.keys() - have.keys():
        problems.append(f"{contract}: field '{name}' in schema but MISSING from code ({cls_name})")
    for name in have.keys() - want.keys():
        problems.append(f"{contract}: field '{name}' in code ({cls_name}) but MISSING from schema")

    for name in want.keys() & have.keys():
        w_type, w_req = want[name]
        h_type, h_req = have[name]
        if w_type != h_type:
            problems.append(
                f"{contract}.{name}: type drift — schema '{w_type}' != code '{h_type}'"
            )
        if w_req != h_req:
            problems.append(
                f"{contract}.{name}: optionality drift — schema "
                f"{'required' if w_req else 'optional'} != code "
                f"{'required' if h_req else 'optional'}"
            )
    return problems


def diff_enum(schema: dict[str, Any]) -> list[str]:
    contract, module, cls_name = ENUM_CONTRACT
    node = _defs(schema)[contract]
    want: dict[str, int] = {sub["title"]: sub["const"] for sub in node["oneOf"]}

    mod = importlib.import_module(module)
    cls = getattr(mod, cls_name)
    assert issubclass(cls, enum.IntEnum)
    have: dict[str, int] = {m.name: int(m.value) for m in cls}

    problems: list[str] = []
    for name in want.keys() - have.keys():
        problems.append(f"{contract}: code '{name}' in schema but MISSING from {cls_name}")
    for name in have.keys() - want.keys():
        problems.append(f"{contract}: code '{name}' in {cls_name} but MISSING from schema")
    for name in want.keys() & have.keys():
        if want[name] != have[name]:
            problems.append(
                f"{contract}.{name}: value drift — schema {want[name]} != code {have[name]}"
            )
    return problems


def check(schema: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    for contract in OBJECT_CONTRACTS:
        problems.extend(diff_contract(contract, schema))
    problems.extend(diff_enum(schema))
    return problems


# ---------------------------------------------------------------------------
# --emit-python
# ---------------------------------------------------------------------------

_PY_LITERAL_ALIASES = {"LifecyclePhase", "BudgetTier", "PluginAckStatus"}


def _py_field_type(prop: dict[str, Any], defs: dict[str, Any]) -> str:
    """Render a JSON-Schema property as a Python annotation string."""
    if "$ref" in prop:
        return _ref_name(prop["$ref"])
    if "oneOf" in prop:
        return " | ".join(_py_field_type(s, defs) for s in prop["oneOf"])
    t = prop.get("type")
    if isinstance(t, list):
        return " | ".join(_py_scalar(x) for x in t)
    if t == "array":
        return f"list[{_py_field_type(prop.get('items', {}), defs)}]"
    if t == "object":
        return "Mapping[str, object]"
    return _py_scalar(t)


def _py_scalar(json_type: str) -> str:
    return {
        "string": "str",
        "integer": "int",
        "number": "float",
        "boolean": "bool",
        "null": "None",
        "object": "Mapping[str, object]",
    }[json_type]


def _py_default(prop: dict[str, Any]) -> str | None:
    if "default" not in prop:
        return None
    d = prop["default"]
    if d == {} and prop.get("type") == "object":
        return "field(default_factory=dict)"
    if d == [] and prop.get("type") == "array":
        return "field(default_factory=list)"
    return repr(d)


def emit_python(schema: dict[str, Any]) -> str:
    defs = _defs(schema)
    lines: list[str] = [
        '"""GENERATED by tools/codegen/generate.py from schema/contracts.json — DO NOT EDIT.',
        "",
        "Additive verification artifact (G9). NOT wired into imports this wave; the",
        "hand-written dataclasses in robit/core remain the live source of truth.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "from dataclasses import dataclass, field",
        "from enum import IntEnum",
        "from typing import Literal, Mapping",
        "",
        "",
    ]

    # Literal aliases
    for alias in ("LifecyclePhase", "BudgetTier", "PluginAckStatus"):
        members = defs[alias]["enum"]
        rendered = ", ".join(repr(m) for m in members)
        lines.append(f"{alias} = Literal[{rendered}]")
    lines.append("")
    lines.append("")

    # ErrorCode enum
    enum_contract = ENUM_CONTRACT[0]
    enum_node = defs[enum_contract]
    lines.append("class ErrorCode(IntEnum):")
    lines.append('    """JSON-RPC 2.0 error codes + robit custom range (-32099..-32000)."""')
    lines.append("")
    for sub in enum_node["oneOf"]:
        lines.append(f"    {sub['title']} = {sub['const']}")
    lines.append("")
    lines.append("")

    # Object dataclasses, emitted in dependency order
    for contract in ("Verdict", "EnchantedEvent", "PluginAck"):
        node = defs[contract]
        frozen = node.get("x-python-frozen", False)
        decorator = "@dataclass(frozen=True)" if frozen else "@dataclass"
        lines.append(decorator)
        lines.append(f"class {contract}:")
        required = set(node.get("required", []))
        props = node["properties"]
        # required fields first (no defaults), then optional — preserves valid
        # dataclass ordering regardless of schema property order.
        ordered = [n for n in props if n in required] + [n for n in props if n not in required]
        for name in ordered:
            prop = props[name]
            ann = _py_field_type(prop, defs)
            default = _py_default(prop)
            if name in required:
                lines.append(f"    {name}: {ann}")
            else:
                lines.append(f"    {name}: {ann} = {default}")
        lines.append("")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# --emit-ts
# ---------------------------------------------------------------------------

def _ts_type(prop: dict[str, Any], defs: dict[str, Any]) -> str:
    if "$ref" in prop:
        return _ref_name(prop["$ref"])
    if "oneOf" in prop:
        return " | ".join(_ts_type(s, defs) for s in prop["oneOf"])
    t = prop.get("type")
    if isinstance(t, list):
        return " | ".join(_ts_scalar(x) for x in t)
    if t == "array":
        return f"{_ts_type(prop.get('items', {}), defs)}[]"
    return _ts_scalar(t)


def _ts_scalar(json_type: str) -> str:
    return {
        "string": "string",
        "integer": "number",
        "number": "number",
        "boolean": "boolean",
        "null": "null",
        "object": "Record<string, unknown>",
    }[json_type]


def emit_ts(schema: dict[str, Any]) -> str:
    defs = _defs(schema)
    lines: list[str] = [
        "// GENERATED by tools/codegen/generate.py from schema/contracts.json — DO NOT EDIT.",
        "//",
        "// Shared wire contracts for beholder (TypeScript) to consume. Generated from the",
        "// same language-neutral source of truth as the robit (Python) types (G9).",
        "",
    ]

    # Literal union type aliases
    for alias in ("LifecyclePhase", "BudgetTier", "PluginAckStatus"):
        members = defs[alias]["enum"]
        rendered = " | ".join(json.dumps(m) for m in members)
        lines.append(f"export type {alias} = {rendered};")
    lines.append("")

    # ErrorCode -> TS enum
    enum_node = defs[ENUM_CONTRACT[0]]
    lines.append("export enum ErrorCode {")
    for sub in enum_node["oneOf"]:
        lines.append(f"  {sub['title']} = {sub['const']},")
    lines.append("}")
    lines.append("")

    # Object interfaces
    for contract in ("Verdict", "EnchantedEvent", "PluginAck"):
        node = defs[contract]
        required = set(node.get("required", []))
        lines.append(f"export interface {contract} {{")
        for name, prop in node["properties"].items():
            optional = "" if name in required else "?"
            lines.append(f"  {name}{optional}: {_ts_type(prop, defs)};")
        lines.append("}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Shared-contract codegen + drift guard (G9).")
    parser.add_argument("--schema", type=Path, default=SCHEMA_PATH, help="Path to contracts.json")
    parser.add_argument("--emit-python", type=Path, metavar="PATH", help="Write generated Python dataclasses")
    parser.add_argument("--emit-ts", type=Path, metavar="PATH", help="Write generated TypeScript types")
    parser.add_argument("--check", action="store_true", help="Fail if code drifts from the schema")
    args = parser.parse_args(argv)

    # Ensure the repo root is importable for --check introspection.
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    schema = load_schema(args.schema)
    did_something = False

    if args.emit_python:
        args.emit_python.parent.mkdir(parents=True, exist_ok=True)
        args.emit_python.write_text(emit_python(schema), encoding="utf-8")
        print(f"wrote Python -> {args.emit_python}")
        did_something = True

    if args.emit_ts:
        args.emit_ts.parent.mkdir(parents=True, exist_ok=True)
        args.emit_ts.write_text(emit_ts(schema), encoding="utf-8")
        print(f"wrote TypeScript -> {args.emit_ts}")
        did_something = True

    if args.check:
        problems = check(schema)
        if problems:
            print("CONTRACT DRIFT — schema/contracts.json does not match the live code:", file=sys.stderr)
            for p in problems:
                print(f"  - {p}", file=sys.stderr)
            return 2
        print("OK — schema matches live dataclasses + ErrorCode enum (no drift).")
        did_something = True

    if not did_something:
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
