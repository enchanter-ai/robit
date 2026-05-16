"""robit.insighter.format — pure formatting helpers for CLI output.

All functions are pure (no I/O, no sys.stdout).  Each returns a string.
Callers write the string themselves so tests can inspect it without
capturing stdout.
"""

from __future__ import annotations

import json
from typing import Any


# ─── Table helpers ────────────────────────────────────────────────────────────


def _col_widths(headers: list[str], rows: list[list[str]]) -> list[int]:
    """Return the max width for each column."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    return widths


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a plain text table with padded columns and a header separator."""
    if not rows:
        return "  (none)\n"
    widths = _col_widths(headers, rows)
    sep = "  ".join("-" * w for w in widths)
    header_row = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    lines = [header_row, sep]
    for row in rows:
        lines.append("  ".join(cell.ljust(w) for cell, w in zip(row, widths)))
    return "\n".join(lines) + "\n"


# ─── JSON helpers ─────────────────────────────────────────────────────────────


def format_json(data: Any) -> str:
    """Serialize *data* as indented JSON (UTF-8 safe)."""
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


# ─── Engine formatters ────────────────────────────────────────────────────────


def engines_list_table(manifests: list[dict[str, Any]]) -> str:
    """Render the engines list as a human-readable table."""
    headers = ["name", "phases", "required", "budget_tier", "description"]
    rows = [
        [
            m["name"],
            ", ".join(m["phases"]),
            str(m["required"]).lower(),
            m["budget_tier"],
            # Truncate long descriptions to keep table readable.
            m["description"][:60] + ("…" if len(m["description"]) > 60 else ""),
        ]
        for m in manifests
    ]
    return format_table(headers, rows)


def engines_list_json(manifests: list[dict[str, Any]]) -> str:
    return format_json(manifests)


def engine_show_text(manifest: dict[str, Any]) -> str:
    """Render a single engine manifest as key: value pairs."""
    lines: list[str] = []
    for key, val in manifest.items():
        if isinstance(val, (list, tuple)):
            lines.append(f"{key}: {', '.join(str(v) for v in val)}")
        elif isinstance(val, dict):
            lines.append(f"{key}:")
            for k2, v2 in val.items():
                lines.append(f"  {k2}: {', '.join(str(v) for v in v2) if isinstance(v2, list) else v2}")
        else:
            lines.append(f"{key}: {val}")
    return "\n".join(lines) + "\n"


def engine_show_json(manifest: dict[str, Any]) -> str:
    return format_json(manifest)


# ─── Conduct formatters ───────────────────────────────────────────────────────


def conduct_list_table(rules: list[dict[str, Any]]) -> str:
    """Render the conduct list as a human-readable table."""
    headers = ["name", "package", "enforcement", "path"]
    rows = [
        [
            r["name"],
            r["package"],
            r["enforcement"],
            r["path"],
        ]
        for r in rules
    ]
    return format_table(headers, rows)


def conduct_list_json(rules: list[dict[str, Any]]) -> str:
    return format_json(rules)


# ─── Inference formatters ─────────────────────────────────────────────────────


def inference_status_text(s: dict[str, Any]) -> str:
    lines = [
        f"enabled:          {s.get('enabled', False)}",
        f"state_dir:        {s.get('state_dir', '?')}",
        f"last_reconciled:  {s.get('last_reconciled') or 'never'}",
        f"total_artifacts:  {s.get('total_artifacts', 0)}",
        f"total_patterns:   {s.get('total_patterns', 0)}",
    ]
    verdicts = s.get("verdicts", {})
    for k, v in sorted(verdicts.items()):
        lines.append(f"  {k}:  {v}")
    return "\n".join(lines) + "\n"


def inference_status_json(s: dict[str, Any]) -> str:
    return format_json(s)


# ─── Status (aggregate) formatters ───────────────────────────────────────────


def aggregate_status_text(s: dict[str, Any]) -> str:
    lines = [
        f"robit {s.get('version', '?')}",
        "",
        f"engines:          {s.get('engine_count', 0)}",
        f"conduct modules:  {s.get('conduct_count', 0)}",
        "",
        "inference:",
        f"  enabled:        {s.get('inference', {}).get('enabled', False)}",
        f"  last_reconciled:{s.get('inference', {}).get('last_reconciled') or 'never'}",
        f"  total_patterns: {s.get('inference', {}).get('total_patterns', 0)}",
        "",
        "tier defaults:",
    ]
    for k, v in sorted(s.get("tier_defaults", {}).items()):
        lines.append(f"  {k}: {v}")
    fp = s.get("fastpath", {})
    lines.append("")
    lines.append("fast-path bypass:")
    lines.append(f"  enabled:        {fp.get('enabled', False)}")
    lines.append(f"  allowed keys:   {fp.get('allowed_keys', 0)}")
    models = fp.get("allowed_models")
    lines.append(f"  allowed models: {','.join(models) if models else 'any'}")
    return "\n".join(lines) + "\n"


def aggregate_status_json(s: dict[str, Any]) -> str:
    return format_json(s)
