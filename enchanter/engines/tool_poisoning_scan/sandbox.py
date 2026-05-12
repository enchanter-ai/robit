"""SandboxConfirmation — static-analysis portion of the M5 sandbox (v0 port).

The TS implementation (lich/sandbox.ts) forks a child_process worker and
runs a resource-bounded code-review inside it.  That subprocess execution
is explicitly OUT OF SCOPE for v0 (see task spec).

What we port here is the pure-logic, static-analysis path: given a tool
schema dict we apply the same pattern scan and emit a structured verdict.
This is the "interval propagation / abstract interpretation" analogue that
the TS sandbox would run inside the worker — we just run it inline since
there is no subprocess boundary in the Python v0.

When the TS m5_sandbox is enabled:
  - If M1 already vetoed: sandbox is skipped (veto is terminal).
  - Otherwise: the sandbox result is advisory; failures set degraded=True
    but do not override the M1 ack/veto decision.

The Python SandboxConfirmation mirrors this: confirm() returns a
SandboxVerdict.  The adapter calls it only when M1 produced a warn-level
ack (suspicion below veto threshold) and the caller has opted in.

Deviation from TS: no subprocess / no worker IPC.  confirm() runs
synchronously (no fork, no timeout).  This is documented as DEVIATION-1.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from .patterns import SUSPICION_PATTERNS, VETO_THRESHOLD


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

SandboxStatus = Literal["clean", "warn", "veto", "error"]


@dataclass(frozen=True)
class SandboxVerdict:
    """Result of a sandbox confirmation pass.

    status: 'clean' | 'warn' | 'veto' | 'error'
    pattern_ids: patterns that fired (empty for clean / error)
    suspicion_score: sum of matched severities
    detail: human-readable detail string (None for clean)
    """

    status: SandboxStatus
    pattern_ids: tuple[str, ...]
    suspicion_score: float
    detail: str | None


# ---------------------------------------------------------------------------
# SandboxConfirmation
# ---------------------------------------------------------------------------

class SandboxConfirmation:
    """Applies the static scan to a tool schema in-process (v0 — no subprocess).

    The confirm() method:
      1. Serialises the schema to a canonical text form (same fields as M1 scanSchema).
      2. Runs each SUSPICION_PATTERN against each text corpus.
      3. Aggregates the suspicion score and returns a SandboxVerdict.

    This is intentionally identical logic to the adapter's M1 scan so that the
    sandbox provides a second, independent confirmation rather than a different
    algorithm.  In a future v1 it would run in an isolated subprocess with a
    resource budget — that is the portion not ported here.
    """

    def confirm(self, tool_schema: dict[str, object]) -> SandboxVerdict:
        """Run a static sandbox analysis on *tool_schema*.

        Returns SandboxVerdict with:
          - 'clean'  if no patterns match
          - 'warn'   if total score < VETO_THRESHOLD
          - 'veto'   if total score >= VETO_THRESHOLD
          - 'error'  if the schema cannot be analysed (malformed input)
        """
        try:
            corpora = _extract_corpora(tool_schema)
        except Exception as exc:  # noqa: BLE001
            return SandboxVerdict(
                status="error",
                pattern_ids=(),
                suspicion_score=0.0,
                detail=f"sandbox-schema-parse-error: {exc}",
            )

        seen_ids: set[str] = set()
        matched_ids: list[str] = []
        total_score: float = 0.0

        for pattern in SUSPICION_PATTERNS:
            if pattern.id in seen_ids:
                continue
            if any(pattern.match.search(corpus) for corpus in corpora):
                seen_ids.add(pattern.id)
                matched_ids.append(pattern.id)
                total_score += pattern.severity  # v0: no FP downweight in sandbox

        if not matched_ids:
            return SandboxVerdict(
                status="clean",
                pattern_ids=(),
                suspicion_score=0.0,
                detail=None,
            )

        if total_score >= VETO_THRESHOLD:
            status: SandboxStatus = "veto"
        else:
            status = "warn"

        return SandboxVerdict(
            status=status,
            pattern_ids=tuple(matched_ids),
            suspicion_score=total_score,
            detail=f"sandbox-static-scan:{','.join(matched_ids)}",
        )


# ---------------------------------------------------------------------------
# Schema text extraction (mirrors scanSchema in lich.adapter.ts)
# ---------------------------------------------------------------------------

def _extract_corpora(schema: dict[str, object]) -> list[str]:
    """Return a list of text strings extracted from the tool schema for pattern matching.

    Covers the same fields the TS scanSchema covers:
      - description
      - parameters / inputSchema.properties (each param description)
      - errorTemplates
      - name / displayName
    """
    texts: list[str] = []

    # Top-level description.
    desc = schema.get("description")
    if isinstance(desc, str):
        texts.append(desc)

    # Parameter descriptions — two conventions (parameters dict or inputSchema.properties).
    props: dict[str, object] = {}
    if isinstance(schema.get("parameters"), dict):
        props = schema["parameters"]  # type: ignore[assignment]
    elif isinstance(schema.get("inputSchema"), dict):
        input_schema = schema["inputSchema"]
        if isinstance(input_schema, dict) and isinstance(input_schema.get("properties"), dict):
            props = input_schema["properties"]  # type: ignore[assignment]

    for _key, val in props.items():
        if isinstance(val, dict) and isinstance(val.get("description"), str):
            texts.append(val["description"])  # type: ignore[arg-type]
        elif isinstance(val, str):
            texts.append(val)

    # Error templates.
    err_templates = schema.get("errorTemplates")
    if err_templates is not None:
        if isinstance(err_templates, str):
            texts.append(err_templates)
        else:
            texts.append(json.dumps(err_templates, default=str))

    # Name fields (hidden unicode check).
    for name_field in ("name", "displayName"):
        val = schema.get(name_field)
        if isinstance(val, str):
            texts.append(val)

    return texts
