"""Unified verdict type — port of `src/bus/verdict.ts`.

Before this type existed the codebase had five different ways to express a
"veto": a ``PluginAck(status="veto")`` carrying a free-form ``reason`` string,
a ``SecurityVetoError`` carrying ``(plugin, phase, reason)``, a derived
``*.veto`` bus event whose payload held ``pattern_id`` / ``pattern_name``, a
``VetoResult`` dataclass in the proxy, and HTTP/JSON-RPC renderings that
re-parsed the reason string to recover the pattern id.

``Verdict`` is the single structured representation. Engines that veto attach
one to their ack (``PluginAck.verdict``); the orchestrator wraps it into a
``SecurityVetoError``; the proxy renders it without ever string-slicing the
reason again.
"""

from __future__ import annotations

from dataclasses import dataclass

from .context import LifecyclePhase


@dataclass(frozen=True)
class Verdict:
    """A structured veto decision.

    Attributes
    ----------
    plugin:
        Name of the engine that issued the verdict.
    phase:
        Lifecycle phase the verdict fired in.
    reason:
        Human-readable reason string (kept for backwards-compat / logging).
        Historically shaped ``"<plugin>:<pattern_id>"``.
    pattern_id:
        Structured pattern identifier (e.g. ``"DOG-014"``).  ``None`` when the
        issuing engine carries no pattern taxonomy.
    pattern_name:
        Human-readable pattern name.  ``None`` when unknown.
    severity:
        Severity classification.  Defaults to ``"veto"`` for a hard block.
    """

    plugin: str
    phase: LifecyclePhase
    reason: str
    pattern_id: str | None = None
    pattern_name: str | None = None
    severity: str = "veto"

    @classmethod
    def from_reason(
        cls,
        *,
        plugin: str,
        phase: LifecyclePhase,
        reason: str | None,
        severity: str = "veto",
    ) -> "Verdict":
        """Build a Verdict from a legacy free-form reason string.

        This is the *single* place the historical ``"<plugin>:<pattern_id>"``
        reason convention is parsed.  Callers (the proxy veto path, HTTP/JSON-RPC
        renderers) must read the structured fields rather than re-slicing the
        string themselves.

        Degrades gracefully: a reason that doesn't fit the convention yields a
        Verdict with ``pattern_id=None`` rather than raising.
        """
        text = reason or "veto"
        pattern_id: str | None = None
        if ":" in text:
            # Reason shape: "<plugin>:<pattern_id>" or "<plugin>:<id> (advisory)".
            _, _, rest = text.partition(":")
            token = rest.strip().split(" ", 1)[0]
            if token:
                pattern_id = token
        return cls(
            plugin=plugin,
            phase=phase,
            reason=text,
            pattern_id=pattern_id,
            pattern_name=None,
            severity=severity,
        )

    def to_header_value(self) -> str:
        """Render a compact, header-safe single-line summary.

        Used by the HTTP layer for ``X-Enchanter-Veto`` and by any JSON-RPC
        error renderer.  Contains only pattern identifiers and the plugin/phase
        — never raw request content — so it is safe to surface verbatim.
        """
        parts = [f"plugin={self.plugin}", f"phase={self.phase}"]
        if self.pattern_id:
            parts.append(f"pattern_id={self.pattern_id}")
        if self.pattern_name:
            parts.append(f"pattern_name={self.pattern_name}")
        parts.append(f"severity={self.severity}")
        return "; ".join(parts)


def render_veto_http(verdict: Verdict) -> tuple[int, dict[str, str], dict[str, object]]:
    """Render a :class:`Verdict` to an HTTP 451 response triple.

    Returns ``(status_code, headers, body)`` where ``status_code`` is always
    451 (RFC 7725 "Unavailable For Legal Reasons", the project's chosen veto
    status), ``headers`` carries the structured ``X-Enchanter-Veto`` summary,
    and ``body`` is a JSON-serialisable dict built from the Verdict's
    structured fields — no string-slicing of the reason.

    Degrades gracefully: a Verdict with missing pattern data still produces a
    valid 451 (the optional fields are simply omitted from the header).
    """
    headers = {"X-Enchanter-Veto": verdict.to_header_value()}
    body: dict[str, object] = {
        "error": "vetoed",
        "plugin": verdict.plugin,
        "phase": str(verdict.phase),
        "reason": verdict.reason,
        "severity": verdict.severity,
    }
    if verdict.pattern_id is not None:
        body["pattern_id"] = verdict.pattern_id
    if verdict.pattern_name is not None:
        body["pattern_name"] = verdict.pattern_name
    return 451, headers, body


__all__ = ["Verdict", "render_veto_http"]
