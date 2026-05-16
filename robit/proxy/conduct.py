"""robit.proxy.conduct — inject enchanter conduct into proxy requests.

The proxy wraps every outbound completion in the enchanter conduct envelope
so that downstream models inherit the same behavioural contract as the
agent itself.  This module is a thin shim over
:func:`robit.conduct.load_conduct` +
:func:`robit.composer.conduct.select_rules` +
:func:`robit.composer.conduct.compose_conduct_xml`.

Usage::

    new_req = apply_conduct_to_request(req)            # default subset
    new_req = apply_conduct_to_request(req, rules)     # custom subset

The injected XML is placed *before* any client-supplied system prompt and
separated by a blank line, so the conduct frame always wins the
attention-budget U-curve top slot.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from robit.composer.conduct import compose_conduct_xml, select_rules
from robit.conduct import load_conduct

from .canonical import CanonicalRequest


# Conservative default — the same five-rule smoke subset used in
# ``scripts/run_one_turn.py``.  Wave 2 may tune this once we have
# token-budget telemetry from the server.
DEFAULT_PROXY_RULES: frozenset[str] = frozenset(
    {
        "discipline",
        "verification",
        "tool-use",
        "refusal-and-recovery",
        "formatting",
    }
)


def apply_conduct_to_request(
    req: CanonicalRequest,
    rules: frozenset[str] | None = None,
) -> CanonicalRequest:
    """Return a copy of *req* with conduct XML prepended to its system prompt.

    Parameters
    ----------
    req:
        The canonical request to wrap.
    rules:
        Subset of conduct rule names to inject.  ``None`` (the default)
        applies :data:`DEFAULT_PROXY_RULES`.  Pass an empty ``frozenset()``
        to opt out of conduct injection entirely (in which case the
        request is returned unchanged).

    Returns
    -------
    CanonicalRequest
        A new dataclass instance — :class:`CanonicalRequest` is frozen so
        we never mutate the input.
    """
    selected = DEFAULT_PROXY_RULES if rules is None else rules

    if not selected:
        # Explicit opt-out — preserve the request verbatim.
        return req

    conduct_xml = _build_conduct_xml(selected)
    if not conduct_xml:
        # No matching rules found — leave the request alone rather than
        # injecting an empty <conduct/> shell.
        return req

    new_system = _prepend_conduct(conduct_xml, req.system)
    return replace(req, system=new_system)


# ---------------------------------------------------------------------------
# Internals.
# ---------------------------------------------------------------------------


def _build_conduct_xml(rule_names: Iterable[str]) -> str:
    """Load all conduct rules, filter to *rule_names*, render the XML."""
    rules = load_conduct()
    rule_dicts = [_rule_to_dict(r) for r in rules]
    selected = select_rules(rule_dicts, required=set(rule_names))
    if not selected:
        return ""
    return compose_conduct_xml(selected)


def _rule_to_dict(rule: object) -> dict:
    """Adapt a :class:`robit.conduct.ConductRule` to the composer's dict shape."""
    return {
        "name": rule.name,  # type: ignore[attr-defined]
        "body": rule.body,  # type: ignore[attr-defined]
        "enforcement": rule.enforcement,  # type: ignore[attr-defined]
        "package": rule.package,  # type: ignore[attr-defined]
        "tags": list(rule.tags),  # type: ignore[attr-defined]
    }


def _prepend_conduct(conduct_xml: str, existing_system: str | None) -> str:
    """Combine conduct XML with any client-supplied system prompt."""
    if not existing_system:
        return conduct_xml
    return f"{conduct_xml}\n\n{existing_system}"


__all__ = [
    "DEFAULT_PROXY_RULES",
    "apply_conduct_to_request",
]
