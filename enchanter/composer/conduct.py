"""enchanter.composer.conduct — assemble per-request conduct XML fragments.

This module is deliberately decoupled from ``enchanter.conduct`` (the loader
being built in parallel).  It operates on plain ``dict`` descriptors so that
the runtime can adapt ``ConductRule`` → dict at the call site once the loader
is ready, with no changes required here.

Expected rule dict shape
------------------------
::

    {
        "name":        str,   # e.g. "discipline"
        "body":        str,   # full Markdown body of the conduct module
        "enforcement": str,   # "prompt" | "code" | "hybrid"
        "package":     str,   # e.g. "core"
        "tags":        tuple[str, ...] | list[str],  # optional metadata
    }

Public API
----------
compose_conduct_xml(rules)
    Filter rules to prompt-conveyed ones, sort them, and produce the
    ``<conduct version="1">…</conduct>`` XML fragment.

select_rules(all_rules, required)
    Pick rules from *all_rules* by name.  Passing ``required=None`` (the
    default) returns all rules unchanged.
"""

from __future__ import annotations

from enchanter.composer.xml import indent_block, xml_escape

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CONDUCT_VERSION = "1"

# Enforcement values that result in prompt injection.
_PROMPT_ENFORCED: frozenset[str] = frozenset({"prompt", "hybrid"})


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def select_rules(
    all_rules: list[dict],
    required: set[str] | None = None,
) -> list[dict]:
    """Return the subset of *all_rules* whose ``name`` is in *required*.

    Parameters
    ----------
    all_rules:
        Full list of rule descriptors.
    required:
        Set of rule names to keep.  ``None`` means "keep all".

    Returns
    -------
    list[dict]
        Filtered list in the same relative order as *all_rules*.
    """
    if required is None:
        return list(all_rules)
    return [r for r in all_rules if r.get("name") in required]


# ---------------------------------------------------------------------------
# Core composer
# ---------------------------------------------------------------------------


def compose_conduct_xml(rules: list[dict]) -> str:
    """Assemble a ``<conduct>`` XML fragment from *rules*.

    Only rules with ``enforcement`` in ``{"prompt", "hybrid"}`` are included.
    The remaining rules are sorted deterministically by ``(package, name)``
    ascending.  XML-special characters in each rule's ``body`` are escaped.

    Parameters
    ----------
    rules:
        List of rule descriptor dicts.  Unrecognised keys are ignored.

    Returns
    -------
    str
        A complete, self-contained XML fragment.  The fragment is valid
        XML and can be parsed by ``xml.etree.ElementTree.fromstring``.

    Output format (illustrative)
    ----------------------------
    ::

        <conduct version="1">
          <module name="discipline" package="core">
            # Discipline — Coding Conduct
            ...
          </module>
        </conduct>

    Indentation
    -----------
    * The ``<conduct>`` open/close tags are at column 0.
    * ``<module>`` open/close tags are indented 2 spaces inside ``<conduct>``.
    * Body lines are indented 4 spaces (2 for ``<conduct>`` + 2 more).
    """
    # 1. Filter to prompt-conveyed rules only.
    included = [
        r for r in rules if r.get("enforcement", "prompt") in _PROMPT_ENFORCED
    ]

    # 2. Sort deterministically: package ascending, then name ascending.
    included.sort(key=lambda r: (r.get("package", ""), r.get("name", "")))

    # 3. Build the XML.
    lines: list[str] = [f'<conduct version="{_CONDUCT_VERSION}">']

    for rule in included:
        name = xml_escape(rule.get("name", ""))
        package = xml_escape(rule.get("package", ""))
        body_raw = rule.get("body", "")
        body_escaped = xml_escape(body_raw)

        # Body is indented 4 spaces (module open tag sits at 2 spaces, and
        # the body is indented 2 more relative to the module tag).
        body_indented = indent_block(body_escaped, spaces=4)

        lines.append(f'  <module name="{name}" package="{package}">')
        lines.append(body_indented)
        lines.append("  </module>")

    lines.append("</conduct>")
    return "\n".join(lines)
