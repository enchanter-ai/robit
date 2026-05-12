"""enchanter.composer — system-prompt fragment assembly.

Public surface
--------------
compose_conduct_xml(rules: list[dict]) -> str
    Assemble a ``<conduct version="1">`` XML fragment from a list of conduct
    rule descriptors.  Only rules with ``enforcement`` in
    ``{"prompt", "hybrid"}`` are included.  Rules are sorted deterministically
    by ``(package, name)`` ascending.

select_rules(all_rules: list[dict], required: set[str] | None = None) -> list[dict]
    Filter *all_rules* to only those whose ``name`` is in *required*.
    Passing ``required=None`` returns all rules unchanged.
"""

from enchanter.composer.conduct import compose_conduct_xml, select_rules

__all__ = ["compose_conduct_xml", "select_rules"]
