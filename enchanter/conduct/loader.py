"""Conduct loader — reads Markdown conduct modules and returns ConductRule objects.

Usage::

    from enchanter.conduct.loader import load_conduct

    rules = load_conduct()          # reads from vis default
    rules = load_conduct(root=tmp)  # for tests or alternate roots

The loader walks ``<root>/packages/*/conduct/*.md``, parses optional YAML
frontmatter, and returns one :class:`~enchanter.conduct.types.ConductRule`
per file.  Files without frontmatter get sensible defaults (name from stem,
enforcement="prompt", no tags).

Stdlib only — no PyYAML, no third-party dependencies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from enchanter.conduct._paths import DEFAULT_VIS_ROOT
from enchanter.conduct.frontmatter import parse_frontmatter
from enchanter.conduct.types import ConductFrontmatterError, ConductRule, EnforcementMode

# The glob pattern for conduct files relative to the vis_root root.
_CONDUCT_GLOB = "packages/*/conduct/*.md"

_VALID_ENFORCEMENT: frozenset[str] = frozenset({"code", "prompt", "hybrid"})


def load_conduct(root: Path | None = None) -> list[ConductRule]:
    """Load all conduct modules from *root*.

    Args:
        root: Root directory of an ``vis``-layout tree.
              Defaults to :data:`~enchanter.conduct._paths.DEFAULT_VIS_ROOT`.
              The function searches ``<root>/packages/*/conduct/*.md``.

    Returns:
        A list of :class:`~enchanter.conduct.types.ConductRule` objects,
        one per ``.md`` file found.  The list is sorted by ``(package, name)``
        for deterministic ordering.

    Raises:
        :class:`~enchanter.conduct.types.ConductFrontmatterError`: propagated
            from :func:`~enchanter.conduct.frontmatter.parse_frontmatter` when
            a file has a malformed frontmatter block.
    """
    vis_root = root if root is not None else DEFAULT_VIS_ROOT
    vis_root = Path(vis_root)

    rules: list[ConductRule] = []

    for md_path in sorted(vis_root.glob(_CONDUCT_GLOB)):
        rule = _load_file(md_path, vis_root)
        rules.append(rule)

    rules.sort(key=lambda r: (r.package, r.name))
    return rules


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_file(path: Path, vis_root: Path) -> ConductRule:
    """Parse a single conduct Markdown file into a ConductRule."""
    text = path.read_text(encoding="utf-8")
    meta, body = parse_frontmatter(text, path=path)

    name = _resolve_name(meta, path)
    enforcement = _resolve_enforcement(meta, path)
    tags = _resolve_tags(meta)
    package = _resolve_package(path, vis_root)

    return ConductRule(
        name=name,
        path=path,
        body=body,
        enforcement=enforcement,
        package=package,
        tags=tags,
    )


def _resolve_name(meta: dict[str, Any], path: Path) -> str:
    raw = meta.get("name")
    if raw is not None:
        return str(raw).strip()
    return path.stem


def _resolve_enforcement(meta: dict[str, Any], path: Path) -> EnforcementMode:
    raw = meta.get("enforcement")
    if raw is None:
        return "prompt"
    value = str(raw).strip().lower()
    if value not in _VALID_ENFORCEMENT:
        raise ConductFrontmatterError(
            path,
            f"invalid enforcement value {raw!r}; expected one of {sorted(_VALID_ENFORCEMENT)}",
        )
    return value  # type: ignore[return-value]


def _resolve_tags(meta: dict[str, Any]) -> tuple[str, ...]:
    raw = meta.get("tags")
    if raw is None:
        return ()
    if isinstance(raw, list):
        return tuple(str(t).strip() for t in raw)
    # Scalar string — treat as single tag.
    return (str(raw).strip(),)


def _resolve_package(path: Path, vis_root: Path) -> str:
    """Extract the package name from the path.

    Expected layout: ``<vis_root>/packages/<package>/conduct/<name>.md``
    Returns the ``<package>`` segment, or an empty string if the path does
    not match the expected depth.
    """
    try:
        rel = path.relative_to(vis_root)
    except ValueError:
        return ""

    parts = rel.parts  # e.g. ("packages", "core", "conduct", "discipline.md")
    if len(parts) >= 2 and parts[0] == "packages":
        return parts[1]
    return ""
