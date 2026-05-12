"""Minimal stdlib-only YAML frontmatter parser for conduct Markdown files.

Handles the small YAML subset used in conduct frontmatter:
  - ``key: value``  (string, int, bool)
  - ``key: [item, item]``  (inline list)
  - ``key:``  followed by ``  - item`` lines  (block list)
  - Lines starting with ``#`` are comments and are ignored.

Does NOT depend on PyYAML or any third-party library.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from enchanter.conduct.types import ConductFrontmatterError

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_frontmatter(text: str, *, path: Path | None = None) -> tuple[dict[str, Any], str]:
    """Split YAML frontmatter from Markdown body.

    The opening ``---`` MUST be the very first line of *text* (no leading
    blank lines).  The closing ``---`` is the next standalone ``---`` line.
    Everything between the delimiters is treated as YAML; everything after
    is the body.

    Returns:
        A ``(meta, body)`` tuple.  *meta* is the parsed frontmatter dict;
        *body* is the Markdown content without the frontmatter block.

    Raises:
        :class:`~enchanter.conduct.types.ConductFrontmatterError`: when the
            file starts with ``---`` but the closing delimiter is missing, or
            when the YAML content cannot be parsed.
    """
    _path: Path = path or Path("<unknown>")

    lines = text.splitlines(keepends=True)

    if not lines or lines[0].rstrip("\r\n") != "---":
        # No frontmatter — return the whole text unchanged.
        return {}, text

    # Find the closing ---
    close_idx: int | None = None
    for i, line in enumerate(lines[1:], start=1):
        if line.rstrip("\r\n") == "---":
            close_idx = i
            break

    if close_idx is None:
        raise ConductFrontmatterError(
            _path, "opening '---' found but no closing '---' delimiter"
        )

    yaml_lines = lines[1:close_idx]
    body = "".join(lines[close_idx + 1 :])

    meta = _parse_yaml_subset(yaml_lines, _path)
    return meta, body


# ---------------------------------------------------------------------------
# Internal minimal YAML parser
# ---------------------------------------------------------------------------


def _parse_yaml_subset(
    lines: list[str], path: Path
) -> dict[str, Any]:
    """Parse a small subset of YAML from a list of lines.

    Supports:
    - ``key: scalar``
    - ``key: [inline, list]``
    - Block lists::

          key:
            - item1
            - item2

    Comments (lines starting with ``#``) are ignored.
    Raises :class:`ConductFrontmatterError` on anything else.
    """
    meta: dict[str, Any] = {}
    # Work through stripped lines; track current block-list key.
    current_list_key: str | None = None

    for raw_line in lines:
        line = raw_line.rstrip("\r\n")

        # Blank lines are fine; they may separate sections.
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            current_list_key = None  # blank line ends a block list context
            continue

        # Block list item?  Must be indented + start with "- "
        if line.startswith((" ", "\t")) and stripped.startswith("- "):
            if current_list_key is None:
                raise ConductFrontmatterError(
                    path,
                    f"unexpected list item without a preceding key: {line!r}",
                )
            meta[current_list_key].append(_coerce(stripped[2:].strip()))
            continue

        # Reset block list context once we encounter a non-item line.
        current_list_key = None

        # Must be a ``key: value`` line.
        if ":" not in stripped:
            raise ConductFrontmatterError(
                path, f"cannot parse YAML line (no colon): {line!r}"
            )

        key, _, rest = stripped.partition(":")
        key = key.strip()
        rest = rest.strip()

        if not key:
            raise ConductFrontmatterError(path, f"empty key in line: {line!r}")

        if rest == "":
            # Block list start: value is absent → expect indented "- " lines.
            meta[key] = []
            current_list_key = key
        elif rest.startswith("[") and rest.endswith("]"):
            # Inline list: ``[item, item, ...]``
            inner = rest[1:-1]
            if inner.strip() == "":
                meta[key] = []
            else:
                meta[key] = [_coerce(item.strip()) for item in inner.split(",")]
        else:
            meta[key] = _coerce(rest)

    return meta


def _coerce(value: str) -> Any:
    """Convert a string scalar to the most natural Python type.

    Handles ``true``/``false`` → bool, integer strings → int,
    and everything else stays str.  Quoted strings have quotes stripped.
    """
    # Strip surrounding quotes (single or double).
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]

    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False

    try:
        return int(value)
    except ValueError:
        pass

    return value
