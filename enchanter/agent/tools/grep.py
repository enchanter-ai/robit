"""enchanter.agent.tools.grep — search file contents by regex.

Companion to :mod:`enchanter.agent.tools.glob` — once the LLM has found files
by name, ``grep`` lets it find symbols/functions/text inside them. Read-only,
no approval gate.

Honest performance note: this is a straightforward Python implementation, not
a ripgrep replacement. For very large repos (>100k files), narrow ``path`` to
a subdirectory before searching.

Safety contracts:

* The optional ``path`` arg resolves via
  :func:`enchanter.agent.tools._paths.safe_resolve` — searches never escape
  ``ctx.cwd``.
* Binary files are skipped (same heuristic as :mod:`file_read`).
* The skip-dirs / skip-patterns lists are imported from :mod:`glob` so the two
  tools agree on what counts as "noise".
"""

from __future__ import annotations

import re
from fnmatch import fnmatch
from pathlib import Path

from ._paths import PathOutsideCwdError, safe_resolve
from ._types import ToolContext, ToolResult
from .glob import _SKIP_DIRS, _SKIP_PATTERNS


# Same constants as :mod:`file_read`. Kept local rather than re-imported to
# avoid coupling the two modules; the values are intentionally identical.
_BINARY_SNIFF_BYTES = 8 * 1024
_BINARY_NONPRINTABLE_THRESHOLD = 0.30


def _is_binary(sample: bytes) -> bool:
    """Heuristic binary-file detector — mirrors :func:`file_read._is_binary`."""
    if not sample:
        return False
    if b"\x00" in sample:
        return True
    try:
        trim = 0
        while trim < min(4, len(sample)) and (sample[-1 - trim] & 0xC0) == 0x80:
            trim += 1
        sample[: len(sample) - trim].decode("utf-8")
        return False
    except UnicodeDecodeError:
        pass
    text_chars = set(range(0x20, 0x7F)) | {0x09, 0x0A, 0x0B, 0x0C, 0x0D}
    nonprintable = sum(1 for b in sample if b < 0x80 and b not in text_chars)
    return (nonprintable / len(sample)) > _BINARY_NONPRINTABLE_THRESHOLD


def _is_skipped_dir(name: str) -> bool:
    return name in _SKIP_DIRS


def _matches_skip_pattern(name: str) -> bool:
    return any(fnmatch(name, p) for p in _SKIP_PATTERNS)


def _iter_search_files(root: Path, glob_filter: str | None):
    """Yield candidate files under ``root`` respecting skip rules + glob filter.

    Walks directories breadth-first via :meth:`Path.iterdir`. Skipped directories
    are pruned (we don't descend into them).
    """
    if root.is_file():
        if glob_filter and not fnmatch(root.name, glob_filter):
            return
        if _matches_skip_pattern(root.name):
            return
        yield root
        return

    stack: list[Path] = [root]
    while stack:
        current = stack.pop()
        try:
            entries = list(current.iterdir())
        except (OSError, PermissionError):
            continue
        for entry in entries:
            try:
                if entry.is_dir():
                    if _is_skipped_dir(entry.name):
                        continue
                    stack.append(entry)
                elif entry.is_file():
                    if _matches_skip_pattern(entry.name):
                        continue
                    if glob_filter and not fnmatch(entry.name, glob_filter):
                        continue
                    yield entry
            except OSError:
                continue


class GrepTool:
    """Search file contents with a regex, returning matches with file paths and line numbers."""

    name: str = "grep"
    description: str = (
        "Search file contents using a regex pattern (Python re syntax). Returns "
        "matching lines as 'relpath:line_no: content'. Use this to find symbols, "
        "functions, or text across the codebase. For very large repos, narrow "
        "`path` to a subdirectory."
    )
    input_schema: dict = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regex pattern. Python re syntax.",
            },
            "path": {
                "type": "string",
                "description": "File or directory to search. Defaults to cwd.",
            },
            "glob": {
                "type": "string",
                "description": "Optional file-glob filter applied during recursion.",
            },
            "case_insensitive": {
                "type": "boolean",
                "default": False,
            },
            "max_results": {
                "type": "integer",
                "minimum": 1,
                "default": 200,
            },
            "context_lines": {
                "type": "integer",
                "minimum": 0,
                "default": 0,
                "description": "Lines of context around each match.",
            },
        },
        "required": ["pattern"],
        "additionalProperties": False,
    }
    requires_approval: bool = False  # Read-only.

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        raw_pattern = args.get("pattern")
        if not isinstance(raw_pattern, str) or not raw_pattern:
            return ToolResult(
                content="grep: 'pattern' arg is required and must be a non-empty string",
                is_error=True,
            )

        flags = 0
        if bool(args.get("case_insensitive", False)):
            flags |= re.IGNORECASE
        try:
            regex = re.compile(raw_pattern, flags)
        except re.error as exc:
            return ToolResult(
                content=f"grep: invalid regex {raw_pattern!r}: {exc}",
                is_error=True,
            )

        max_raw = args.get("max_results", 200)
        max_results = max_raw if isinstance(max_raw, int) and max_raw >= 1 else 200

        ctx_raw = args.get("context_lines", 0)
        context_lines = ctx_raw if isinstance(ctx_raw, int) and ctx_raw >= 0 else 0

        glob_filter_raw = args.get("glob")
        glob_filter = glob_filter_raw if isinstance(glob_filter_raw, str) and glob_filter_raw else None

        # Resolve the search root.
        raw_path = args.get("path")
        if isinstance(raw_path, str) and raw_path:
            try:
                root = safe_resolve(ctx.cwd, raw_path)
            except PathOutsideCwdError as exc:
                return ToolResult(
                    content=f"path is outside the allowed working directory: {exc}",
                    is_error=True,
                )
            except (TypeError, OSError) as exc:
                return ToolResult(
                    content=f"could not resolve path {raw_path!r}: {exc}",
                    is_error=True,
                )
            if not root.exists():
                return ToolResult(
                    content=f"grep: path not found: {raw_path}",
                    is_error=True,
                )
        else:
            root = ctx.cwd.resolve()

        cwd_resolved = ctx.cwd.resolve()

        matches: list[str] = []
        files_with_matches: set[str] = set()
        truncated = False

        for file_path in _iter_search_files(root, glob_filter):
            if truncated:
                break
            try:
                with file_path.open("rb") as fh:
                    sniff = fh.read(_BINARY_SNIFF_BYTES)
            except OSError:
                continue
            if _is_binary(sniff):
                continue

            try:
                text = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            lines = text.splitlines()
            # Find matching line indices.
            match_indices = [i for i, line in enumerate(lines) if regex.search(line)]
            if not match_indices:
                continue

            try:
                rel = file_path.resolve().relative_to(cwd_resolved).as_posix()
            except ValueError:
                rel = str(file_path)
            files_with_matches.add(rel)

            if context_lines == 0:
                for idx in match_indices:
                    matches.append(f"{rel}:{idx + 1}: {lines[idx]}")
                    if len(matches) >= max_results:
                        truncated = True
                        break
            else:
                # Emit each match with ±context_lines around it. We don't merge
                # overlapping windows — keep the implementation simple, the LLM
                # can dedupe visually.
                seen_match_keys: set[tuple[str, int]] = set()
                for idx in match_indices:
                    lo = max(0, idx - context_lines)
                    hi = min(len(lines) - 1, idx + context_lines)
                    for j in range(lo, hi + 1):
                        key = (rel, j)
                        if key in seen_match_keys:
                            continue
                        seen_match_keys.add(key)
                        sep = ":" if j == idx or regex.search(lines[j]) else "-"
                        matches.append(f"{rel}:{j + 1}{sep} {lines[j]}")
                    if len(matches) >= max_results:
                        truncated = True
                        break

        # Hard cap, in case context lines pushed us over.
        if len(matches) > max_results:
            matches = matches[:max_results]
            truncated = True

        content_parts = matches[:]
        if truncated:
            content_parts.append(f"...[truncated to {max_results} match lines]")
        content = "\n".join(content_parts)

        # Count "real" matches (not context lines) for the side-effect summary.
        # With context_lines=0 every emitted line is a match; with context > 0
        # we count lines ending with ':' separator between line_no and body.
        # Simpler: report `files_with_matches` size and emitted-line count.
        match_count = sum(1 for m in matches if _is_match_line(m))
        side = f"found {match_count} match(es) in {len(files_with_matches)} file(s)"
        if truncated:
            side += " (truncated)"

        return ToolResult(
            content=content,
            is_error=False,
            side_effects=(side,),
        )


def _is_match_line(line: str) -> bool:
    """Detect 'relpath:line_no: body' (match) vs 'relpath:line_no- body' (context).

    The split happens at the SECOND colon (the first colon separates the path
    from the line number — paths on Windows can contain colons, but our
    relative paths from cwd never do because we normalise to POSIX).
    """
    # Match lines look like `path:N: ...`; context lines look like `path:N- ...`.
    # Find first ':' after a digit-run.
    i = 0
    n = len(line)
    # skip path (up to first ':')
    while i < n and line[i] != ":":
        i += 1
    if i >= n:
        return False
    i += 1  # past first ':'
    # skip digits (line number)
    while i < n and line[i].isdigit():
        i += 1
    if i >= n:
        return False
    return line[i] == ":"


__all__ = ["GrepTool"]
