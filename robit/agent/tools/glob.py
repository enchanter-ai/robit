"""robit.agent.tools.glob — find files by pattern.

Mirrors the Claude Code ``glob`` tool: takes a single glob pattern, walks the
agent's ``cwd``, and returns matching file paths sorted by modification time
(most recent first). Read-only — no approval gate.

Safety contracts (parallel to :mod:`robit.agent.tools.file_read`):

* Results stay inside ``ctx.cwd``. Absolute patterns that point outside cwd are
  rejected with ``is_error=True``.
* Standard "noise" directories (``.git``, ``node_modules``, virtual envs,
  build artefacts) are skipped by default — see :data:`_SKIP_DIRS`.
* Compiled artefacts (``*.pyc``, ``*.pyo``) are filtered out too — see
  :data:`_SKIP_PATTERNS`.

Wave 15.2 may render UI hints when a user's pattern matches nothing — the
two module-level constants below are the authoritative skip lists.
"""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path, PurePath

from ._paths import PathOutsideCwdError, safe_resolve
from ._types import ToolContext, ToolResult


# Directory *names* that are skipped at any depth in the walk. Any path with
# one of these as a path component is filtered out before stat'ing.
_SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "dist",
        "build",
        "target",  # Rust
    }
)

# Filename glob patterns to skip outright.
_SKIP_PATTERNS: tuple[str, ...] = (
    "*.pyc",
    "*.pyo",
)


def _is_skipped(path: Path, cwd: Path) -> bool:
    """Return True if ``path`` lies inside a skipped directory or matches a skip pattern."""
    try:
        rel = path.relative_to(cwd)
    except ValueError:
        # Outside cwd — caller handles this separately; treat as skipped here.
        return True
    parts = rel.parts
    for part in parts:
        if part in _SKIP_DIRS:
            return True
    name = path.name
    for pat in _SKIP_PATTERNS:
        if fnmatch(name, pat):
            return True
    return False


class GlobTool:
    """Find files matching a glob pattern within ``ctx.cwd``."""

    name: str = "glob"
    description: str = (
        "Find files matching a glob pattern. Returns matching paths relative to cwd, "
        "sorted by modification time (most recent first). Use this to locate files "
        "by name before reading them. Hidden/build dirs (.git, node_modules, "
        "__pycache__, .venv, dist, build, target) are skipped by default."
    )
    input_schema: dict = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern e.g. '**/*.py' or 'src/*.ts'.",
            },
            "max_results": {
                "type": "integer",
                "minimum": 1,
                "default": 100,
                "description": "Maximum number of paths to return.",
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
                content="glob: 'pattern' arg is required and must be a non-empty string",
                is_error=True,
            )

        max_raw = args.get("max_results", 100)
        if not isinstance(max_raw, int) or max_raw < 1:
            max_results = 100
        else:
            max_results = max_raw

        # If the pattern is absolute, reject outright — globbing outside cwd
        # is not a supported mode.
        if PurePath(raw_pattern).is_absolute():
            return ToolResult(
                content=(
                    f"glob: absolute patterns are not allowed "
                    f"(pattern={raw_pattern!r})"
                ),
                is_error=True,
            )

        cwd_resolved = ctx.cwd.resolve()

        # pathlib's .glob handles both shallow ('*.py') and recursive
        # ('**/*.py') patterns correctly on modern Python (3.11+).
        try:
            raw_matches = list(cwd_resolved.glob(raw_pattern))
        except (ValueError, OSError) as exc:
            return ToolResult(
                content=f"glob: failed to expand pattern {raw_pattern!r}: {exc}",
                is_error=True,
            )

        # Filter: real files, in-cwd, not in skip dirs, not skip-pattern names.
        kept: list[tuple[Path, float]] = []
        for p in raw_matches:
            try:
                if not p.is_file():
                    continue
            except OSError:
                continue
            # Stay inside cwd (symlinks following .resolve()).
            try:
                resolved = p.resolve()
            except OSError:
                continue
            try:
                resolved.relative_to(cwd_resolved)
            except ValueError:
                continue
            if _is_skipped(resolved, cwd_resolved):
                continue
            try:
                mtime = resolved.stat().st_mtime
            except OSError:
                continue
            kept.append((resolved, mtime))

        total = len(kept)
        # Sort by mtime descending (most recent first), tie-break by path so
        # tests on filesystems with mtime granularity issues are still stable.
        kept.sort(key=lambda item: (-item[1], str(item[0])))

        truncated = total > max_results
        shown = kept[:max_results]

        # Render: one relative path per line, POSIX-style for portability.
        lines: list[str] = []
        for path, _mtime in shown:
            rel = path.relative_to(cwd_resolved)
            lines.append(rel.as_posix())
        if truncated:
            lines.append(f"...[truncated to {max_results} of {total} total]")

        content = "\n".join(lines)
        side = f"found {total} file(s) matching {raw_pattern!r}"
        if truncated:
            side += f" (showing {max_results})"

        return ToolResult(
            content=content,
            is_error=False,
            side_effects=(side,),
        )


__all__ = ["GlobTool", "_SKIP_DIRS", "_SKIP_PATTERNS"]
