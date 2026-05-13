"""enchanter.agent.tools.file_edit — surgical string replacement in a file.

Companion to :mod:`enchanter.agent.tools.file_write`. Where ``file_write``
overwrites the whole file, ``file_edit`` performs a precise find-and-replace
on an exact substring and refuses to act if the match is ambiguous.

Safety contracts:

* Paths resolve via :func:`enchanter.agent.tools._paths.safe_resolve`; edits
  outside ``ctx.cwd`` are refused.
* Default mode (``replace_all=False``) demands a single, unique match. Multiple
  matches must be resolved by the caller supplying more context — never by
  guessing. Renames across a file use ``replace_all=True`` explicitly.
* Writes are atomic: contents land in a sibling ``.tmp.<pid>.<ts>`` file,
  ``fsync`` is called, then :func:`os.replace` swaps the new file into place.
  :func:`os.replace` is the cross-platform atomic rename — on Windows the
  default :meth:`pathlib.Path.rename` fails when the target exists, so we use
  :func:`os.replace` which overwrites atomically on POSIX and Windows alike.
* The returned ``content`` is a compact unified-diff-style summary so a future
  diff renderer (Wave 15.2G) can consume it directly.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from ._paths import PathOutsideCwdError, safe_resolve
from ._types import ToolContext, ToolResult


# Lines of context to render on either side of a changed hunk in the diff
# summary returned via ``ToolResult.content``.
_DIFF_CONTEXT_LINES = 3


def _format_relpath(path: Path, cwd: Path) -> str:
    """Return ``path`` relative to ``cwd`` for UI display, or absolute if outside."""
    try:
        return str(path.relative_to(cwd.resolve()))
    except ValueError:
        return str(path)


def _render_diff(before: str, after: str, relpath: str) -> str:
    """Render a compact unified-diff-ish summary of the change.

    Format:

        --- a/<relpath>
        +++ b/<relpath>
        @@ line <n> @@
         context-before
        -removed
        +added
         context-after

    This is not a strictly RFC-compliant unified diff (no exact hunk header
    line counts) — it is a human-readable rendering that the Wave 15.2G diff
    renderer can parse on a line-prefix basis (`-` / `+` / ` `).
    """
    before_lines = before.splitlines()
    after_lines = after.splitlines()

    # Find the first and last differing line indices via a simple
    # common-prefix / common-suffix scan. Good enough for surgical edits where
    # the change is contiguous; for multi-region edits the user runs
    # file_edit multiple times.
    n_before = len(before_lines)
    n_after = len(after_lines)
    common_prefix = 0
    while (
        common_prefix < n_before
        and common_prefix < n_after
        and before_lines[common_prefix] == after_lines[common_prefix]
    ):
        common_prefix += 1

    common_suffix = 0
    while (
        common_suffix < (n_before - common_prefix)
        and common_suffix < (n_after - common_prefix)
        and before_lines[n_before - 1 - common_suffix]
        == after_lines[n_after - 1 - common_suffix]
    ):
        common_suffix += 1

    before_changed = before_lines[common_prefix : n_before - common_suffix]
    after_changed = after_lines[common_prefix : n_after - common_suffix]

    ctx_start = max(0, common_prefix - _DIFF_CONTEXT_LINES)
    ctx_before = before_lines[ctx_start:common_prefix]
    ctx_after = before_lines[
        n_before - common_suffix : n_before - common_suffix + _DIFF_CONTEXT_LINES
    ]

    parts: list[str] = [
        f"--- a/{relpath}",
        f"+++ b/{relpath}",
        f"@@ line {common_prefix + 1} @@",
    ]
    parts.extend(f" {line}" for line in ctx_before)
    parts.extend(f"-{line}" for line in before_changed)
    parts.extend(f"+{line}" for line in after_changed)
    parts.extend(f" {line}" for line in ctx_after)
    return "\n".join(parts) + "\n"


def _atomic_write(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` via tmp-file + fsync + atomic replace.

    Raises whatever :func:`open`/:meth:`write`/:meth:`os.replace` raise;
    cleans up the tmp file on failure.
    """
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with tmp.open("wb") as fh:
            fh.write(data)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                # fsync may fail on some platforms (e.g. /tmp on certain CI
                # images); the rename below still gives durability + atomicity
                # within the directory.
                pass
        # os.replace is the cross-platform atomic rename. On Windows
        # Path.rename fails if the target exists; os.replace overwrites.
        os.replace(tmp, path)
    except BaseException:
        # Clean up the tmp file on any failure (including KeyboardInterrupt).
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        raise


class FileEditTool:
    """Surgical exact-string replacement in an existing text file."""

    name: str = "file_edit"
    description: str = (
        "Replace an exact string in a file with new content. The `old_string` must "
        "appear exactly once in the file — if it appears zero or more than once, "
        "the edit fails. Use this for precise, surgical changes. Set replace_all=true "
        "to replace every occurrence (e.g. renaming a variable)."
    )
    input_schema: dict = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file, absolute or relative to cwd.",
            },
            "old_string": {
                "type": "string",
                "description": "Exact string to find.",
            },
            "new_string": {
                "type": "string",
                "description": "Replacement string.",
            },
            "replace_all": {
                "type": "boolean",
                "default": False,
                "description": "Replace every occurrence instead of demanding a unique match.",
            },
        },
        "required": ["path", "old_string", "new_string"],
        "additionalProperties": False,
    }
    requires_approval: bool = True  # Mutates the file — approve.

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        raw_path = args.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            return ToolResult(
                content="file_edit: 'path' arg is required and must be a non-empty string",
                is_error=True,
            )

        old_string = args.get("old_string")
        new_string = args.get("new_string")
        if not isinstance(old_string, str):
            return ToolResult(
                content="file_edit: 'old_string' is required and must be a string",
                is_error=True,
            )
        if not isinstance(new_string, str):
            return ToolResult(
                content="file_edit: 'new_string' is required and must be a string",
                is_error=True,
            )

        replace_all = args.get("replace_all", False)
        if not isinstance(replace_all, bool):
            return ToolResult(
                content="file_edit: 'replace_all' must be a boolean",
                is_error=True,
            )

        # 1. Resolve.
        try:
            resolved = safe_resolve(ctx.cwd, raw_path)
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

        relpath = _format_relpath(resolved, ctx.cwd)

        # 2. Must exist and be a regular file.
        if not resolved.exists():
            return ToolResult(
                content=f"file not found: {relpath}",
                is_error=True,
            )
        if resolved.is_dir():
            return ToolResult(
                content=f"path is a directory, refusing to edit: {relpath}",
                is_error=True,
            )

        # 3. Identical old/new is a no-op — flag rather than silently succeed,
        # so the LLM notices the bug.
        if old_string == new_string:
            return ToolResult(
                content="old_string and new_string are identical — no change requested",
                is_error=True,
            )

        # 4. Read the file. errors="replace" mirrors file_read's tolerance.
        try:
            original = resolved.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return ToolResult(
                content=f"could not read file {relpath}: {exc}",
                is_error=True,
            )

        # 5. Match policy.
        if replace_all:
            if old_string not in original:
                return ToolResult(
                    content=f"old_string not found in {relpath}",
                    is_error=True,
                )
            count = original.count(old_string)
            new_content = original.replace(old_string, new_string)
        else:
            count = original.count(old_string)
            if count == 0:
                return ToolResult(
                    content=f"old_string not found in {relpath}",
                    is_error=True,
                )
            if count > 1:
                return ToolResult(
                    content=(
                        f"old_string appears {count} times in {relpath}; "
                        f"supply more context or set replace_all=true"
                    ),
                    is_error=True,
                )
            new_content = original.replace(old_string, new_string, 1)

        # 6. Atomic write — preserve original on any failure.
        try:
            _atomic_write(resolved, new_content.encode("utf-8"))
        except OSError as exc:
            return ToolResult(
                content=f"could not write file {relpath}: {exc}",
                is_error=True,
            )

        # 7. Diff summary for the LLM / future renderer.
        diff = _render_diff(original, new_content, relpath)

        plural = "replacement" if count == 1 else "replacements"
        side_effect = f"edited {relpath}: {count} {plural}"

        return ToolResult(
            content=diff,
            is_error=False,
            side_effects=(side_effect,),
        )


__all__ = ["FileEditTool"]
