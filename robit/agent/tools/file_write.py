"""robit.agent.tools.file_write — create or replace a file.

Destructive by design: this tool always replaces the target file's contents
(no diff, no merge). Use :mod:`robit.agent.tools.file_edit` for surgical
changes to an existing file.

Safety contracts:

* Paths resolve via :func:`robit.agent.tools._paths.safe_resolve`; writes
  outside ``ctx.cwd`` are refused.
* Parent directory MUST exist. The tool does not create intermediate dirs —
  the LLM should call a dedicated mkdir tool first, or be explicit about the
  expected layout.
* A 10 MiB content cap protects the agent from accidentally materialising a
  generated megalith into the workspace.
* All writes are UTF-8 with LF line endings; if the LLM hands us CRLF, we
  normalise to LF on disk. This is documented in the tool description.
"""

from __future__ import annotations

from pathlib import Path

from ._paths import PathOutsideCwdError, safe_resolve
from ._types import ToolContext, ToolResult


# Refuse to write content larger than this — guards against accidental
# multi-megabyte LLM outputs landing on disk.
_MAX_CONTENT_BYTES = 10 * 1024 * 1024  # 10 MiB


def _format_relpath(path: Path, cwd: Path) -> str:
    """Return ``path`` relative to ``cwd`` for UI display, or absolute if outside."""
    try:
        return str(path.relative_to(cwd.resolve()))
    except ValueError:
        return str(path)


class FileWriteTool:
    """Create or replace a file's contents."""

    name: str = "file_write"
    description: str = (
        "Write content to a file, creating it if missing or replacing if it exists. "
        "Use this for new files or full rewrites. For surgical changes to an existing "
        "file, prefer file_edit. Parent directories must already exist."
    )
    input_schema: dict = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file, absolute or relative to cwd.",
            },
            "content": {
                "type": "string",
                "description": "Full file contents to write.",
            },
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    }
    requires_approval: bool = True  # Destructive — always approve.

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        raw_path = args.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            return ToolResult(
                content="file_write: 'path' arg is required and must be a non-empty string",
                is_error=True,
            )

        content = args.get("content")
        if not isinstance(content, str):
            return ToolResult(
                content="file_write: 'content' arg is required and must be a string",
                is_error=True,
            )

        # 1. Resolve safely.
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

        # 2. Parent must already exist.
        parent = resolved.parent
        if not parent.exists():
            return ToolResult(
                content=(
                    f"parent directory does not exist: "
                    f"{_format_relpath(parent, ctx.cwd)}"
                ),
                is_error=True,
            )
        if not parent.is_dir():
            return ToolResult(
                content=(
                    f"parent path is not a directory: "
                    f"{_format_relpath(parent, ctx.cwd)}"
                ),
                is_error=True,
            )

        # 3. Refuse if the target itself is a directory.
        if resolved.is_dir():
            return ToolResult(
                content=f"path is a directory, refusing to overwrite: {relpath}",
                is_error=True,
            )

        # 4. Size cap on incoming content (measured as UTF-8 bytes).
        encoded = content.encode("utf-8")
        n_bytes = len(encoded)
        if n_bytes > _MAX_CONTENT_BYTES:
            return ToolResult(
                content=(
                    f"content too large: {n_bytes} bytes exceeds cap "
                    f"of {_MAX_CONTENT_BYTES} bytes (10 MiB)"
                ),
                is_error=True,
            )

        # 5. Normalise to LF line endings. The description promises LF on disk;
        # callers handing us CRLF (or mixed) get a clean file regardless.
        normalised = content.replace("\r\n", "\n").replace("\r", "\n")

        existed_before = resolved.exists() and resolved.is_file()

        # 6. Write. Use binary mode + pre-encoded bytes so we control newline
        # translation explicitly — Python's text-mode "newline" handling on
        # Windows would otherwise rewrite "\n" → "\r\n".
        try:
            with resolved.open("wb") as fh:
                fh.write(normalised.encode("utf-8"))
        except OSError as exc:
            return ToolResult(
                content=f"could not write file {relpath}: {exc}",
                is_error=True,
            )

        # Re-measure on-disk byte count (after newline normalisation).
        final_bytes = len(normalised.encode("utf-8"))
        # Line count: number of '\n' + 1 if there is a trailing non-newline
        # tail, else just count of '\n'. Empty file → 0 lines.
        if not normalised:
            n_lines = 0
        else:
            n_lines = normalised.count("\n")
            if not normalised.endswith("\n"):
                n_lines += 1

        verb = "overwrote" if existed_before else "created"
        side_effect = (
            f"{verb} {relpath}: wrote {n_lines} lines / {final_bytes} bytes"
        )

        return ToolResult(
            content=f"wrote {final_bytes} bytes to {relpath}",
            is_error=False,
            side_effects=(side_effect,),
        )


__all__ = ["FileWriteTool"]
