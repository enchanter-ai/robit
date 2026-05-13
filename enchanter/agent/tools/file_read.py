"""enchanter.agent.tools.file_read — read a text file, with line numbers.

The LLM consumes the output of this tool, then refers to the line numbers
when proposing edits via the ``edit`` tool (sibling agent B). The line-number
format here therefore matters: ``{1-indexed-line:>5}\\t{line content}``.

Safety contracts:

* Paths resolve via :func:`enchanter.agent.tools._paths.safe_resolve`, which
  refuses to leave ``ctx.cwd`` unless the caller opts in. ``file_read`` does
  not opt in — reads are always cwd-scoped.
* Binary files are rejected outright: the LLM cannot do anything useful with
  raw bytes in a text channel, and emitting them would burn the context window.
* The file size is checked *before* the read; a file ten times larger than
  ``ctx.max_output_bytes`` is rejected with a clear error instead of being
  read just to be truncated.
"""

from __future__ import annotations

from pathlib import Path

from ._paths import PathOutsideCwdError, safe_resolve
from ._types import ToolContext, ToolResult


# Read this many bytes when sniffing for binary content.
_BINARY_SNIFF_BYTES = 8 * 1024

# A file is "binary" if its sniff sample contains a null byte, or if more
# than this fraction of the sample is non-printable / non-whitespace.
_BINARY_NONPRINTABLE_THRESHOLD = 0.30


def _is_binary(sample: bytes) -> bool:
    """Heuristic binary-file detector.

    Returns True if ``sample`` contains a null byte, or if more than 30% of
    its bytes are non-printable, non-whitespace, *and* non-UTF-8-continuation.
    Bytes >= 0x80 are treated as text candidates because UTF-8-encoded
    files (e.g. ``café``) routinely contain them; the null-byte check still
    catches the typical truly-binary cases.
    """
    if not sample:
        return False
    if b"\x00" in sample:
        return True
    # First try strict UTF-8 decode of a safe prefix; if it decodes cleanly,
    # the file is text. We trim possible split-multibyte tail bytes (up to 3).
    try:
        trim = 0
        while trim < min(4, len(sample)) and (sample[-1 - trim] & 0xC0) == 0x80:
            trim += 1
        sample[: len(sample) - trim].decode("utf-8")
        return False
    except UnicodeDecodeError:
        pass
    # Fallback: printable ASCII + whitespace + high bytes (≥0x80) count as text.
    text_chars = set(range(0x20, 0x7F)) | {0x09, 0x0A, 0x0B, 0x0C, 0x0D}
    nonprintable = sum(1 for b in sample if b < 0x80 and b not in text_chars)
    return (nonprintable / len(sample)) > _BINARY_NONPRINTABLE_THRESHOLD


def _format_relpath(path: Path, cwd: Path) -> str:
    """Return ``path`` relative to ``cwd`` for UI display, or absolute if outside."""
    try:
        return str(path.relative_to(cwd.resolve()))
    except ValueError:
        return str(path)


class FileReadTool:
    """Read a text file's contents, with optional line slicing."""

    name: str = "file_read"
    description: str = (
        "Read the contents of a text file. Returns the file's content as a string. "
        "Supports an optional line range. Binary files are rejected. Use this tool "
        "when you need to inspect existing code before editing it."
    )
    input_schema: dict = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file, absolute or relative to cwd.",
            },
            "start_line": {
                "type": "integer",
                "minimum": 1,
                "description": "Optional 1-indexed line to start from.",
            },
            "end_line": {
                "type": "integer",
                "minimum": 1,
                "description": "Optional 1-indexed line to stop at (inclusive).",
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    }
    requires_approval: bool = False  # Read-only — safe.

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        raw_path = args.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            return ToolResult(
                content="file_read: 'path' arg is required and must be a non-empty string",
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

        # 2. Existence check.
        if not resolved.exists():
            return ToolResult(
                content=f"file not found: {_format_relpath(resolved, ctx.cwd)}",
                is_error=True,
            )

        # 3. Directory check.
        if resolved.is_dir():
            return ToolResult(
                content=(
                    f"path is a directory: {_format_relpath(resolved, ctx.cwd)} "
                    f"— use glob instead"
                ),
                is_error=True,
            )

        # 4. Size sanity — reject files clearly too large to be useful.
        size_cap = ctx.max_output_bytes * 10
        try:
            size = resolved.stat().st_size
        except OSError as exc:
            return ToolResult(
                content=f"could not stat file {_format_relpath(resolved, ctx.cwd)}: {exc}",
                is_error=True,
            )
        if size > size_cap:
            return ToolResult(
                content=(
                    f"file too large to read: {_format_relpath(resolved, ctx.cwd)} "
                    f"is {size} bytes (cap: {size_cap})"
                ),
                is_error=True,
            )

        # 5. Binary sniff.
        try:
            with resolved.open("rb") as fh:
                sniff = fh.read(_BINARY_SNIFF_BYTES)
        except OSError as exc:
            return ToolResult(
                content=f"could not open file {_format_relpath(resolved, ctx.cwd)}: {exc}",
                is_error=True,
            )
        if _is_binary(sniff):
            return ToolResult(
                content=(
                    f"refusing to read binary file: "
                    f"{_format_relpath(resolved, ctx.cwd)}"
                ),
                is_error=True,
            )

        # 6. Read as text. errors="replace" so a stray invalid byte doesn't
        # blow up the read; the binary sniff above already rejected the
        # truly non-text cases.
        try:
            text = resolved.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return ToolResult(
                content=f"could not read file {_format_relpath(resolved, ctx.cwd)}: {exc}",
                is_error=True,
            )

        # 7. Split + slice. splitlines(keepends=True) preserves the trailing
        # newlines so the rendered output looks like the file.
        lines = text.splitlines(keepends=True)
        total = len(lines)

        start_raw = args.get("start_line")
        end_raw = args.get("end_line")
        start = start_raw if isinstance(start_raw, int) and start_raw >= 1 else 1
        end = end_raw if isinstance(end_raw, int) and end_raw >= 1 else total

        # Clamp end to actual file length; allow start to exceed total
        # (returns empty range — handled below).
        end = min(end, total)

        if start > total:
            relpath = _format_relpath(resolved, ctx.cwd)
            return ToolResult(
                content="",
                is_error=False,
                side_effects=(
                    f"read 0 lines from {relpath} "
                    f"(start_line={start} > {total} total)",
                ),
            )

        sliced = lines[start - 1 : end]

        # 8. Render with line numbers.
        rendered_parts: list[str] = []
        for offset, line in enumerate(sliced):
            line_no = start + offset
            rendered_parts.append(f"{line_no:>5}\t{line}")
        rendered = "".join(rendered_parts)

        # 9. Truncate to max_output_bytes if needed.
        truncated = False
        if len(rendered.encode("utf-8")) > ctx.max_output_bytes:
            # Truncate by characters; we may slightly overshoot due to multibyte
            # chars but the loop's own cap will catch any residual.
            cap = ctx.max_output_bytes
            encoded = rendered.encode("utf-8")[:cap]
            # Drop a partial multibyte tail.
            rendered = encoded.decode("utf-8", errors="ignore")
            rendered += "\n...[truncated]"
            truncated = True

        relpath = _format_relpath(resolved, ctx.cwd)
        n_lines = len(sliced)
        side_effect = f"read {n_lines} lines from {relpath}"
        if truncated:
            side_effect += " (truncated)"

        return ToolResult(
            content=rendered,
            is_error=False,
            side_effects=(side_effect,),
        )


__all__ = ["FileReadTool"]
