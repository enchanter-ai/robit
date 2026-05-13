"""Tests for enchanter.agent.tools.file_read.FileReadTool."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from enchanter.agent.tools._types import ToolContext
from enchanter.agent.tools.file_read import FileReadTool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ctx(cwd: Path, *, max_output_bytes: int = 64 * 1024) -> ToolContext:
    return ToolContext(
        cwd=cwd,
        session_id="test-session",
        max_output_bytes=max_output_bytes,
        timeout_s=5.0,
    )


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Static-attribute sanity
# ---------------------------------------------------------------------------


def test_static_attributes():
    t = FileReadTool()
    assert t.name == "file_read"
    assert t.requires_approval is False
    assert "path" in t.input_schema["properties"]
    assert t.input_schema["required"] == ["path"]
    assert t.input_schema["additionalProperties"] is False


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_read_small_text_file(tmp_path):
    f = tmp_path / "hello.txt"
    f.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    result = _run(FileReadTool().execute({"path": "hello.txt"}, _ctx(tmp_path)))

    assert result.is_error is False
    # Line-numbered, 5-wide right-aligned, tab-separated.
    assert result.content == (
        "    1\talpha\n"
        "    2\tbeta\n"
        "    3\tgamma\n"
    )
    assert any("hello.txt" in s for s in result.side_effects)
    assert any("3 lines" in s for s in result.side_effects)


def test_read_with_start_and_end_line(tmp_path):
    f = tmp_path / "many.txt"
    f.write_text("\n".join(f"line{i}" for i in range(1, 11)) + "\n", encoding="utf-8")

    result = _run(
        FileReadTool().execute(
            {"path": "many.txt", "start_line": 3, "end_line": 5},
            _ctx(tmp_path),
        )
    )

    assert result.is_error is False
    assert result.content == (
        "    3\tline3\n"
        "    4\tline4\n"
        "    5\tline5\n"
    )


def test_read_only_start_line(tmp_path):
    f = tmp_path / "many.txt"
    f.write_text("a\nb\nc\nd\n", encoding="utf-8")
    result = _run(
        FileReadTool().execute(
            {"path": "many.txt", "start_line": 3},
            _ctx(tmp_path),
        )
    )
    assert result.is_error is False
    assert result.content == "    3\tc\n    4\td\n"


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_missing_file(tmp_path):
    result = _run(
        FileReadTool().execute({"path": "nope.txt"}, _ctx(tmp_path))
    )
    assert result.is_error is True
    assert "not found" in result.content.lower()


def test_directory_path(tmp_path):
    sub = tmp_path / "subdir"
    sub.mkdir()
    result = _run(
        FileReadTool().execute({"path": "subdir"}, _ctx(tmp_path))
    )
    assert result.is_error is True
    assert "directory" in result.content.lower()
    assert "glob" in result.content.lower()


def test_binary_file_rejected(tmp_path):
    f = tmp_path / "blob.bin"
    f.write_bytes(b"\x00\x01\x02\x03nulls inside\x00\x00\x00\x00")
    result = _run(
        FileReadTool().execute({"path": "blob.bin"}, _ctx(tmp_path))
    )
    assert result.is_error is True
    assert "binary" in result.content.lower()


def test_truncation_marker(tmp_path):
    f = tmp_path / "big.txt"
    # 100 lines of 50 chars each ≈ 5 KB raw — under the 10*1024-byte
    # too-large gate but well over the 1 KB output cap.
    body = "\n".join("x" * 50 for _ in range(100)) + "\n"
    f.write_text(body, encoding="utf-8")

    result = _run(
        FileReadTool().execute(
            {"path": "big.txt"}, _ctx(tmp_path, max_output_bytes=1024)
        )
    )
    assert result.is_error is False
    assert result.content.endswith("...[truncated]")
    assert len(result.content.encode("utf-8")) <= 1024 + len("\n...[truncated]")
    # side-effect should also flag the truncation.
    assert any("truncated" in s for s in result.side_effects)


def test_start_line_beyond_file(tmp_path):
    f = tmp_path / "tiny.txt"
    f.write_text("only-one-line\n", encoding="utf-8")

    result = _run(
        FileReadTool().execute(
            {"path": "tiny.txt", "start_line": 99},
            _ctx(tmp_path),
        )
    )
    assert result.is_error is False
    assert result.content == ""
    joined = " ".join(result.side_effects)
    assert "0 lines" in joined
    assert "99" in joined


def test_path_outside_cwd_rejected(tmp_path):
    # cwd is a child of tmp_path; ../../etc resolves outside it.
    work = tmp_path / "work"
    work.mkdir()
    result = _run(
        FileReadTool().execute({"path": "../../etc/passwd"}, _ctx(work))
    )
    assert result.is_error is True
    assert "outside" in result.content.lower()


def test_absolute_path_outside_cwd_rejected(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    elsewhere = tmp_path / "elsewhere.txt"
    elsewhere.write_text("hi", encoding="utf-8")

    result = _run(
        FileReadTool().execute({"path": str(elsewhere)}, _ctx(work))
    )
    assert result.is_error is True
    assert "outside" in result.content.lower()


def test_empty_path_arg(tmp_path):
    result = _run(FileReadTool().execute({"path": ""}, _ctx(tmp_path)))
    assert result.is_error is True


def test_missing_path_arg(tmp_path):
    result = _run(FileReadTool().execute({}, _ctx(tmp_path)))
    assert result.is_error is True


def test_side_effects_format(tmp_path):
    f = tmp_path / "nested" / "deep.txt"
    f.parent.mkdir()
    f.write_text("one\ntwo\n", encoding="utf-8")
    result = _run(
        FileReadTool().execute({"path": "nested/deep.txt"}, _ctx(tmp_path))
    )
    assert result.is_error is False
    assert len(result.side_effects) == 1
    msg = result.side_effects[0]
    assert "2 lines" in msg
    # Use os.sep-agnostic check.
    assert "deep.txt" in msg
    assert "nested" in msg


def test_too_large_file_rejected(tmp_path):
    f = tmp_path / "huge.txt"
    # 11 KB; cap is 1 KB → size cap is 10 KB → rejected.
    f.write_text("x" * 11_000, encoding="utf-8")
    result = _run(
        FileReadTool().execute(
            {"path": "huge.txt"}, _ctx(tmp_path, max_output_bytes=1024)
        )
    )
    assert result.is_error is True
    assert "too large" in result.content.lower()


def test_utf8_content_preserved(tmp_path):
    f = tmp_path / "utf8.txt"
    f.write_text("café\nnaïve\nπ ≈ 3.14\n", encoding="utf-8")
    result = _run(FileReadTool().execute({"path": "utf8.txt"}, _ctx(tmp_path)))
    assert result.is_error is False
    assert "café" in result.content
    assert "π" in result.content
