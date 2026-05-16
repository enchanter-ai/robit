"""Tests for robit.agent.tools.file_write.FileWriteTool."""

from __future__ import annotations

import asyncio
from pathlib import Path

from robit.agent.tools._types import ToolContext
from robit.agent.tools.file_write import FileWriteTool


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
    t = FileWriteTool()
    assert t.name == "file_write"
    assert t.requires_approval is True
    assert "path" in t.input_schema["properties"]
    assert "content" in t.input_schema["properties"]
    assert set(t.input_schema["required"]) == {"path", "content"}
    assert t.input_schema["additionalProperties"] is False


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_write_new_file(tmp_path):
    result = _run(
        FileWriteTool().execute(
            {"path": "hello.txt", "content": "alpha\nbeta\n"},
            _ctx(tmp_path),
        )
    )
    assert result.is_error is False
    f = tmp_path / "hello.txt"
    assert f.read_text(encoding="utf-8") == "alpha\nbeta\n"
    joined = " ".join(result.side_effects)
    assert "created" in joined
    assert "hello.txt" in joined


def test_overwrite_existing_file(tmp_path):
    f = tmp_path / "old.txt"
    f.write_text("old\n", encoding="utf-8")
    result = _run(
        FileWriteTool().execute(
            {"path": "old.txt", "content": "new\n"},
            _ctx(tmp_path),
        )
    )
    assert result.is_error is False
    assert f.read_text(encoding="utf-8") == "new\n"
    joined = " ".join(result.side_effects)
    assert "overwrote" in joined


def test_write_empty_string(tmp_path):
    result = _run(
        FileWriteTool().execute(
            {"path": "empty.txt", "content": ""},
            _ctx(tmp_path),
        )
    )
    assert result.is_error is False
    f = tmp_path / "empty.txt"
    assert f.exists()
    assert f.stat().st_size == 0


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_parent_directory_missing(tmp_path):
    result = _run(
        FileWriteTool().execute(
            {"path": "nope/inside.txt", "content": "x"},
            _ctx(tmp_path),
        )
    )
    assert result.is_error is True
    assert "parent" in result.content.lower()
    assert not (tmp_path / "nope" / "inside.txt").exists()


def test_path_is_a_directory(tmp_path):
    sub = tmp_path / "subdir"
    sub.mkdir()
    result = _run(
        FileWriteTool().execute(
            {"path": "subdir", "content": "x"},
            _ctx(tmp_path),
        )
    )
    assert result.is_error is True
    assert "directory" in result.content.lower()


def test_content_too_large(tmp_path):
    big = "x" * (12 * 1024 * 1024)  # 12 MiB
    result = _run(
        FileWriteTool().execute(
            {"path": "big.txt", "content": big},
            _ctx(tmp_path),
        )
    )
    assert result.is_error is True
    assert "too large" in result.content.lower()
    assert not (tmp_path / "big.txt").exists()


def test_path_outside_cwd(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    result = _run(
        FileWriteTool().execute(
            {"path": "../escape.txt", "content": "x"},
            _ctx(work),
        )
    )
    assert result.is_error is True
    assert "outside" in result.content.lower()
    assert not (tmp_path / "escape.txt").exists()


def test_missing_path_arg(tmp_path):
    result = _run(FileWriteTool().execute({"content": "x"}, _ctx(tmp_path)))
    assert result.is_error is True


def test_missing_content_arg(tmp_path):
    result = _run(FileWriteTool().execute({"path": "x.txt"}, _ctx(tmp_path)))
    assert result.is_error is True


def test_crlf_normalised_to_lf(tmp_path):
    """The tool description promises LF on disk regardless of input."""
    result = _run(
        FileWriteTool().execute(
            {"path": "crlf.txt", "content": "a\r\nb\r\nc\n"},
            _ctx(tmp_path),
        )
    )
    assert result.is_error is False
    raw = (tmp_path / "crlf.txt").read_bytes()
    assert b"\r" not in raw
    assert raw == b"a\nb\nc\n"


def test_utf8_content_round_trip(tmp_path):
    text = "café\nπ ≈ 3.14\n"
    result = _run(
        FileWriteTool().execute(
            {"path": "utf8.txt", "content": text},
            _ctx(tmp_path),
        )
    )
    assert result.is_error is False
    assert (tmp_path / "utf8.txt").read_text(encoding="utf-8") == text
