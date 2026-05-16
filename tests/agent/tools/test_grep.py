"""Tests for robit.agent.tools.grep.GrepTool."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from robit.agent.tools._types import ToolContext
from robit.agent.tools.grep import GrepTool


def _ctx(cwd: Path) -> ToolContext:
    return ToolContext(
        cwd=cwd,
        session_id="test-session",
        max_output_bytes=64 * 1024,
        timeout_s=5.0,
    )


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Static-attribute sanity
# ---------------------------------------------------------------------------


def test_static_attributes():
    t = GrepTool()
    assert t.name == "grep"
    assert t.requires_approval is False
    assert "pattern" in t.input_schema["properties"]
    assert t.input_schema["required"] == ["pattern"]
    assert t.input_schema["additionalProperties"] is False


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_grep_finds_simple_match(tmp_path):
    (tmp_path / "a.py").write_text("alpha\nbeta needle here\ngamma\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("nothing here\n", encoding="utf-8")

    result = _run(GrepTool().execute({"pattern": "needle"}, _ctx(tmp_path)))

    assert result.is_error is False
    assert "a.py:2: beta needle here" in result.content
    assert "b.py" not in result.content
    assert any("1 match(es)" in s for s in result.side_effects)
    assert any("1 file(s)" in s for s in result.side_effects)


def test_grep_case_insensitive(tmp_path):
    (tmp_path / "a.py").write_text("HELLO World\nhello there\n", encoding="utf-8")
    result = _run(
        GrepTool().execute(
            {"pattern": "hello", "case_insensitive": True}, _ctx(tmp_path)
        )
    )
    assert result.is_error is False
    assert "a.py:1: HELLO World" in result.content
    assert "a.py:2: hello there" in result.content


def test_grep_bad_regex_returns_error(tmp_path):
    (tmp_path / "a.py").write_text("x", encoding="utf-8")
    result = _run(GrepTool().execute({"pattern": "[unterminated"}, _ctx(tmp_path)))
    assert result.is_error is True
    assert "invalid regex" in result.content.lower()


def test_grep_glob_filter(tmp_path):
    (tmp_path / "keep.py").write_text("match here\n", encoding="utf-8")
    (tmp_path / "skip.txt").write_text("match here\n", encoding="utf-8")

    result = _run(
        GrepTool().execute(
            {"pattern": "match", "glob": "*.py"}, _ctx(tmp_path)
        )
    )

    assert result.is_error is False
    assert "keep.py:1: match here" in result.content
    assert "skip.txt" not in result.content


def test_grep_context_lines_separators(tmp_path):
    body = "before2\nbefore1\nMATCH line\nafter1\nafter2\n"
    (tmp_path / "a.py").write_text(body, encoding="utf-8")

    result = _run(
        GrepTool().execute(
            {"pattern": "MATCH", "context_lines": 2}, _ctx(tmp_path)
        )
    )

    assert result.is_error is False
    lines = result.content.splitlines()
    # Expect 5 lines: 2 before (with `-`), match (with `:`), 2 after (with `-`).
    assert "a.py:1- before2" in lines
    assert "a.py:2- before1" in lines
    assert "a.py:3: MATCH line" in lines
    assert "a.py:4- after1" in lines
    assert "a.py:5- after2" in lines


def test_grep_skips_binary_files(tmp_path):
    (tmp_path / "real.py").write_text("needle\n", encoding="utf-8")
    (tmp_path / "blob.bin").write_bytes(b"needle\x00\x00more\x00bytes")

    result = _run(GrepTool().execute({"pattern": "needle"}, _ctx(tmp_path)))

    assert result.is_error is False
    assert "real.py:1: needle" in result.content
    assert "blob.bin" not in result.content


def test_grep_max_results_truncates(tmp_path):
    # 10 lines, all matching.
    body = "\n".join(f"hit line {i}" for i in range(10)) + "\n"
    (tmp_path / "a.py").write_text(body, encoding="utf-8")

    result = _run(
        GrepTool().execute(
            {"pattern": "hit", "max_results": 3}, _ctx(tmp_path)
        )
    )

    assert result.is_error is False
    assert "truncated" in result.content.lower()
    # 3 match lines + 1 truncation marker.
    lines = result.content.splitlines()
    assert len(lines) == 4


def test_grep_single_file_path(tmp_path):
    (tmp_path / "search.py").write_text("hit\n", encoding="utf-8")
    (tmp_path / "other.py").write_text("hit\n", encoding="utf-8")

    result = _run(
        GrepTool().execute(
            {"pattern": "hit", "path": "search.py"}, _ctx(tmp_path)
        )
    )

    assert result.is_error is False
    assert "search.py:1: hit" in result.content
    assert "other.py" not in result.content


def test_grep_path_outside_cwd_rejected(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    elsewhere = tmp_path / "elsewhere.txt"
    elsewhere.write_text("hit\n", encoding="utf-8")

    result = _run(
        GrepTool().execute(
            {"pattern": "hit", "path": str(elsewhere)}, _ctx(work)
        )
    )
    assert result.is_error is True
    assert "outside" in result.content.lower()


def test_grep_skips_noise_dirs(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "hidden.py").write_text("needle\n", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "nope.py").write_text("needle\n", encoding="utf-8")
    (tmp_path / "real.py").write_text("needle\n", encoding="utf-8")

    result = _run(GrepTool().execute({"pattern": "needle"}, _ctx(tmp_path)))

    assert result.is_error is False
    assert "real.py:1: needle" in result.content
    assert ".git" not in result.content
    assert "node_modules" not in result.content


def test_grep_zero_matches(tmp_path):
    (tmp_path / "a.py").write_text("nothing\n", encoding="utf-8")
    result = _run(GrepTool().execute({"pattern": "needle"}, _ctx(tmp_path)))
    assert result.is_error is False
    assert result.content == ""
    assert any("0 match(es)" in s for s in result.side_effects)


def test_grep_missing_pattern(tmp_path):
    result = _run(GrepTool().execute({}, _ctx(tmp_path)))
    assert result.is_error is True
