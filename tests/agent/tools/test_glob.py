"""Tests for enchanter.agent.tools.glob.GlobTool."""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import pytest

from enchanter.agent.tools._types import ToolContext
from enchanter.agent.tools.glob import GlobTool, _SKIP_DIRS, _SKIP_PATTERNS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    t = GlobTool()
    assert t.name == "glob"
    assert t.requires_approval is False
    assert "pattern" in t.input_schema["properties"]
    assert t.input_schema["required"] == ["pattern"]
    assert t.input_schema["additionalProperties"] is False


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_glob_matches_files_in_cwd(tmp_path):
    (tmp_path / "a.py").write_text("a", encoding="utf-8")
    (tmp_path / "b.py").write_text("b", encoding="utf-8")
    (tmp_path / "c.txt").write_text("c", encoding="utf-8")

    result = _run(GlobTool().execute({"pattern": "*.py"}, _ctx(tmp_path)))

    assert result.is_error is False
    lines = set(result.content.splitlines())
    assert lines == {"a.py", "b.py"}
    assert any("2 file(s)" in s for s in result.side_effects)


def test_glob_recursive_double_star(tmp_path):
    (tmp_path / "top.py").write_text("x", encoding="utf-8")
    nested = tmp_path / "pkg" / "sub"
    nested.mkdir(parents=True)
    (nested / "deep.py").write_text("y", encoding="utf-8")

    result = _run(GlobTool().execute({"pattern": "**/*.py"}, _ctx(tmp_path)))

    assert result.is_error is False
    lines = set(result.content.splitlines())
    assert "top.py" in lines
    assert "pkg/sub/deep.py" in lines


def test_glob_skips_noise_dirs(tmp_path):
    # Files in skipped dirs should not be returned.
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD.py").write_text("hi", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "evil.py").write_text("hi", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "stale.py").write_text("hi", encoding="utf-8")
    (tmp_path / "keep.py").write_text("real", encoding="utf-8")

    result = _run(GlobTool().execute({"pattern": "**/*.py"}, _ctx(tmp_path)))

    assert result.is_error is False
    lines = set(result.content.splitlines())
    assert lines == {"keep.py"}


def test_glob_max_results_truncates(tmp_path):
    for i in range(5):
        (tmp_path / f"f{i}.py").write_text(str(i), encoding="utf-8")

    result = _run(
        GlobTool().execute({"pattern": "*.py", "max_results": 2}, _ctx(tmp_path))
    )

    assert result.is_error is False
    lines = result.content.splitlines()
    # 2 path lines + 1 truncation marker.
    assert len(lines) == 3
    assert "truncated" in lines[-1]
    # Side-effect notes the total (5) not the shown count.
    assert any("5 file(s)" in s for s in result.side_effects)


def test_glob_zero_matches_empty_content(tmp_path):
    (tmp_path / "only.txt").write_text("x", encoding="utf-8")
    result = _run(GlobTool().execute({"pattern": "*.py"}, _ctx(tmp_path)))

    assert result.is_error is False
    assert result.content == ""
    assert any("0 file(s)" in s for s in result.side_effects)


def test_glob_sorted_by_mtime_desc(tmp_path):
    older = tmp_path / "older.py"
    newer = tmp_path / "newer.py"
    older.write_text("o", encoding="utf-8")
    # Set explicit mtimes to be filesystem-granularity-independent.
    now = time.time()
    os.utime(older, (now - 100, now - 100))
    newer.write_text("n", encoding="utf-8")
    os.utime(newer, (now, now))

    result = _run(GlobTool().execute({"pattern": "*.py"}, _ctx(tmp_path)))

    assert result.is_error is False
    lines = result.content.splitlines()
    assert lines == ["newer.py", "older.py"]


def test_glob_absolute_pattern_rejected(tmp_path):
    # Use a platform-appropriate absolute glob pattern.
    abs_pattern = (tmp_path / "*.py").as_posix()
    # Make it absolute even on Windows; tmp_path is already absolute.
    assert Path(abs_pattern).is_absolute()

    result = _run(GlobTool().execute({"pattern": abs_pattern}, _ctx(tmp_path)))

    assert result.is_error is True
    assert "absolute" in result.content.lower()


# ---------------------------------------------------------------------------
# Extra sanity
# ---------------------------------------------------------------------------


def test_skip_constants_exposed():
    # Wave 15.2 will read these to render UI hints; lock the surface.
    assert ".git" in _SKIP_DIRS
    assert "node_modules" in _SKIP_DIRS
    assert "__pycache__" in _SKIP_DIRS
    assert "*.pyc" in _SKIP_PATTERNS


def test_glob_skips_pyc_files(tmp_path):
    (tmp_path / "keep.py").write_text("k", encoding="utf-8")
    (tmp_path / "junk.pyc").write_bytes(b"\x00\x01")
    result = _run(GlobTool().execute({"pattern": "*"}, _ctx(tmp_path)))
    assert result.is_error is False
    lines = set(result.content.splitlines())
    assert lines == {"keep.py"}


def test_glob_missing_pattern(tmp_path):
    result = _run(GlobTool().execute({}, _ctx(tmp_path)))
    assert result.is_error is True


def test_glob_only_matches_files_not_directories(tmp_path):
    (tmp_path / "afile.py").write_text("x", encoding="utf-8")
    (tmp_path / "adir").mkdir()
    result = _run(GlobTool().execute({"pattern": "a*"}, _ctx(tmp_path)))
    assert result.is_error is False
    lines = set(result.content.splitlines())
    assert lines == {"afile.py"}
