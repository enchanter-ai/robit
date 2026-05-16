"""Tests for robit.agent.tools.file_edit.FileEditTool."""

from __future__ import annotations

import asyncio
import builtins
from pathlib import Path

from robit.agent.tools._types import ToolContext
from robit.agent.tools.file_edit import FileEditTool


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
    t = FileEditTool()
    assert t.name == "file_edit"
    assert t.requires_approval is True
    assert set(t.input_schema["required"]) == {"path", "old_string", "new_string"}
    assert "replace_all" in t.input_schema["properties"]
    assert t.input_schema["additionalProperties"] is False


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_single_match_replacement(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello world\ngoodbye world\n", encoding="utf-8")
    result = _run(
        FileEditTool().execute(
            {"path": "a.txt", "old_string": "hello", "new_string": "HELLO"},
            _ctx(tmp_path),
        )
    )
    assert result.is_error is False
    assert f.read_text(encoding="utf-8") == "HELLO world\ngoodbye world\n"
    # Diff content includes both --- / +++ headers and a +/- pair.
    assert result.content.startswith("--- a/")
    assert "+++ b/" in result.content
    assert "-hello world" in result.content
    assert "+HELLO world" in result.content
    joined = " ".join(result.side_effects)
    assert "1 replacement" in joined


def test_replace_all_multiple_matches(tmp_path):
    f = tmp_path / "vars.txt"
    f.write_text("foo + foo = 2*foo\n", encoding="utf-8")
    result = _run(
        FileEditTool().execute(
            {
                "path": "vars.txt",
                "old_string": "foo",
                "new_string": "bar",
                "replace_all": True,
            },
            _ctx(tmp_path),
        )
    )
    assert result.is_error is False
    assert f.read_text(encoding="utf-8") == "bar + bar = 2*bar\n"
    joined = " ".join(result.side_effects)
    assert "3 replacements" in joined


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_zero_matches(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello\n", encoding="utf-8")
    result = _run(
        FileEditTool().execute(
            {"path": "a.txt", "old_string": "absent", "new_string": "X"},
            _ctx(tmp_path),
        )
    )
    assert result.is_error is True
    assert "not found" in result.content.lower()
    # File unchanged.
    assert f.read_text(encoding="utf-8") == "hello\n"


def test_multiple_matches_without_replace_all(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("foo foo foo\n", encoding="utf-8")
    result = _run(
        FileEditTool().execute(
            {"path": "a.txt", "old_string": "foo", "new_string": "bar"},
            _ctx(tmp_path),
        )
    )
    assert result.is_error is True
    assert "3 times" in result.content
    assert "replace_all" in result.content.lower()
    # File unchanged.
    assert f.read_text(encoding="utf-8") == "foo foo foo\n"


def test_identical_old_and_new(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello\n", encoding="utf-8")
    result = _run(
        FileEditTool().execute(
            {"path": "a.txt", "old_string": "hello", "new_string": "hello"},
            _ctx(tmp_path),
        )
    )
    assert result.is_error is True
    assert "identical" in result.content.lower()


def test_missing_file(tmp_path):
    result = _run(
        FileEditTool().execute(
            {"path": "nope.txt", "old_string": "a", "new_string": "b"},
            _ctx(tmp_path),
        )
    )
    assert result.is_error is True
    assert "not found" in result.content.lower()


def test_directory_path(tmp_path):
    sub = tmp_path / "subdir"
    sub.mkdir()
    result = _run(
        FileEditTool().execute(
            {"path": "subdir", "old_string": "a", "new_string": "b"},
            _ctx(tmp_path),
        )
    )
    assert result.is_error is True
    assert "directory" in result.content.lower()


def test_path_outside_cwd(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (tmp_path / "outside.txt").write_text("x", encoding="utf-8")
    result = _run(
        FileEditTool().execute(
            {"path": "../outside.txt", "old_string": "x", "new_string": "y"},
            _ctx(work),
        )
    )
    assert result.is_error is True
    assert "outside" in result.content.lower()


def test_atomic_write_failure_preserves_original(tmp_path, monkeypatch):
    """Simulate an OSError during the tmp-file write; original must survive
    intact and no .tmp.* sidecar must remain."""
    f = tmp_path / "a.txt"
    original = "hello world\n"
    f.write_text(original, encoding="utf-8")

    real_open = builtins.open

    def boom(file, mode="r", *args, **kwargs):
        # Trigger only on a write to a .tmp.* sibling of our target.
        path_str = str(file)
        if ".tmp." in path_str and "w" in mode and "b" in mode:
            raise OSError("simulated disk failure")
        return real_open(file, mode, *args, **kwargs)

    # Patch both builtins.open and Path.open since file_edit uses Path.open
    # for the tmp file write.
    real_path_open = Path.open

    def path_open(self, mode="r", *args, **kwargs):
        if ".tmp." in self.name and "w" in mode and "b" in mode:
            raise OSError("simulated disk failure")
        return real_path_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", boom)
    monkeypatch.setattr(Path, "open", path_open)

    result = _run(
        FileEditTool().execute(
            {"path": "a.txt", "old_string": "hello", "new_string": "HELLO"},
            _ctx(tmp_path),
        )
    )

    assert result.is_error is True
    # Original file untouched.
    assert f.read_text(encoding="utf-8") == original
    # No .tmp.* sidecars left behind.
    leftovers = [p for p in tmp_path.iterdir() if ".tmp." in p.name]
    assert leftovers == [], f"tmp leftover: {leftovers}"


def test_missing_args(tmp_path):
    # No path.
    result = _run(
        FileEditTool().execute(
            {"old_string": "a", "new_string": "b"}, _ctx(tmp_path)
        )
    )
    assert result.is_error is True
    # No old_string.
    result = _run(
        FileEditTool().execute(
            {"path": "x.txt", "new_string": "b"}, _ctx(tmp_path)
        )
    )
    assert result.is_error is True


def test_diff_contains_context_lines(tmp_path):
    f = tmp_path / "many.txt"
    body = "a\nb\nc\nTARGET\nd\ne\nf\n"
    f.write_text(body, encoding="utf-8")
    result = _run(
        FileEditTool().execute(
            {"path": "many.txt", "old_string": "TARGET", "new_string": "CHANGED"},
            _ctx(tmp_path),
        )
    )
    assert result.is_error is False
    # Context lines render with a leading space.
    assert " c" in result.content  # context before
    assert " d" in result.content  # context after
    assert "-TARGET" in result.content
    assert "+CHANGED" in result.content
