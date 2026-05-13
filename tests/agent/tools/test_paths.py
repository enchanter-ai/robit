"""Tests for enchanter.agent.tools._paths.safe_resolve."""

from __future__ import annotations

import os
import sys

import pytest

from enchanter.agent.tools._paths import PathOutsideCwdError, safe_resolve


def test_relative_path_inside_cwd_resolves_to_absolute(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    f = sub / "x.txt"
    f.write_text("hi", encoding="utf-8")

    resolved = safe_resolve(tmp_path, "sub/x.txt")

    assert resolved.is_absolute()
    assert resolved == f.resolve()


def test_absolute_path_inside_cwd_is_returned(tmp_path):
    f = tmp_path / "y.txt"
    f.write_text("yo", encoding="utf-8")

    resolved = safe_resolve(tmp_path, str(f))

    assert resolved == f.resolve()


def test_dot_dot_traversal_raises(tmp_path):
    # tmp_path / "child" / "../../etc" → tmp_path/.. on resolve → outside cwd.
    child = tmp_path / "child"
    child.mkdir()
    with pytest.raises(PathOutsideCwdError):
        safe_resolve(child, "../../etc")


def test_allow_outside_cwd_true_returns_resolved(tmp_path):
    child = tmp_path / "child"
    child.mkdir()
    resolved = safe_resolve(child, "../../etc", allow_outside_cwd=True)
    assert resolved.is_absolute()
    # No exception, and it does NOT live inside `child`.
    assert not str(resolved).startswith(str(child.resolve()))


@pytest.mark.skipif(
    sys.platform == "win32" and not os.environ.get("CI_SYMLINKS_OK"),
    reason="Symlink creation on Windows requires elevated perms or developer mode.",
)
def test_symlink_pointing_outside_cwd_raises(tmp_path):
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside_target = outside_dir / "secret.txt"
    outside_target.write_text("classified", encoding="utf-8")

    work = tmp_path / "work"
    work.mkdir()
    link = work / "link_to_secret"
    try:
        link.symlink_to(outside_target)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlink not supported in this environment: {exc}")

    with pytest.raises(PathOutsideCwdError):
        safe_resolve(work, "link_to_secret")


def test_bad_type_raises_typeerror(tmp_path):
    with pytest.raises(TypeError):
        safe_resolve(tmp_path, 123)  # type: ignore[arg-type]


def test_path_object_user_path_works(tmp_path):
    f = tmp_path / "z.txt"
    f.write_text("z", encoding="utf-8")
    from pathlib import Path

    resolved = safe_resolve(tmp_path, Path("z.txt"))
    assert resolved == f.resolve()
