"""Tests for ``robit._env`` — the stdlib .env autoloader."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

from robit._env import (
    _default_user_dir,
    load_env_files,
    parse_env_file,
)


# ─── parse_env_file ──────────────────────────────────────────────────────────


def test_parse_basic_key_value() -> None:
    pairs = parse_env_file("FOO=bar\nBAZ=qux\n")
    assert pairs == [("FOO", "bar"), ("BAZ", "qux")]


def test_parse_double_and_single_quoted() -> None:
    text = (
        'GREETING="hello world"\n'
        "LITERAL='single quoted'\n"
        'KEEP_HASH="value # inside quotes"\n'
    )
    pairs = parse_env_file(text)
    assert pairs == [
        ("GREETING", "hello world"),
        ("LITERAL", "single quoted"),
        ("KEEP_HASH", "value # inside quotes"),
    ]


def test_parse_double_quote_escape_sequences() -> None:
    # Real backslash followed by n/t/etc. in the file content.
    text = 'ESCAPED="line1\\nline2\\ttab\\\\back\\"quote"\n'
    pairs = parse_env_file(text)
    assert pairs == [("ESCAPED", 'line1\nline2\ttab\\back"quote')]


def test_parse_export_prefix_tolerated() -> None:
    pairs = parse_env_file("export FOO=bar\nexport BAR=baz\n")
    assert pairs == [("FOO", "bar"), ("BAR", "baz")]


def test_parse_comments_and_inline_comment() -> None:
    text = (
        "# top-level comment\n"
        "\n"
        "FOO=bar # trailing comment\n"
        'QUOTED="keep # inside"\n'
        "# another comment\n"
    )
    pairs = parse_env_file(text)
    assert pairs == [
        ("FOO", "bar"),
        ("QUOTED", "keep # inside"),
    ]


def test_parse_invalid_lines_skipped_and_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    text = (
        "=novalue\n"
        "NOEQUALS line\n"
        "GOOD=value\n"
        "1BAD=startsWithDigit\n"
        "ALSO_GOOD=ok\n"
    )
    with caplog.at_level(logging.WARNING, logger="robit._env"):
        pairs = parse_env_file(text)
    assert pairs == [("GOOD", "value"), ("ALSO_GOOD", "ok")]
    # Three invalid lines should produce three warnings.
    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warning_records) >= 3


def test_parse_empty_value_valid() -> None:
    pairs = parse_env_file("EMPTY=\nALSO_EMPTY=   \n")
    assert pairs == [("EMPTY", ""), ("ALSO_EMPTY", "")]


def test_parse_no_interpolation() -> None:
    # $VAR should be preserved literally — no expansion.
    pairs = parse_env_file("LITERAL=$HOME/path\n")
    assert pairs == [("LITERAL", "$HOME/path")]


def test_parse_duplicate_keys_preserved_in_order() -> None:
    pairs = parse_env_file("K=1\nK=2\nK=3\n")
    assert pairs == [("K", "1"), ("K", "2"), ("K", "3")]


# ─── load_env_files ──────────────────────────────────────────────────────────


def test_load_shell_env_wins_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WIXIE17_SHELL_WINS", "from_shell")
    (tmp_path / ".env").write_text("WIXIE17_SHELL_WINS=from_envfile\n")
    user_dir = tmp_path / "user"
    user_dir.mkdir()

    applied = load_env_files(cwd=tmp_path, user_dir=user_dir)

    assert "WIXIE17_SHELL_WINS" not in applied
    assert os.environ["WIXIE17_SHELL_WINS"] == "from_shell"


def test_load_override_true_envfile_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WIXIE17_OVERRIDE", "from_shell")
    (tmp_path / ".env").write_text("WIXIE17_OVERRIDE=from_envfile\n")
    user_dir = tmp_path / "user"
    user_dir.mkdir()

    applied = load_env_files(cwd=tmp_path, user_dir=user_dir, override=True)

    assert applied.get("WIXIE17_OVERRIDE") == "from_envfile"
    assert os.environ["WIXIE17_OVERRIDE"] == "from_envfile"


def test_load_cwd_wins_over_user_dir_on_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("WIXIE17_COLLISION", raising=False)

    cwd_dir = tmp_path / "cwd"
    user_dir = tmp_path / "user"
    cwd_dir.mkdir()
    user_dir.mkdir()
    (cwd_dir / ".env").write_text("WIXIE17_COLLISION=from_cwd\n")
    (user_dir / ".env").write_text("WIXIE17_COLLISION=from_user\n")

    applied = load_env_files(cwd=cwd_dir, user_dir=user_dir)

    assert applied["WIXIE17_COLLISION"] == "from_cwd"
    assert os.environ["WIXIE17_COLLISION"] == "from_cwd"
    # Cleanup so other tests don't see it.
    monkeypatch.delenv("WIXIE17_COLLISION", raising=False)


def test_load_user_dir_only_applied_when_no_cwd_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("WIXIE17_USER_ONLY", raising=False)
    cwd_dir = tmp_path / "cwd"
    user_dir = tmp_path / "user"
    cwd_dir.mkdir()
    user_dir.mkdir()
    (user_dir / ".env").write_text("WIXIE17_USER_ONLY=from_user\n")

    applied = load_env_files(cwd=cwd_dir, user_dir=user_dir)

    assert applied["WIXIE17_USER_ONLY"] == "from_user"
    assert os.environ["WIXIE17_USER_ONLY"] == "from_user"
    monkeypatch.delenv("WIXIE17_USER_ONLY", raising=False)


def test_load_missing_files_silent_no_op(tmp_path: Path) -> None:
    # Neither cwd/.env nor user_dir/.env exists.
    cwd_dir = tmp_path / "cwd"
    user_dir = tmp_path / "user"
    cwd_dir.mkdir()
    user_dir.mkdir()

    applied = load_env_files(cwd=cwd_dir, user_dir=user_dir)

    assert applied == {}


def test_load_within_file_last_key_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("WIXIE17_LAST_WINS", raising=False)
    (tmp_path / ".env").write_text(
        "WIXIE17_LAST_WINS=first\nWIXIE17_LAST_WINS=second\nWIXIE17_LAST_WINS=third\n"
    )
    user_dir = tmp_path / "user"
    user_dir.mkdir()

    applied = load_env_files(cwd=tmp_path, user_dir=user_dir)

    assert applied["WIXIE17_LAST_WINS"] == "third"
    monkeypatch.delenv("WIXIE17_LAST_WINS", raising=False)


def test_default_user_dir_windows_appdata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(os, "name", "nt")
    fake_appdata = tmp_path / "AppData" / "Roaming"
    monkeypatch.setenv("APPDATA", str(fake_appdata))
    monkeypatch.delenv("ROBIT_HOME", raising=False)
    monkeypatch.delenv("ENCHANTER_HOME", raising=False)

    resolved = _default_user_dir()

    # New default — neither legacy nor new dir exists, so the uncreated
    # new path (%APPDATA%\robit) wins.
    assert resolved == fake_appdata / "robit"


def test_default_user_dir_posix_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(os, "name", "posix")
    monkeypatch.delenv("ROBIT_HOME", raising=False)
    monkeypatch.delenv("ENCHANTER_HOME", raising=False)
    # Path.home() is consulted by both robit._env and robit._compat. Stub
    # both module-local imports so neither resolves a real home dir.
    monkeypatch.setattr("robit._env.Path.home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr("robit._compat.Path.home", classmethod(lambda cls: tmp_path))

    resolved = _default_user_dir()

    # New default — neither ~/.robit nor ~/.enchanter exists in tmp, so
    # the uncreated new path wins.
    assert resolved == tmp_path / ".robit"


def test_default_user_dir_robit_home_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    custom = tmp_path / "custom-robit"
    monkeypatch.setenv("ROBIT_HOME", str(custom))

    resolved = _default_user_dir()

    assert resolved == custom


def test_load_quoted_values_applied_to_environ(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("WIXIE17_QUOTED", raising=False)
    (tmp_path / ".env").write_text('WIXIE17_QUOTED="hello world"\n')
    user_dir = tmp_path / "user"
    user_dir.mkdir()

    applied = load_env_files(cwd=tmp_path, user_dir=user_dir)

    assert applied["WIXIE17_QUOTED"] == "hello world"
    assert os.environ["WIXIE17_QUOTED"] == "hello world"
    monkeypatch.delenv("WIXIE17_QUOTED", raising=False)
