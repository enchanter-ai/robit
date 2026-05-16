"""Tests for the enchanter → robit compatibility shim.

Covers:
- ``get_env`` reads the canonical ``ROBIT_*`` name when set.
- ``get_env`` falls back to the legacy ``ENCHANTER_*`` name with a one-shot
  WARNING.
- Canonical wins over legacy with no warning.
- ``resolve_user_dir`` prefers ``ROBIT_HOME``, then ``~/.robit``, then falls
  back to ``~/.enchanter`` with a migration WARNING.
- The deprecation warning fires once per process even on repeated reads.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

from robit import _compat


@pytest.fixture(autouse=True)
def reset_warning_state(monkeypatch):
    """Clear the one-shot warning dedup set between tests so each test starts
    with a fresh warning state. Also strip all relevant env vars so a stray
    shell export in the developer's environment doesn't poison the tests."""
    monkeypatch.setattr(_compat, "_warned_env", set())
    for canonical, legacy in _compat.LEGACY_ENV_MAP.items():
        monkeypatch.delenv(canonical, raising=False)
        monkeypatch.delenv(legacy, raising=False)
    yield


# ---------------------------------------------------------------------------
# get_env
# ---------------------------------------------------------------------------


def test_get_env_canonical_only(monkeypatch):
    monkeypatch.setenv("ROBIT_HOME", "/canonical")
    assert _compat.get_env("ROBIT_HOME") == "/canonical"


def test_get_env_legacy_only_emits_warning(monkeypatch, caplog):
    monkeypatch.setenv("ENCHANTER_HOME", "/legacy")
    with caplog.at_level(logging.WARNING, logger="robit.compat"):
        value = _compat.get_env("ROBIT_HOME")
    assert value == "/legacy"
    # Exactly one WARNING with both names mentioned.
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    msg = warnings[0].getMessage()
    assert "ENCHANTER_HOME" in msg and "ROBIT_HOME" in msg


def test_get_env_canonical_wins_no_warning(monkeypatch, caplog):
    monkeypatch.setenv("ROBIT_HOME", "/canonical")
    monkeypatch.setenv("ENCHANTER_HOME", "/legacy")
    with caplog.at_level(logging.WARNING, logger="robit.compat"):
        value = _compat.get_env("ROBIT_HOME")
    assert value == "/canonical"
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


def test_get_env_default_when_neither_set():
    assert _compat.get_env("ROBIT_HOME", default="fallback") == "fallback"
    assert _compat.get_env("ROBIT_HOME") is None


def test_get_env_warning_fires_once_per_process(monkeypatch, caplog):
    """Two reads of the same legacy var produce exactly one WARNING."""
    monkeypatch.setenv("ENCHANTER_AGENT_MOCK", "1")
    with caplog.at_level(logging.WARNING, logger="robit.compat"):
        _compat.get_env("ROBIT_AGENT_MOCK")
        _compat.get_env("ROBIT_AGENT_MOCK")
        _compat.get_env("ROBIT_AGENT_MOCK")
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, f"expected one warning, got {len(warnings)}"


def test_get_env_unknown_name_no_legacy_lookup(monkeypatch):
    """An env name with no entry in LEGACY_ENV_MAP returns default cleanly."""
    monkeypatch.delenv("NOT_A_ROBIT_VAR", raising=False)
    assert _compat.get_env("NOT_A_ROBIT_VAR", default="ok") == "ok"


# ---------------------------------------------------------------------------
# resolve_user_dir
# ---------------------------------------------------------------------------


def test_resolve_user_dir_robit_home_override_wins(monkeypatch, tmp_path):
    custom = tmp_path / "custom"
    monkeypatch.setenv("ROBIT_HOME", str(custom))
    assert _compat.resolve_user_dir() == custom


def test_resolve_user_dir_legacy_home_override_with_warning(
    monkeypatch, tmp_path, caplog
):
    """ENCHANTER_HOME still works via get_env() and triggers the warning."""
    custom = tmp_path / "legacy-home"
    monkeypatch.setenv("ENCHANTER_HOME", str(custom))
    with caplog.at_level(logging.WARNING, logger="robit.compat"):
        result = _compat.resolve_user_dir()
    assert result == custom
    # The env-var warning fires.
    assert any(
        "ENCHANTER_HOME" in r.getMessage() and "ROBIT_HOME" in r.getMessage()
        for r in caplog.records
    )


def test_resolve_user_dir_prefers_new_dir_when_it_exists(
    monkeypatch, tmp_path, caplog
):
    monkeypatch.setattr(_compat.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(os, "name", "posix")
    (tmp_path / ".robit").mkdir()
    (tmp_path / ".enchanter").mkdir()
    with caplog.at_level(logging.WARNING, logger="robit.compat"):
        result = _compat.resolve_user_dir()
    assert result == tmp_path / ".robit"
    # No migration WARNING when the new dir exists.
    assert not [r for r in caplog.records if "legacy path" in r.getMessage()]


def test_resolve_user_dir_falls_back_to_legacy_with_warning(
    monkeypatch, tmp_path, caplog
):
    monkeypatch.setattr(_compat.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(os, "name", "posix")
    # Only the legacy dir exists.
    (tmp_path / ".enchanter").mkdir()
    with caplog.at_level(logging.WARNING, logger="robit.compat"):
        result = _compat.resolve_user_dir()
    assert result == tmp_path / ".enchanter"
    # Migration WARNING fires with the actionable mv command shape.
    migration = [r for r in caplog.records if "legacy path" in r.getMessage()]
    assert len(migration) == 1
    msg = migration[0].getMessage()
    assert str(tmp_path / ".enchanter") in msg
    assert str(tmp_path / ".robit") in msg


def test_resolve_user_dir_returns_uncreated_robit_when_neither_exists(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(_compat.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(os, "name", "posix")
    # Neither dir exists.
    result = _compat.resolve_user_dir()
    assert result == tmp_path / ".robit"
    assert not result.exists()  # caller is responsible for mkdir on first write
