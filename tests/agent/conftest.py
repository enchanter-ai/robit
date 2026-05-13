"""Shared fixtures for enchanter.agent tests.

Critical: redirect ENCHANTER_HOME to a tmp dir at module import time so
session writes never touch the developer's real ~/.enchanter directory.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_enchanter_home(tmp_path, monkeypatch):
    """Per-test ENCHANTER_HOME so session JSONLs land in tmp."""
    monkeypatch.setenv("ENCHANTER_HOME", str(tmp_path))
    monkeypatch.setenv("ENCHANTER_AGENT_MOCK", "1")
    yield tmp_path
