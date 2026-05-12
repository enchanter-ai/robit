"""Tests for enchanter.loader.discovery — real engine registry integration."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from enchanter.core.plugin import PluginAdapter
from enchanter.loader import load_engine_registry, find_engine_manifests
from enchanter.loader.errors import EngineLoadError


# ──────────────────────────────────────────────────────────────────────────────
# Resolve the repo root (two levels up from enchanter/loader/discovery.py)
# ──────────────────────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).parent.parent.parent  # enchanter-agent/


# ──────────────────────────────────────────────────────────────────────────────
# Test 1: load_engine_registry() against real engines returns 12 engines
# ──────────────────────────────────────────────────────────────────────────────

def test_load_real_registry_returns_13_engines() -> None:
    registry = load_engine_registry(_REPO_ROOT)
    assert len(registry) == 14, (
        f"Expected 14 engines, got {len(registry)}: {sorted(registry.keys())}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 2: Every returned adapter passes isinstance(a, PluginAdapter)
#         and has the required protocol attributes
# ──────────────────────────────────────────────────────────────────────────────

def test_every_adapter_is_plugin_adapter() -> None:
    registry = load_engine_registry(_REPO_ROOT)

    for engine_name, adapter in registry.items():
        assert isinstance(adapter, PluginAdapter), (
            f"Engine {engine_name!r}: adapter does not satisfy PluginAdapter protocol"
        )
        # Protocol attributes
        assert hasattr(adapter, "name"), f"{engine_name}: missing 'name'"
        assert hasattr(adapter, "phases"), f"{engine_name}: missing 'phases'"
        assert hasattr(adapter, "required"), f"{engine_name}: missing 'required'"
        assert hasattr(adapter, "topics"), f"{engine_name}: missing 'topics'"
        assert hasattr(adapter, "budget_tier"), f"{engine_name}: missing 'budget_tier'"
        assert callable(getattr(adapter, "on_phase", None)), (
            f"{engine_name}: 'on_phase' is not callable"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Test 3: Broken adapter import path raises EngineLoadError with engine name
#         and adapter path
# ──────────────────────────────────────────────────────────────────────────────

def test_broken_adapter_raises_engine_load_error(tmp_path: Path) -> None:
    engines_dir = tmp_path / "enchanter" / "engines" / "broken_engine"
    engines_dir.mkdir(parents=True)
    (engines_dir / "engine.toml").write_text(
        textwrap.dedent("""\
            name = "broken-engine"
            description = "Engine with a broken adapter import path."
            version = "1.0.0"
            phases = ["trust-gate"]
            required = false
            budget_tier = "always"
            adapter = "enchanter.engines.nonexistent_module.adapter:adapter"

            [topics]
            subscribes = ["x"]
            emits = ["y"]
        """),
        encoding="utf-8",
    )

    with pytest.raises(EngineLoadError) as exc_info:
        load_engine_registry(tmp_path)

    err = exc_info.value
    assert err.engine_name == "broken-engine"
    assert err.adapter_path == "enchanter.engines.nonexistent_module.adapter:adapter"


# ──────────────────────────────────────────────────────────────────────────────
# Test 4: A directory without engine.toml is silently skipped
# ──────────────────────────────────────────────────────────────────────────────

def test_directory_without_engine_toml_is_skipped(tmp_path: Path) -> None:
    engines_dir = tmp_path / "enchanter" / "engines"

    # One valid engine.
    valid_dir = engines_dir / "valid_engine"
    valid_dir.mkdir(parents=True)
    (valid_dir / "engine.toml").write_text(
        textwrap.dedent("""\
            name = "valid-engine"
            description = "Valid engine."
            version = "1.0.0"
            phases = ["trust-gate"]
            required = false
            budget_tier = "always"
            adapter = "enchanter.engines.destructive_op_gate.adapter:adapter"

            [topics]
            subscribes = ["x"]
            emits = ["y"]
        """),
        encoding="utf-8",
    )

    # One directory without engine.toml — should be ignored.
    no_toml_dir = engines_dir / "no_manifest_engine"
    no_toml_dir.mkdir(parents=True)
    (no_toml_dir / "adapter.py").write_text("# placeholder", encoding="utf-8")

    manifests = find_engine_manifests(tmp_path)
    assert len(manifests) == 1
    assert manifests[0].parent.name == "valid_engine"

    registry = load_engine_registry(tmp_path)
    assert list(registry.keys()) == ["valid-engine"]
