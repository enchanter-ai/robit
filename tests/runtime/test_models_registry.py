"""tests/runtime/test_models_registry.py — unit tests for ModelsRegistry.

All tests run against the real bundled models-registry.json (255 models as of
2026-04-24).  No mocking — if the registry file changes shape, tests catch it.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from enchanter.runtime.models_registry import (
    ModelEntry,
    ModelsRegistry,
    UnknownFamilyError,
    UnknownModelError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def registry() -> ModelsRegistry:
    """Load the real bundled registry once for the whole module."""
    return ModelsRegistry.load()


# ---------------------------------------------------------------------------
# Test 1 — load() against the real file returns >= 200 models
# ---------------------------------------------------------------------------

def test_load_returns_enough_models(registry: ModelsRegistry) -> None:
    """The bundled registry has 255 models; we assert >= 200 as a stable floor."""
    assert len(registry) >= 200, (
        f"Expected >= 200 models in registry, got {len(registry)}"
    )


# ---------------------------------------------------------------------------
# Test 2 — model(model_id) returns a typed ModelEntry
# ---------------------------------------------------------------------------

def test_model_returns_typed_entry(registry: ModelsRegistry) -> None:
    entry = registry.model("claude-opus-4-7")
    assert isinstance(entry, ModelEntry)
    assert entry.model_id == "claude-opus-4-7"
    assert entry.family == "Claude 4.x"
    assert entry.context_window > 0
    assert isinstance(entry.display_name, str) and entry.display_name


# ---------------------------------------------------------------------------
# Test 3 — family("Claude 4.x") returns at least one entry
# ---------------------------------------------------------------------------

def test_family_returns_entries(registry: ModelsRegistry) -> None:
    members = registry.family("Claude 4.x")
    assert len(members) >= 1
    assert all(isinstance(e, ModelEntry) for e in members)
    assert all(e.family == "Claude 4.x" for e in members)


# ---------------------------------------------------------------------------
# Test 4 — latest_in_family("Claude 4.x") returns the highest-versioned entry
# ---------------------------------------------------------------------------

def test_latest_in_family_returns_highest_version(registry: ModelsRegistry) -> None:
    latest = registry.latest_in_family("Claude 4.x")
    members = registry.family("Claude 4.x")
    # The latest must have the lexicographically greatest model_id in the family.
    expected_id = max(m.model_id for m in members)
    assert latest.model_id == expected_id, (
        f"latest_in_family returned {latest.model_id!r}, expected {expected_id!r}"
    )


# ---------------------------------------------------------------------------
# Test 5 — unknown model_id raises UnknownModelError with the id in the message
# ---------------------------------------------------------------------------

def test_unknown_model_raises(registry: ModelsRegistry) -> None:
    with pytest.raises(UnknownModelError) as exc_info:
        registry.model("this-model-does-not-exist-xyz")
    assert "this-model-does-not-exist-xyz" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 6 — malformed JSON raises ValueError with the path in the message
# ---------------------------------------------------------------------------

def test_malformed_json_raises_with_path() -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        f.write("{not valid json")
        bad_path = Path(f.name)

    with pytest.raises(ValueError) as exc_info:
        ModelsRegistry.load(bad_path)

    assert str(bad_path) in str(exc_info.value), (
        f"Expected path {bad_path} in error message, got: {exc_info.value}"
    )


# ---------------------------------------------------------------------------
# Bonus test 7 — all() returns every entry
# ---------------------------------------------------------------------------

def test_all_returns_all_entries(registry: ModelsRegistry) -> None:
    entries = registry.all()
    assert len(entries) == len(registry)
    assert all(isinstance(e, ModelEntry) for e in entries)


# ---------------------------------------------------------------------------
# Bonus test 8 — unknown family raises UnknownFamilyError
# ---------------------------------------------------------------------------

def test_unknown_family_raises(registry: ModelsRegistry) -> None:
    with pytest.raises(UnknownFamilyError) as exc_info:
        registry.family("NoSuchFamily-XYZ")
    assert "NoSuchFamily-XYZ" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Bonus test 9 — ModelEntry preserves extras for unknown fields
# ---------------------------------------------------------------------------

def test_model_entry_extras() -> None:
    """Extra keys in the raw dict end up in ModelEntry.extras."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        data = {
            "models": {
                "test-model-1": {
                    "family": "TestFamily",
                    "display_name": "Test Model",
                    "context_window": 8000,
                    "format": "xml",
                    "reasoning": "standard",
                    "cot_approach": "none",
                    "few_shot": "none",
                    "key_constraint": "none",
                    "custom_field_xyz": "hello",
                }
            }
        }
        json.dump(data, f)
        path = Path(f.name)

    reg = ModelsRegistry.load(path)
    entry = reg.model("test-model-1")
    assert entry.extras.get("custom_field_xyz") == "hello"
