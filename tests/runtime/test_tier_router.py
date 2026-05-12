"""tests/runtime/test_tier_router.py — unit tests for TierRouter.

All tests run against the real bundled registry (255 models).
"""

from __future__ import annotations

import pytest

from enchanter.runtime.models_registry import ModelsRegistry, UnknownModelError
from enchanter.runtime.tier_router import (
    MissingDefaultFamilyError,
    TierRouter,
    UnknownTaskClassError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def registry() -> ModelsRegistry:
    return ModelsRegistry.load()


@pytest.fixture(scope="module")
def router(registry: ModelsRegistry) -> TierRouter:
    return TierRouter(registry)


# ---------------------------------------------------------------------------
# Test 1 — default routing: orchestrator → Opus, executor → Sonnet, validator → Haiku
# ---------------------------------------------------------------------------

def test_default_orchestrator(router: TierRouter) -> None:
    model_id = router.route("orchestrator")
    assert "opus" in model_id.lower(), (
        f"Expected orchestrator to route to an Opus model, got {model_id!r}"
    )
    # Specific preferred model should be present in the registry.
    assert model_id == "claude-opus-4-7", (
        f"Expected 'claude-opus-4-7', got {model_id!r}"
    )


def test_default_executor(router: TierRouter) -> None:
    model_id = router.route("executor")
    assert "sonnet" in model_id.lower(), (
        f"Expected executor to route to a Sonnet model, got {model_id!r}"
    )
    assert model_id == "claude-sonnet-4-6", (
        f"Expected 'claude-sonnet-4-6', got {model_id!r}"
    )


def test_default_validator(router: TierRouter) -> None:
    model_id = router.route("validator")
    assert "haiku" in model_id.lower(), (
        f"Expected validator to route to a Haiku model, got {model_id!r}"
    )
    assert model_id == "claude-haiku-4-5", (
        f"Expected 'claude-haiku-4-5', got {model_id!r}"
    )


# ---------------------------------------------------------------------------
# Test 2 — overrides: pinned validator
# ---------------------------------------------------------------------------

def test_override_validator(registry: ModelsRegistry) -> None:
    router = TierRouter(registry, overrides={"validator": "claude-haiku-4-5"})
    assert router.route("validator") == "claude-haiku-4-5"


def test_override_does_not_affect_other_tiers(registry: ModelsRegistry) -> None:
    """An override on validator must not change orchestrator or executor."""
    router = TierRouter(registry, overrides={"validator": "claude-haiku-4-5"})
    assert "opus" in router.route("orchestrator").lower()
    assert "sonnet" in router.route("executor").lower()


# ---------------------------------------------------------------------------
# Test 3 — unknown task_class raises UnknownTaskClassError
# ---------------------------------------------------------------------------

def test_unknown_task_class_raises(router: TierRouter) -> None:
    with pytest.raises(UnknownTaskClassError) as exc_info:
        router.route("supreme-overlord")  # type: ignore[arg-type]
    assert "supreme-overlord" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 4 — size_hint accepted without error, result unchanged
# ---------------------------------------------------------------------------

def test_size_hint_accepted(router: TierRouter) -> None:
    result_without = router.route("executor")
    result_with = router.route("executor", size_hint=200_000)
    assert result_without == result_with, (
        "size_hint should not change the result (not yet implemented)"
    )


def test_size_hint_zero_accepted(router: TierRouter) -> None:
    # Edge case: size_hint=0 should not raise.
    model_id = router.route("validator", size_hint=0)
    assert isinstance(model_id, str) and model_id


# ---------------------------------------------------------------------------
# Test 5 — missing default family raises MissingDefaultFamilyError
# ---------------------------------------------------------------------------

def test_missing_default_family_raises(registry: ModelsRegistry) -> None:
    """If the preferred family is absent, router must raise, naming the family."""
    import json
    import tempfile
    from pathlib import Path

    # Build a minimal registry that has NO Claude models and NO image/embed models.
    minimal = {
        "models": {
            "some-model-1": {
                "family": "SomeOtherFamily",
                "display_name": "Some Model",
                "context_window": 8000,
                "format": "markdown",
                "reasoning": "standard",
                "cot_approach": "none",
                "few_shot": "none",
                "key_constraint": "none",
            }
        }
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(minimal, f)
        path = Path(f.name)

    stub_registry = ModelsRegistry.load(path)

    with pytest.raises(MissingDefaultFamilyError) as exc_info:
        TierRouter(stub_registry)

    err = exc_info.value
    # The error must name a task_class and expected families.
    assert err.task_class in ("orchestrator", "executor", "validator", "image", "embed")
    assert len(err.families) > 0


# ---------------------------------------------------------------------------
# Test 6 — route() is idempotent: same args → same model_id
# ---------------------------------------------------------------------------

def test_route_idempotent(router: TierRouter) -> None:
    for task_class in ("orchestrator", "executor", "validator", "image", "embed"):
        first = router.route(task_class)  # type: ignore[arg-type]
        second = router.route(task_class)  # type: ignore[arg-type]
        assert first == second, (
            f"route({task_class!r}) returned different results: "
            f"{first!r} vs {second!r}"
        )


# ---------------------------------------------------------------------------
# Test 7 — image and embed task classes resolve to real registry entries
# ---------------------------------------------------------------------------

def test_image_routes_to_known_model(router: TierRouter, registry: ModelsRegistry) -> None:
    model_id = router.route("image")
    entry = registry.model(model_id)  # must not raise
    assert entry.model_id == model_id


def test_embed_routes_to_known_model(router: TierRouter, registry: ModelsRegistry) -> None:
    model_id = router.route("embed")
    entry = registry.model(model_id)
    assert entry.model_id == model_id


# ---------------------------------------------------------------------------
# Test 8 — override with unknown model_id raises UnknownModelError at construction
# ---------------------------------------------------------------------------

def test_override_unknown_model_raises(registry: ModelsRegistry) -> None:
    with pytest.raises(UnknownModelError):
        TierRouter(registry, overrides={"executor": "no-such-model-xyz"})


# ---------------------------------------------------------------------------
# Test 9 — override with unknown task_class raises UnknownTaskClassError
# ---------------------------------------------------------------------------

def test_override_unknown_task_class_raises(registry: ModelsRegistry) -> None:
    with pytest.raises(UnknownTaskClassError):
        TierRouter(registry, overrides={"supreme-overlord": "claude-opus-4-7"})
