"""Integration test — loads from the REAL vis conduct files.

This test is intentionally NOT hermetic: it exercises the actual on-disk
conduct modules in ``vis/packages/*/conduct/*.md``.
It is the single integration gate that confirms the loader works end-to-end
against the live source tree.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from robit.conduct import ConductRule, load_conduct
from robit.conduct._paths import DEFAULT_VIS_ROOT

# ---------------------------------------------------------------------------
# Known good packages in vis at the time of writing.
# The loader must resolve at least one file per present package.
# ---------------------------------------------------------------------------
_KNOWN_PACKAGES = frozenset(
    {"core", "web", "skills", "cost", "safety", "orchestration", "memory"}
)


@pytest.fixture(scope="module")
def real_rules() -> list[ConductRule]:
    """Load conduct rules from the real vis tree."""
    return load_conduct(root=DEFAULT_VIS_ROOT)


def test_at_least_10_rules_loaded(real_rules: list[ConductRule]):
    assert len(real_rules) >= 10, (
        f"Expected ≥10 ConductRule objects; got {len(real_rules)}. "
        f"Check that DEFAULT_VIS_ROOT={DEFAULT_VIS_ROOT} is correct."
    )


def test_every_rule_has_non_empty_name(real_rules: list[ConductRule]):
    bad = [r for r in real_rules if not r.name]
    assert not bad, f"Rules with empty name: {bad}"


def test_every_rule_has_non_empty_body(real_rules: list[ConductRule]):
    bad = [r for r in real_rules if not r.body.strip()]
    assert not bad, f"Rules with empty body: {bad}"


def test_every_rule_defaults_to_prompt_enforcement(real_rules: list[ConductRule]):
    """Since no real conduct file has frontmatter yet, all should be 'prompt'."""
    non_prompt = [r for r in real_rules if r.enforcement != "prompt"]
    assert not non_prompt, (
        f"Rules with non-prompt enforcement (unexpected frontmatter?): {non_prompt}"
    )


def test_every_rule_has_known_package(real_rules: list[ConductRule]):
    unknown = [r for r in real_rules if r.package not in _KNOWN_PACKAGES]
    assert not unknown, (
        f"Rules with unrecognised package (new package added?): {unknown}"
    )


def test_default_vis_root_resolves(real_rules: list[ConductRule]):
    """Sanity-check that the default path constant actually points somewhere real."""
    assert DEFAULT_VIS_ROOT.is_dir(), (
        f"DEFAULT_VIS_ROOT does not exist: {DEFAULT_VIS_ROOT}"
    )


def test_paths_are_absolute(real_rules: list[ConductRule]):
    bad = [r for r in real_rules if not r.path.is_absolute()]
    assert not bad, f"Rules with non-absolute paths: {bad}"


def test_rule_names_are_unique(real_rules: list[ConductRule]):
    """Within a package, rule names should be unique."""
    seen: dict[tuple[str, str], ConductRule] = {}
    dupes: list[str] = []
    for r in real_rules:
        key = (r.package, r.name)
        if key in seen:
            dupes.append(f"{r.package}/{r.name}")
        seen[key] = r
    assert not dupes, f"Duplicate (package, name) pairs: {dupes}"
