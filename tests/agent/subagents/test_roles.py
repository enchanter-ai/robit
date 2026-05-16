"""Tests for robit.agent.subagents.roles + registry."""

from __future__ import annotations

import pytest

from robit.agent.subagents.registry import (
    SubagentRegistry,
    SubagentRole,
    default_registry,
)
from robit.agent.subagents.roles import (
    DEEP_RESEARCH,
    FIND_REFERENCES,
    REVIEW_DIFF,
    default_roles,
)


# Tool names a real production registry exposes; the role's allowed_tools
# must be a subset of this. Pulled from default_registry() in agent.tools.
_PRODUCTION_TOOL_NAMES = {
    "file_read",
    "file_write",
    "file_edit",
    "glob",
    "grep",
    "bash",
    "web_fetch",
}


def test_default_roles_returns_three():
    roles = default_roles()
    assert len(roles) == 3
    names = [r.name for r in roles]
    assert "deep-research" in names
    assert "find-references" in names
    assert "review-diff" in names


def test_each_role_allowed_tools_subset_of_production():
    for role in default_roles():
        assert role.allowed_tools is not None, role.name
        assert len(role.allowed_tools) > 0, role.name
        for tool in role.allowed_tools:
            assert tool in _PRODUCTION_TOOL_NAMES, (
                f"role {role.name!r} lists unknown tool {tool!r}"
            )


def test_each_role_system_prompt_non_empty_and_substantial():
    for role in default_roles():
        assert role.system_prompt
        # > 100 chars is the contract from the wave spec — these are
        # specialist prompts, not one-liners.
        assert len(role.system_prompt) > 100, role.name
        # Should at least mention the role name or its purpose.
        assert role.name.split("-")[0] in role.system_prompt.lower() or (
            "subagent" in role.system_prompt.lower()
        ), role.name


def test_each_role_has_summary_schema_with_required_fields():
    for role in default_roles():
        assert role.summary_schema is not None, role.name
        assert role.summary_schema.get("type") == "object", role.name
        assert "properties" in role.summary_schema, role.name


def test_max_turns_within_sane_bounds():
    for role in default_roles():
        assert 1 <= role.max_turns <= 50, (role.name, role.max_turns)


def test_descriptions_are_paragraph_length():
    """Descriptions feed tool-choice routing — must be informative."""
    for role in default_roles():
        assert len(role.description) > 80, role.name


def test_registry_register_get_contains():
    reg = SubagentRegistry()
    role = SubagentRole(
        name="dummy",
        description="just a test",
        system_prompt="you are dummy" * 20,  # >100 chars
        allowed_tools=("file_read",),
    )
    assert "dummy" not in reg
    reg.register(role)
    assert "dummy" in reg
    assert reg.get("dummy") is role
    assert reg.names() == ("dummy",)
    assert len(reg) == 1


def test_registry_double_register_raises():
    reg = SubagentRegistry()
    role = SubagentRole(
        name="dup",
        description="x",
        system_prompt="x" * 200,
        allowed_tools=None,
    )
    reg.register(role)
    with pytest.raises(ValueError, match="already registered"):
        reg.register(role)


def test_registry_get_unknown_raises_keyerror():
    reg = SubagentRegistry()
    with pytest.raises(KeyError, match="no such subagent role"):
        reg.get("nope")


def test_registry_register_rejects_non_role():
    reg = SubagentRegistry()
    with pytest.raises(TypeError):
        reg.register("not-a-role")  # type: ignore[arg-type]


def test_default_registry_loads_three_roles():
    reg = default_registry()
    assert len(reg) == 3
    assert "deep-research" in reg
    assert "find-references" in reg
    assert "review-diff" in reg


def test_deep_research_prompt_mentions_tools_and_turns():
    """Spot-check: the prompt must name its tools + the turn budget."""
    p = DEEP_RESEARCH.system_prompt.lower()
    assert "web_fetch" in p
    assert "file_read" in p
    assert "15" in p  # turn budget
    assert "json" in p  # output format


def test_find_references_prompt_mentions_grep():
    p = FIND_REFERENCES.system_prompt.lower()
    assert "grep" in p
    assert "5" in p  # turn budget


def test_review_diff_prompt_mentions_verdict():
    p = REVIEW_DIFF.system_prompt.lower()
    assert "verdict" in p
    assert "severity" in p
