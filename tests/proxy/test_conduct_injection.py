"""Tests for enchanter.proxy.conduct — system-prompt injection contract."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from enchanter.proxy import conduct as proxy_conduct
from enchanter.proxy.canonical import CanonicalRequest, Message, TextPart
from enchanter.proxy.conduct import DEFAULT_PROXY_RULES, apply_conduct_to_request


# A minimal fake ConductRule that quacks like the real one for ``_rule_to_dict``.
class _FakeRule:
    def __init__(self, name: str, body: str, package: str = "core"):
        self.name = name
        self.body = body
        self.enforcement = "prompt"
        self.package = package
        self.tags: tuple[str, ...] = ()


_FAKE_RULES = [
    _FakeRule("discipline", "# Discipline body"),
    _FakeRule("verification", "# Verification body"),
    _FakeRule("tool-use", "# Tool-use body"),
    _FakeRule("refusal-and-recovery", "# Refusal body"),
    _FakeRule("formatting", "# Formatting body"),
    _FakeRule("never-selected", "# Unrelated body", package="extra"),
]


def _basic_req(system: str | None = None) -> CanonicalRequest:
    return CanonicalRequest(
        model="gpt-4o-mini",
        messages=(Message(role="user", content=(TextPart(text="hi"),)),),
        system=system,
    )


@pytest.fixture
def fake_loader():
    with patch.object(proxy_conduct, "load_conduct", return_value=_FAKE_RULES) as p:
        yield p


def test_default_rule_set_injects_conduct(fake_loader):
    new = apply_conduct_to_request(_basic_req())
    assert new.system is not None
    assert new.system.startswith("<conduct version=")
    # Each default rule's body shows up in the XML.
    for rule in DEFAULT_PROXY_RULES:
        # The body is module-tag matched by name.
        assert f'name="{rule}"' in new.system
    # The unrelated rule is NOT injected.
    assert "never-selected" not in new.system


def test_empty_rule_set_is_a_pass_through(fake_loader):
    req = _basic_req(system="client-system")
    new = apply_conduct_to_request(req, rules=frozenset())
    assert new is req or new.system == "client-system"
    # load_conduct should not even be called for the empty opt-out.
    assert fake_loader.call_count == 0


def test_custom_rule_set_filters_correctly(fake_loader):
    new = apply_conduct_to_request(
        _basic_req(), rules=frozenset({"discipline"})
    )
    assert new.system is not None
    assert 'name="discipline"' in new.system
    assert 'name="verification"' not in new.system


def test_client_system_prompt_is_preserved_and_follows_conduct(fake_loader):
    req = _basic_req(system="You are a pirate.")
    new = apply_conduct_to_request(req)
    assert new.system is not None
    # Client system appears after the conduct XML, separated by blank line.
    assert new.system.endswith("You are a pirate.")
    assert new.system.startswith("<conduct version=")
    assert "\n\nYou are a pirate." in new.system


def test_request_is_not_mutated_in_place(fake_loader):
    req = _basic_req(system="orig")
    new = apply_conduct_to_request(req)
    assert req.system == "orig"
    assert new is not req
    assert new.system != req.system


def test_no_matching_rules_returns_request_unchanged(fake_loader):
    """If the rules subset names nothing the loader returned, leave system alone."""
    req = _basic_req(system="orig")
    new = apply_conduct_to_request(
        req, rules=frozenset({"completely-unknown-rule"})
    )
    assert new.system == "orig"
