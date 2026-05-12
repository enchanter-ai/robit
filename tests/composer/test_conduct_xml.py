"""Tests for enchanter.composer.conduct — compose_conduct_xml and select_rules."""

import xml.etree.ElementTree as ET

import pytest

from enchanter.composer.conduct import compose_conduct_xml, select_rules


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _rule(
    name: str,
    body: str = "# Body",
    enforcement: str = "prompt",
    package: str = "core",
    tags: tuple = (),
) -> dict:
    return {
        "name": name,
        "body": body,
        "enforcement": enforcement,
        "package": package,
        "tags": tags,
    }


# ---------------------------------------------------------------------------
# compose_conduct_xml
# ---------------------------------------------------------------------------


def test_empty_rule_list_produces_empty_conduct_tag():
    """Empty input → self-contained <conduct version="1"> with no children."""
    result = compose_conduct_xml([])
    # Must be parseable.
    root = ET.fromstring(result)
    assert root.tag == "conduct"
    assert root.attrib["version"] == "1"
    assert len(list(root)) == 0  # no <module> children


def test_single_rule_wrapped_in_module():
    """A single prompt-enforcement rule produces exactly one <module>."""
    rules = [_rule("discipline", body="# Discipline\n\nContent here.")]
    result = compose_conduct_xml(rules)
    root = ET.fromstring(result)
    modules = list(root)
    assert len(modules) == 1
    mod = modules[0]
    assert mod.tag == "module"
    assert mod.attrib["name"] == "discipline"
    assert mod.attrib["package"] == "core"
    # Body text must be preserved (stripped because ET normalises whitespace
    # around element text, but the key words must survive).
    assert "Discipline" in mod.text
    assert "Content here." in mod.text


def test_multiple_rules_ordered_by_package_then_name():
    """Rules are sorted (package asc, name asc) regardless of input order."""
    rules = [
        _rule("verification", package="core"),
        _rule("tool-use", package="core"),
        _rule("alpha", package="beta"),
    ]
    result = compose_conduct_xml(rules)
    root = ET.fromstring(result)
    names = [m.attrib["name"] for m in root]
    # "beta" < "core" → "alpha" first, then "tool-use", then "verification"
    assert names == ["alpha", "tool-use", "verification"]


def test_code_enforcement_rule_is_skipped():
    """Rules with enforcement='code' must NOT appear in the XML output."""
    rules = [
        _rule("discipline", enforcement="prompt"),
        _rule("secret-gate", enforcement="code"),
    ]
    result = compose_conduct_xml(rules)
    root = ET.fromstring(result)
    names = [m.attrib["name"] for m in root]
    assert "secret-gate" not in names
    assert "discipline" in names


def test_hybrid_enforcement_rule_is_included():
    """Rules with enforcement='hybrid' ARE included — treated like 'prompt'."""
    rules = [_rule("dual-mode", enforcement="hybrid")]
    result = compose_conduct_xml(rules)
    root = ET.fromstring(result)
    names = [m.attrib["name"] for m in root]
    assert "dual-mode" in names


def test_xml_special_chars_in_body_are_escaped():
    """< > & in the body must be escaped; the output must still be valid XML."""
    body = "Use <foo> & 'bar' > baz for <context>."
    rules = [_rule("escaping-rule", body=body)]
    result = compose_conduct_xml(rules)
    # Parseable → escape was correct.
    root = ET.fromstring(result)
    mod = list(root)[0]
    # ET parses entities back to their original characters; check round-trip.
    assert "<foo>" in mod.text
    assert "&" in mod.text
    assert ">" in mod.text


def test_output_is_valid_xml():
    """The raw string output must parse without errors via ElementTree."""
    rules = [
        _rule("discipline", body="# Discipline\n\n<example> & test >"),
        _rule("tool-use", body="# Tool Use"),
        _rule("skipped", enforcement="code"),
    ]
    result = compose_conduct_xml(rules)
    # fromstring raises if the XML is malformed.
    ET.fromstring(result)


def test_output_has_conduct_version_attribute():
    """Root tag is <conduct version="1">."""
    result = compose_conduct_xml([_rule("x")])
    root = ET.fromstring(result)
    assert root.attrib.get("version") == "1"


def test_all_three_enforcement_types():
    """prompt and hybrid included; code excluded — three rules in, two out."""
    rules = [
        _rule("a", enforcement="prompt"),
        _rule("b", enforcement="hybrid"),
        _rule("c", enforcement="code"),
    ]
    result = compose_conduct_xml(rules)
    root = ET.fromstring(result)
    names = {m.attrib["name"] for m in root}
    assert names == {"a", "b"}


# ---------------------------------------------------------------------------
# select_rules
# ---------------------------------------------------------------------------


def test_select_rules_none_required_returns_all():
    """required=None means return all rules unchanged."""
    rules = [_rule("a"), _rule("b"), _rule("c")]
    assert select_rules(rules, required=None) == rules


def test_select_rules_filters_by_name():
    """Only rules whose name is in required survive."""
    rules = [_rule("discipline"), _rule("tool-use"), _rule("verification")]
    result = select_rules(rules, required={"discipline", "verification"})
    names = [r["name"] for r in result]
    assert set(names) == {"discipline", "verification"}
    assert "tool-use" not in names


def test_select_rules_empty_required_returns_empty():
    """required={} (empty set) → no rules match."""
    rules = [_rule("a"), _rule("b")]
    assert select_rules(rules, required=set()) == []


def test_select_rules_unknown_names_silently_ignored():
    """Names in required that don't exist in all_rules produce no error."""
    rules = [_rule("a")]
    result = select_rules(rules, required={"a", "nonexistent"})
    assert len(result) == 1
    assert result[0]["name"] == "a"


def test_select_rules_preserves_relative_order():
    """The returned list maintains the original relative order from all_rules."""
    rules = [_rule("z"), _rule("a"), _rule("m")]
    result = select_rules(rules, required={"z", "m"})
    assert [r["name"] for r in result] == ["z", "m"]


# ---------------------------------------------------------------------------
# Integration: select_rules → compose_conduct_xml
# ---------------------------------------------------------------------------


def test_select_then_compose_end_to_end():
    """select_rules piped into compose_conduct_xml produces the expected XML."""
    all_rules = [
        _rule("discipline", body="Discipline body", package="core"),
        _rule("tool-use", body="Tool-use body", package="core"),
        _rule("secret", enforcement="code", package="core"),
        _rule("hook", body="Hook body", package="plugin"),
    ]
    selected = select_rules(all_rules, required={"discipline", "tool-use", "hook"})
    result = compose_conduct_xml(selected)
    root = ET.fromstring(result)
    names = [m.attrib["name"] for m in root]
    # "core" < "plugin" → discipline and tool-use before hook
    assert names == ["discipline", "tool-use", "hook"]
