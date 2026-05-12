"""Tests for enchanter.conduct.frontmatter.parse_frontmatter."""

from pathlib import Path

import pytest

from enchanter.conduct.frontmatter import parse_frontmatter
from enchanter.conduct.types import ConductFrontmatterError

_FAKE_PATH = Path("fake/discipline.md")


# ---------------------------------------------------------------------------
# 1. No frontmatter → ({}, full_text)
# ---------------------------------------------------------------------------


def test_no_frontmatter_returns_empty_meta_and_full_text():
    text = "# Discipline\n\nSome body content.\n"
    meta, body = parse_frontmatter(text, path=_FAKE_PATH)
    assert meta == {}
    assert body == text


# ---------------------------------------------------------------------------
# 2. Empty frontmatter (---\n---\n) → ({}, body after closing ---)
# ---------------------------------------------------------------------------


def test_empty_frontmatter_returns_empty_meta():
    text = "---\n---\n# Title\n\nBody here.\n"
    meta, body = parse_frontmatter(text, path=_FAKE_PATH)
    assert meta == {}
    assert body == "# Title\n\nBody here.\n"


# ---------------------------------------------------------------------------
# 3. Single key:value → parsed correctly
# ---------------------------------------------------------------------------


def test_single_key_value_parsed():
    text = "---\nname: discipline\n---\n# Discipline\n"
    meta, body = parse_frontmatter(text, path=_FAKE_PATH)
    assert meta == {"name": "discipline"}
    assert body == "# Discipline\n"


# ---------------------------------------------------------------------------
# 4. Inline list  tags: [a, b]  → list
# ---------------------------------------------------------------------------


def test_inline_list_parsed():
    text = "---\ntags: [coding, behavior]\n---\nBody.\n"
    meta, body = parse_frontmatter(text, path=_FAKE_PATH)
    assert meta["tags"] == ["coding", "behavior"]


# ---------------------------------------------------------------------------
# 5. Block list  (with "- " indent) → parsed as list
# ---------------------------------------------------------------------------


def test_block_list_parsed():
    text = (
        "---\n"
        "tags:\n"
        "  - coding\n"
        "  - behavior\n"
        "---\n"
        "Body.\n"
    )
    meta, body = parse_frontmatter(text, path=_FAKE_PATH)
    assert meta["tags"] == ["coding", "behavior"]
    assert body == "Body.\n"


# ---------------------------------------------------------------------------
# 6. Malformed frontmatter (unclosed ---) raises ConductFrontmatterError
# ---------------------------------------------------------------------------


def test_unclosed_frontmatter_raises():
    text = "---\nname: discipline\n# no closing delimiter\n"
    with pytest.raises(ConductFrontmatterError) as exc_info:
        parse_frontmatter(text, path=_FAKE_PATH)
    assert "closing" in str(exc_info.value).lower() or "delimiter" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Extra coverage: bool and int coercion
# ---------------------------------------------------------------------------


def test_bool_and_int_coercion():
    text = "---\nenabled: true\npriority: 3\n---\n"
    meta, _ = parse_frontmatter(text, path=_FAKE_PATH)
    assert meta["enabled"] is True
    assert meta["priority"] == 3


# ---------------------------------------------------------------------------
# Extra coverage: leading blank line → treated as no frontmatter
# ---------------------------------------------------------------------------


def test_leading_blank_line_means_no_frontmatter():
    text = "\n---\nname: oops\n---\nBody.\n"
    meta, body = parse_frontmatter(text, path=_FAKE_PATH)
    assert meta == {}
    assert body == text
