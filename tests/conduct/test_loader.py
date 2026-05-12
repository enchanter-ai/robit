"""Tests for enchanter.conduct.loader.load_conduct using temporary directories."""

from pathlib import Path

import pytest

from enchanter.conduct.loader import load_conduct
from enchanter.conduct.types import ConductFrontmatterError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_conduct_file(
    tmp_path: Path, package: str, stem: str, content: str
) -> Path:
    """Create <tmp_path>/packages/<package>/conduct/<stem>.md with content."""
    conduct_dir = tmp_path / "packages" / package / "conduct"
    conduct_dir.mkdir(parents=True, exist_ok=True)
    md = conduct_dir / f"{stem}.md"
    md.write_text(content, encoding="utf-8")
    return md


# ---------------------------------------------------------------------------
# 1. Empty directory → empty list
# ---------------------------------------------------------------------------


def test_empty_directory_returns_empty_list(tmp_path: Path):
    rules = load_conduct(root=tmp_path)
    assert rules == []


# ---------------------------------------------------------------------------
# 2. Single file without frontmatter → defaults applied
# ---------------------------------------------------------------------------


def test_single_file_no_frontmatter_defaults(tmp_path: Path):
    _make_conduct_file(
        tmp_path, "core", "discipline", "# Discipline\n\nBody here.\n"
    )
    rules = load_conduct(root=tmp_path)
    assert len(rules) == 1
    r = rules[0]
    assert r.name == "discipline"
    assert r.enforcement == "prompt"
    assert r.tags == ()
    assert r.package == "core"
    assert "Body here." in r.body


# ---------------------------------------------------------------------------
# 3. Multiple files in nested packages/*/conduct/ → all loaded with correct package
# ---------------------------------------------------------------------------


def test_multiple_files_across_packages(tmp_path: Path):
    _make_conduct_file(tmp_path, "core", "discipline", "# D\n")
    _make_conduct_file(tmp_path, "core", "context", "# C\n")
    _make_conduct_file(tmp_path, "web", "web-fetch", "# W\n")

    rules = load_conduct(root=tmp_path)
    assert len(rules) == 3

    by_name = {r.name: r for r in rules}
    assert by_name["discipline"].package == "core"
    assert by_name["context"].package == "core"
    assert by_name["web-fetch"].package == "web"


# ---------------------------------------------------------------------------
# 4. File with explicit enforcement: code → respected
# ---------------------------------------------------------------------------


def test_explicit_enforcement_code(tmp_path: Path):
    content = "---\nenforcement: code\n---\n# Rule\n"
    _make_conduct_file(tmp_path, "safety", "guard", content)
    rules = load_conduct(root=tmp_path)
    assert len(rules) == 1
    assert rules[0].enforcement == "code"


# ---------------------------------------------------------------------------
# 5. Tags field parsed correctly
# ---------------------------------------------------------------------------


def test_tags_parsed_correctly(tmp_path: Path):
    content = "---\ntags: [security, runtime]\n---\n# Rule\n"
    _make_conduct_file(tmp_path, "safety", "audit", content)
    rules = load_conduct(root=tmp_path)
    assert len(rules) == 1
    assert rules[0].tags == ("security", "runtime")


# ---------------------------------------------------------------------------
# 6. Name override in frontmatter takes precedence over stem
# ---------------------------------------------------------------------------


def test_name_override_from_frontmatter(tmp_path: Path):
    content = "---\nname: custom-name\n---\n# Rule\n"
    _make_conduct_file(tmp_path, "core", "original-stem", content)
    rules = load_conduct(root=tmp_path)
    assert rules[0].name == "custom-name"


# ---------------------------------------------------------------------------
# 7. Malformed frontmatter propagates ConductFrontmatterError
# ---------------------------------------------------------------------------


def test_malformed_frontmatter_raises(tmp_path: Path):
    content = "---\nname: broken\n# no closing delimiter"
    _make_conduct_file(tmp_path, "core", "broken", content)
    with pytest.raises(ConductFrontmatterError):
        load_conduct(root=tmp_path)


# ---------------------------------------------------------------------------
# 8. Body does NOT include frontmatter lines
# ---------------------------------------------------------------------------


def test_body_excludes_frontmatter(tmp_path: Path):
    content = "---\nname: my-rule\n---\n# My Rule\n\nActual body.\n"
    _make_conduct_file(tmp_path, "core", "my-rule", content)
    rules = load_conduct(root=tmp_path)
    assert "---" not in rules[0].body
    assert "name: my-rule" not in rules[0].body
    assert "# My Rule" in rules[0].body
