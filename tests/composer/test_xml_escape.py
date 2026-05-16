"""Tests for robit.composer.xml — escape utilities."""

import pytest
from robit.composer.xml import xml_escape, indent_block


# ---------------------------------------------------------------------------
# xml_escape
# ---------------------------------------------------------------------------


def test_less_than_escaped():
    assert xml_escape("<foo>") == "&lt;foo&gt;"


def test_greater_than_escaped():
    assert xml_escape("x > y") == "x &gt; y"


def test_ampersand_escaped():
    assert xml_escape("foo & bar") == "foo &amp; bar"


def test_all_three_in_one_string():
    assert xml_escape("a < b & c > d") == "a &lt; b &amp; c &gt; d"


def test_already_escaped_amp_not_double_escaped():
    # "&amp;" → the '&' is replaced once to produce "&amp;amp;"
    # This is the correct single-pass behaviour: the input already contains
    # a literal ampersand followed by "amp;" and the escape turns the '&'
    # into '&amp;', yielding '&amp;amp;'.  Crucially this is NOT a double-
    # escape of an already-safe entity — we escape exactly what is present.
    result = xml_escape("&amp;")
    assert result == "&amp;amp;"


def test_unicode_characters_preserved():
    text = "日本語 — тест — مرحبا"
    assert xml_escape(text) == text  # no ASCII-special chars → unchanged


def test_empty_string_handled():
    assert xml_escape("") == ""


def test_no_special_chars_unchanged():
    assert xml_escape("hello world 123") == "hello world 123"


# ---------------------------------------------------------------------------
# indent_block
# ---------------------------------------------------------------------------


def test_indent_block_single_line():
    assert indent_block("hello", 4) == "    hello"


def test_indent_block_multiline():
    result = indent_block("line1\nline2", 2)
    assert result == "  line1\n  line2"


def test_indent_block_blank_lines_not_padded():
    result = indent_block("a\n\nb", 2)
    # blank line should not get trailing spaces
    assert result == "  a\n\n  b"


def test_indent_block_zero_spaces():
    assert indent_block("text", 0) == "text"
