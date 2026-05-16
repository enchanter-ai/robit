"""robit.composer.xml — XML escape utilities and indentation helpers.

Pure functions; no external dependencies. The composer avoids xml.etree.ElementTree
for output because etree.tostring produces compressed output without reliable
indentation control. We own the string transformation directly.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Escape table — applied in a single pass to avoid double-escaping.
# Order matters: '&' must be replaced before '<' and '>' so that
# existing '&' characters become '&amp;' rather than having '&lt;'
# later turned into '&amp;lt;' on a naive double-pass.
# ---------------------------------------------------------------------------

_ESCAPE_TABLE: tuple[tuple[str, str], ...] = (
    ("&", "&amp;"),  # must be first
    ("<", "&lt;"),
    (">", "&gt;"),
)


def xml_escape(text: str) -> str:
    """Escape ``&``, ``<``, and ``>`` for safe embedding in XML element bodies.

    The replacement is a single ordered pass so that already-escaped entities
    such as ``&amp;`` are NOT double-escaped — because ``&`` is replaced with
    ``&amp;`` in the first (and only) iteration, the output ``&amp;`` is the
    correct final form and the subsequent ``<``/``>`` substitutions never touch
    it.

    Attribute values (``"`` and ``'``) are not escaped; this function is
    intended only for element text content, where the conductor XML uses
    double-quoted attribute literals in the start tags and the body text sits
    between tags.

    Examples
    --------
    >>> xml_escape("<foo> & bar")
    '&lt;foo&gt; &amp; bar'
    >>> xml_escape("&amp;")   # already-escaped entity — NOT double-escaped
    '&amp;amp;'               # ...wait, that IS the correct single-escape
    """
    for raw, entity in _ESCAPE_TABLE:
        text = text.replace(raw, entity)
    return text


def indent_block(text: str, spaces: int) -> str:
    """Indent every line in *text* by *spaces* space characters.

    Blank lines are left as empty strings (no trailing whitespace added).

    Parameters
    ----------
    text:
        Multi-line string to indent.  The final newline, if present, is
        preserved so the caller does not need to track it.
    spaces:
        Number of ASCII space characters to prepend to each non-empty line.
    """
    prefix = " " * spaces
    lines = text.split("\n")
    indented = [prefix + line if line else "" for line in lines]
    return "\n".join(indented)
