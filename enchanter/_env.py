"""enchanter._env — stdlib .env autoloader.

Loads ``.env`` files at CLI startup so users don't have to ``source`` or
``export`` manually. Both ``enchanter`` and ``insighter`` call
:func:`load_env_files` very early in their ``main()``.

Lookup precedence (highest wins, never overrides shell env unless
``override=True``):

1. ``<cwd>/.env``
2. ``<user_dir>/.env``  (``ENCHANTER_HOME`` env var if set, else
   ``%APPDATA%/enchanter`` on Windows or ``~/.enchanter`` on POSIX)

Within a single file, last-key-wins on duplicate keys.

Format: a subset of standard dotenv — ``KEY=value``, ``# comments``,
blank lines, double- and single-quoted values, ``export`` prefix
tolerated. Double-quoted strings support ``\\n`` ``\\t`` ``\\\\`` ``\\"``
escapes. Single-quoted strings are literal. No interpolation: ``$VAR``
in a value is a literal ``$VAR``.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

_LOG = logging.getLogger(__name__)

# A valid identifier-like key: letter/underscore then letters/digits/underscores.
_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _default_user_dir() -> Path:
    """Resolve the user-level .env directory.

    Honors ``ENCHANTER_HOME`` first. Else: ``%APPDATA%/enchanter`` on
    Windows (falling back to ``~/.enchanter`` if APPDATA is unset), or
    ``~/.enchanter`` on POSIX.
    """
    override = os.environ.get("ENCHANTER_HOME")
    if override:
        return Path(override)
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "enchanter"
        return Path.home() / ".enchanter"
    return Path.home() / ".enchanter"


def _unescape_double_quoted(s: str) -> str:
    """Apply ``\\n \\t \\\\ \\"`` escapes inside a double-quoted value."""
    out: list[str] = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "\\" and i + 1 < len(s):
            nxt = s[i + 1]
            if nxt == "n":
                out.append("\n")
            elif nxt == "t":
                out.append("\t")
            elif nxt == "r":
                out.append("\r")
            elif nxt == "\\":
                out.append("\\")
            elif nxt == '"':
                out.append('"')
            else:
                # Unknown escape — keep both chars literally.
                out.append(ch)
                out.append(nxt)
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _strip_inline_comment(unquoted_value: str) -> str:
    """Remove a trailing ` #...` comment from an unquoted value.

    Splits on the first ' #' (space-hash). A leading '#' on the value
    itself (no space before it) is preserved as-is, since dotenv files
    rarely intend that and the spec here only strips space-prefixed
    comments.
    """
    idx = unquoted_value.find(" #")
    if idx == -1:
        return unquoted_value
    return unquoted_value[:idx]


def parse_env_file(text: str) -> list[tuple[str, str]]:
    """Parse dotenv *text* into an ordered list of ``(key, value)`` pairs.

    Invalid lines are logged at WARNING and skipped; parsing continues.
    Duplicate keys are preserved in order — callers decide last-wins.
    """
    pairs: list[tuple[str, str]] = []
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue

        # Tolerate `export KEY=value`.
        if line.startswith("export ") or line.startswith("export\t"):
            line = line[len("export"):].lstrip()

        eq = line.find("=")
        if eq <= 0:
            # No '=' or empty key (e.g. '=value' or 'KEY VALUE').
            _LOG.warning("dotenv: line %d: invalid syntax: %r", lineno, raw_line)
            continue

        key = line[:eq].strip()
        rest = line[eq + 1:]

        if not _KEY_RE.match(key):
            _LOG.warning("dotenv: line %d: invalid key %r", lineno, key)
            continue

        # Strip leading whitespace from the value side; preserve quoted content.
        rest = rest.lstrip()
        if not rest:
            pairs.append((key, ""))
            continue

        first = rest[0]
        if first == '"':
            # Find closing unescaped double quote.
            i = 1
            while i < len(rest):
                if rest[i] == "\\" and i + 1 < len(rest):
                    i += 2
                    continue
                if rest[i] == '"':
                    break
                i += 1
            else:
                _LOG.warning(
                    "dotenv: line %d: unterminated double-quoted value", lineno
                )
                continue
            value = _unescape_double_quoted(rest[1:i])
            pairs.append((key, value))
        elif first == "'":
            # Literal; find next single quote.
            j = rest.find("'", 1)
            if j == -1:
                _LOG.warning(
                    "dotenv: line %d: unterminated single-quoted value", lineno
                )
                continue
            value = rest[1:j]
            pairs.append((key, value))
        else:
            value = _strip_inline_comment(rest).rstrip()
            pairs.append((key, value))

    return pairs


def _load_file(path: Path) -> dict[str, str]:
    """Read *path* and parse it; last-key-wins within the file."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        _LOG.warning("dotenv: cannot read %s: %s", path, exc)
        return {}

    merged: dict[str, str] = {}
    for k, v in parse_env_file(text):
        merged[k] = v  # last wins
    return merged


def load_env_files(
    *,
    cwd: Path | None = None,
    user_dir: Path | None = None,
    override: bool = False,
) -> dict[str, str]:
    """Load .env files and apply to ``os.environ``.

    Precedence (highest wins on key collision across files): ``cwd/.env``
    then ``user_dir/.env``. Within a file: last-key-wins.

    By default (``override=False``), variables already present in
    ``os.environ`` are skipped — the shell wins. With ``override=True``,
    .env values overwrite the shell.

    Returns a dict of vars that were actually applied (so callers can
    log if they want). Silently no-ops if files don't exist.
    """
    if cwd is None:
        cwd = Path.cwd()
    if user_dir is None:
        user_dir = _default_user_dir()

    # Lower precedence first; higher precedence overlays.
    user_pairs = _load_file(user_dir / ".env")
    cwd_pairs = _load_file(cwd / ".env")

    merged: dict[str, str] = {}
    merged.update(user_pairs)
    merged.update(cwd_pairs)  # cwd > user_dir

    applied: dict[str, str] = {}
    for key, value in merged.items():
        if not override and key in os.environ:
            continue
        os.environ[key] = value
        applied[key] = value
    return applied
