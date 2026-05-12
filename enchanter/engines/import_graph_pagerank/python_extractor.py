"""python_extractor — AST-based Python import extractor.

Port of gorgon/python-extractor.ts (extractPythonImports), but using the
Python stdlib ``ast`` module instead of regex so that parenthesised imports,
``__future__`` imports, and other edge cases are handled correctly.

Public API
----------
extract_imports(source: str) -> list[str]
    Given a Python source string, return a deduplicated sorted list of
    imported module names.

    Mapping (matches TS extractPythonImports behaviour):
      import foo             → "foo"
      import foo.bar         → "foo.bar"
      import foo as f        → "foo"
      import foo, bar        → ["foo", "bar"]
      from foo import bar    → "foo"
      from foo.bar import baz → "foo.bar"
      from . import x        → "__relative__"
      from .pkg import x     → "__relative__"

    Malformed source (SyntaxError or any parse error) returns [] — fail-open.
    Never raises.
"""

from __future__ import annotations

import ast


_RELATIVE_SENTINEL = "__relative__"


def extract_imports(source: str) -> list[str]:
    """Parse *source* with ``ast.parse`` and return imported module names.

    Returns a deduplicated, sorted list.  Relative imports are returned as
    the sentinel ``"__relative__"`` (one entry regardless of how many
    relative imports appear).  If parsing fails for any reason, returns ``[]``.
    """
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError:
        return []
    except Exception:  # noqa: BLE001 — defensive; ast.parse can also raise ValueError
        return []

    seen: set[str] = set()
    result: list[str] = []

    def _add(name: str) -> None:
        if name not in seen:
            seen.add(name)
            result.append(name)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            # import foo, import foo.bar, import foo as f
            for alias in node.names:
                _add(alias.name)

        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                # relative import: from . import x  /  from .pkg import y
                _add(_RELATIVE_SENTINEL)
            else:
                # absolute import: from foo.bar import baz
                if node.module:
                    _add(node.module)

    result.sort()
    return result
