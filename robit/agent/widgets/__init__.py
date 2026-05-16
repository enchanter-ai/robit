"""robit.agent.widgets — Textual widget package.

Wave 15.2 splits the REPL UI into focused widgets so siblings can land in
parallel without stepping on each other:

* ``repl.py``           — main REPL container (log + input + approval slot)
* ``footer.py``         — model / session / cost ticker mount point
* ``diff.py``  (G)      — exports ``DiffView(diff_text)``
* ``enforcement.py`` (H) — exports ``EnforcementChip(kind, label)``
* ``cost.py``  (I)      — exports ``CostTicker`` with ``add_turn(...)``

Sibling files may not yet exist on disk. Importers must use ``try/except
ImportError`` and fall back to plain-text rendering — never crash the app
because a sibling widget is missing.
"""

from __future__ import annotations

__all__: list[str] = []
