"""enchanter.agent.slash_commands — opt-in slash command bundles.

This namespace is reserved for slash-command groupings that ship beyond the
Wave 15.0 built-ins in :mod:`enchanter.agent.slash`.  Each submodule provides
an ``all_*_commands()`` factory returning a list of objects conforming to the
:class:`enchanter.agent.slash.SlashCommand` Protocol.

The current bundles:

  * :mod:`enchanter.agent.slash_commands.plan` — ``/plan``, ``/edit``,
    ``/cancel``, ``/execute`` (Wave 15.3 / Agent K).
"""

from __future__ import annotations

from .plan import all_plan_commands

__all__ = ["all_plan_commands"]
