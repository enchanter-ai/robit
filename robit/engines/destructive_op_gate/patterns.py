"""W5 destructive-op pattern table — port of `src/plugins/sylph.adapter.ts`.

Each entry is a regex that matches a dangerous/irrecoverable operation.
`requires_consent=True` → veto on match. `requires_consent=False` → advisory
warning only (does not block).
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class DestructiveOpPattern:
    id: str
    name: str
    regex: re.Pattern[str]
    requires_consent: bool  # True = always veto; False = advisory only


DESTRUCTIVE_OP_PATTERNS: tuple[DestructiveOpPattern, ...] = (
    DestructiveOpPattern(
        id="w5-force-push",
        name="git push --force",
        regex=re.compile(r"git\s+push\b[^|&\n]*--force(?!-with-lease)"),
        requires_consent=True,
    ),
    DestructiveOpPattern(
        id="w5-force-push-with-lease-protected",
        name="git push --force-with-lease to protected branch",
        regex=re.compile(r"git\s+push\b[^|&\n]*--force-with-lease"),
        requires_consent=True,
    ),
    DestructiveOpPattern(
        id="w5-reset-hard",
        name="git reset --hard",
        regex=re.compile(r"git\s+reset\b[^|&\n]*--hard"),
        requires_consent=True,
    ),
    DestructiveOpPattern(
        id="w5-branch-delete-force",
        name="git branch -D (force delete)",
        regex=re.compile(r"git\s+branch\b[^|&\n]*-D\b"),
        requires_consent=True,
    ),
    DestructiveOpPattern(
        id="w5-rm-rf",
        name="rm -rf (irrecoverable delete)",
        regex=re.compile(
            r"\brm\b[^|&\n]*-[a-zA-Z]*r[a-zA-Z]*f\b|\brm\b[^|&\n]*-[a-zA-Z]*f[a-zA-Z]*r\b"
        ),
        requires_consent=True,
    ),
    DestructiveOpPattern(
        id="w5-git-push-bare",
        name="git push (plain, potential protected-branch push)",
        regex=re.compile(r"git\s+push\b(?![^|&\n]*--force)(?![^|&\n]*--delete)"),
        requires_consent=False,  # advisory only; force variants above take priority
    ),
)
