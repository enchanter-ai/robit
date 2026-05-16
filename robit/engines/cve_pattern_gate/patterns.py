"""CVE pattern table — port of `src/plugins/hydra/cve-patterns.ts` CVE_PATTERNS_V0_1.

Each CvePattern is a frozen dataclass holding a compiled regex, a severity tier,
the CVE/CWE anchor it represents, and a human-readable rationale surfaced in veto
reason strings.

severity tiers:
  critical → veto (fail-closed)
  high     → ack degraded=True (advisory warn)
  medium   → ack degraded=True (advisory warn)
  low      → ack degraded=True (advisory warn)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class CvePattern:
    id: str
    match: re.Pattern[str]
    severity: Literal["critical", "high", "medium", "low"]
    cve_anchor: str  # CVE-XXXX-XXXX identifier or CWE label
    rationale: str


CVE_PATTERNS: tuple[CvePattern, ...] = (
    CvePattern(
        id="h-rm-rf-root",
        match=re.compile(r"\brm\s+-[rRf]+\s+\/(?![a-zA-Z0-9_])"),
        severity="critical",
        cve_anchor="CWE-78 (OS Command Injection); historical: shellshock-class",
        rationale="destructive recursive delete from filesystem root",
    ),
    CvePattern(
        id="h-curl-pipe-shell",
        match=re.compile(
            r"\b(curl|wget)\s+[^|;&\n]+\|\s*(sh|bash|zsh|fish|powershell)"
        ),
        severity="critical",
        cve_anchor="CWE-494 (Download of Code Without Integrity Check)",
        rationale="piping remote content directly into a shell interpreter (RCE)",
    ),
    CvePattern(
        id="h-ssh-key-exfil",
        match=re.compile(
            r"(?:cat|less|more|head|tail|read)\s+[^\n]*\.ssh/id_(rsa|ed25519|ecdsa|dsa)\b"
        ),
        severity="critical",
        cve_anchor="CWE-200 (Exposure of Sensitive Information)",
        rationale="reading SSH private key",
    ),
    CvePattern(
        id="h-sudo-nopasswd",
        match=re.compile(
            r"\bsudo\s+(?:-n\s+)?(?:visudo|tee\s+/etc/sudoers|sh\s+-c\s+[\"']?echo[^\"']*NOPASSWD)"
        ),
        severity="critical",
        cve_anchor="CWE-269 (Improper Privilege Management)",
        rationale="attempting to grant passwordless sudo",
    ),
    CvePattern(
        id="h-fork-bomb",
        match=re.compile(
            r":\s*\(\s*\)\s*\{[^}]*:\s*\|\s*:\s*&\s*[^}]*\}\s*;\s*:"
        ),
        severity="high",
        cve_anchor="CWE-400 (Resource Exhaustion)",
        rationale="classic fork-bomb pattern :(){:|:&};:",
    ),
)
