"""Tool-poisoning suspicion-pattern table — port of lich.adapter.ts PATTERNS + SuspicionPattern.

Five pattern categories (P1–P5) directly from the TS source. Each pattern carries:
  - id:        unique string, format `P<N>:<slug>`
  - match:     compiled regex (IGNORECASE where the TS uses /i flag)
  - severity:  integer weight (1 or 2) contributing toward VETO_THRESHOLD (3)
  - rationale: human-readable label surfaced in veto/warn reasons

Severity routing (mirrors the TS adapter):
  suspicion_score >= VETO_THRESHOLD (3) → veto
  0 < suspicion_score < VETO_THRESHOLD → ack with degraded=True (advisory warn)
  0                                     → clean ack

M6-simplified false-positive (FP) EMA tracking: patterns with EMA FP rate > 0.5
receive 50% effective-severity downweight. FP state is per-instance (per-engine).
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SuspicionPattern:
    id: str
    match: re.Pattern[str]
    severity: int  # raw weight (1 or 2); effective_severity may be lower after FP downweight
    rationale: str


# VETO_THRESHOLD: suspicion score at or above this fires a veto.
# [author judgment] threshold=3 matches the TS adapter.  A single P1/P2/P4/P5 hit
# (severity 2) alone does not veto; two hits do.  A lone P3 hit (severity 1) never
# vetoes alone, which is correct — suspicious TLDs are ambiguous without corroboration.
VETO_THRESHOLD: int = 3

SUSPICION_PATTERNS: tuple[SuspicionPattern, ...] = (
    SuspicionPattern(
        # P1: Imperative override verbs at the start of a description field.
        # Ported from TS: /(?:^|\n)\s*(?:MUST|IGNORE|OVERRIDE|DISREGARD|BYPASS)\b/i
        id="P1:imperative-override",
        match=re.compile(
            r"(?:^|\n)\s*(?:MUST|IGNORE|OVERRIDE|DISREGARD|BYPASS)\b",
            re.IGNORECASE,
        ),
        severity=2,
        rationale="imperative override verb at the start of a description field",
    ),
    SuspicionPattern(
        # P2: Credential-request phrases in parameter descriptions.
        # Ported from TS: /\b(?:api[\s_-]?key|password|secret|token)\b/i
        id="P2:credential-request",
        match=re.compile(
            r"\b(?:api[\s_\-]?key|password|secret|token)\b",
            re.IGNORECASE,
        ),
        severity=2,
        rationale="credential-request phrase in tool schema",
    ),
    SuspicionPattern(
        # P3: Suspicious TLD in URL-shaped strings or raw IP addresses.
        # Ported from TS: /https?:\/\/[^\s"']+\.(?:tk|ml|cf|gq|ga)\b|https?:\/\/\d{1,3}(?:\.\d{1,3}){3}/i
        id="P3:suspicious-url",
        match=re.compile(
            r"https?://[^\s\"']+\.(?:tk|ml|cf|gq|ga)\b"
            r"|https?://\d{1,3}(?:\.\d{1,3}){3}",
            re.IGNORECASE,
        ),
        severity=1,
        rationale="URL with suspicious free TLD or raw IP address in tool schema",
    ),
    SuspicionPattern(
        # P4: Base64-encoded payloads > 100 chars in description fields.
        # Ported from TS: /[A-Za-z0-9+/]{100,}={0,2}/
        id="P4:base64-payload",
        match=re.compile(r"[A-Za-z0-9+/]{100,}={0,2}"),
        severity=2,
        rationale="long base64-shaped payload (>100 chars) in tool schema",
    ),
    SuspicionPattern(
        # P5: Hidden Unicode — zero-width chars or RTL override codepoints.
        # Ported from TS: /[​-‏‪-‮⁠-⁤﻿]/
        # Unicode codepoints: U+200B–U+200F, U+202A–U+202E, U+2060–U+2064, U+FEFF
        id="P5:hidden-unicode",
        match=re.compile(
            r"[​-‏‪-‮⁠-⁤﻿]"
        ),
        severity=2,
        rationale="hidden unicode (zero-width or RTL-override codepoints) in tool schema",
    ),
)
