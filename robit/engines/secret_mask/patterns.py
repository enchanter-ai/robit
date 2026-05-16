"""Secret-mask pattern table — port of `SECRET_PATTERNS_V0_1` from
`client/enchanter/src/plugins/hydra/cve-patterns.ts`.

Each entry is a frozen dataclass holding a compiled regex and the
redaction string that replaces every match in the scanned corpus.
Redaction strings are kept verbatim from the TS source.

Note on TS global flag: TypeScript regexes carry a `/g` flag so
`String.prototype.replace` replaces all occurrences. Python's `re.sub`
replaces all occurrences by default, so the compiled patterns here use
no special flag for that.  The DOTALL flag is required for the PEM
block pattern to match across newlines.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SecretPattern:
    id: str
    name: str
    match: re.Pattern[str]
    redaction: str


SECRET_PATTERNS: tuple[SecretPattern, ...] = (
    SecretPattern(
        id="s-aws-key",
        name="AWS access key",
        match=re.compile(r"\b(AKIA[0-9A-Z]{16})\b"),
        redaction="AKIA****[REDACTED]",
    ),
    SecretPattern(
        id="s-bearer-token",
        name="bearer token in Authorization header",
        match=re.compile(r"(Bearer\s+)([A-Za-z0-9._\-]{20,})"),
        redaction=r"\1[REDACTED]",
    ),
    SecretPattern(
        id="s-pem-private-key",
        name="PEM private key",
        match=re.compile(
            r"-----BEGIN\s+(?:RSA\s+|EC\s+|OPENSSH\s+)?PRIVATE KEY-----[\s\S]+?-----END[^-]+-----",
            re.DOTALL,
        ),
        redaction="[REDACTED PRIVATE KEY]",
    ),
    SecretPattern(
        id="s-anthropic-key",
        name="Anthropic API key",
        match=re.compile(r"\b(sk-ant-[A-Za-z0-9_\-]{20,})\b"),
        redaction="sk-ant-****[REDACTED]",
    ),
    SecretPattern(
        id="s-openai-key",
        name="OpenAI API key",
        match=re.compile(r"\b(sk-[A-Za-z0-9]{32,})\b"),
        redaction="sk-****[REDACTED]",
    ),
)
