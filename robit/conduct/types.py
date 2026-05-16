"""ConductRule dataclass — the structured output of the conduct loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


EnforcementMode = Literal["code", "prompt", "hybrid"]


@dataclass(frozen=True)
class ConductRule:
    """A single conduct module parsed from a Markdown file.

    Attributes:
        name:        Slug identifier, e.g. ``"discipline"``.
        path:        Absolute path to the source ``.md`` file.
        body:        Full Markdown content *without* frontmatter.
        enforcement: How the rule is applied: ``"prompt"`` (inject into
                     system prompt), ``"code"`` (runtime guard), or
                     ``"hybrid"`` (both).  Defaults to ``"prompt"`` when
                     not declared in frontmatter.
        package:     Parent package name derived from the path, e.g.
                     ``"core"``, ``"web"``, ``"cost"``.
        tags:        Tuple of string tags from frontmatter; empty when not
                     declared.
    """

    name: str
    path: Path
    body: str
    enforcement: EnforcementMode = "prompt"
    package: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)


class ConductFrontmatterError(ValueError):
    """Raised when frontmatter in a conduct file cannot be parsed."""

    def __init__(self, path: Path, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"Frontmatter parse error in {path}: {reason}")
