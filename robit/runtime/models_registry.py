"""robit.runtime.models_registry — typed wrapper around the wixie models-registry.json.

The registry ships as a data file at enchanter/runtime/data/models-registry.json (a
copy of wixie/shared/models-registry.json).  All consumers load via ModelsRegistry.load()
so they never depend on a sibling-repo path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Default path: relative to this source file so it works regardless of cwd.
_DEFAULT_REGISTRY = Path(__file__).parent / "data" / "models-registry.json"


class UnknownModelError(KeyError):
    """Raised when model_id is not present in the registry."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        super().__init__(f"Model '{model_id}' not found in registry")


class UnknownFamilyError(KeyError):
    """Raised when no models match the requested family name."""

    def __init__(self, family: str) -> None:
        self.family = family
        super().__init__(f"Family '{family}' not found in registry")


@dataclass(frozen=True)
class ModelEntry:
    """A single entry from the models registry.

    Explicitly modelled fields mirror the canonical keys present in every entry.
    Any additional keys the registry may add in future are captured verbatim in
    ``extras`` so callers never silently lose data.
    """

    model_id: str
    family: str
    display_name: str
    context_window: int
    format: str
    reasoning: str
    cot_approach: str
    few_shot: str
    key_constraint: str
    # The registry does not have a "tier" field — tier is determined by the
    # TierRouter based on model_id / family patterns.
    tier: str | None = None
    last_updated: str | None = None
    # Catch-all for future registry fields.
    extras: dict[str, Any] = field(default_factory=dict)

    # Convenience: make the entry usable as a plain dict when needed.
    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "family": self.family,
            "display_name": self.display_name,
            "context_window": self.context_window,
            "format": self.format,
            "reasoning": self.reasoning,
            "cot_approach": self.cot_approach,
            "few_shot": self.few_shot,
            "key_constraint": self.key_constraint,
            "tier": self.tier,
            "last_updated": self.last_updated,
            **self.extras,
        }


_KNOWN_FIELDS = frozenset(
    {
        "family",
        "display_name",
        "context_window",
        "format",
        "reasoning",
        "cot_approach",
        "few_shot",
        "key_constraint",
        "tier",
        "last_updated",
    }
)


def _parse_entry(model_id: str, raw: dict[str, Any]) -> ModelEntry:
    extras = {k: v for k, v in raw.items() if k not in _KNOWN_FIELDS}
    raw_cw = raw.get("context_window")
    context_window = int(raw_cw) if raw_cw is not None else 0
    return ModelEntry(
        model_id=model_id,
        family=raw.get("family", ""),
        display_name=raw.get("display_name", ""),
        context_window=context_window,
        format=raw.get("format", ""),
        reasoning=raw.get("reasoning", ""),
        cot_approach=raw.get("cot_approach", ""),
        few_shot=raw.get("few_shot", ""),
        key_constraint=raw.get("key_constraint", ""),
        tier=raw.get("tier"),
        last_updated=raw.get("last_updated"),
        extras=extras,
    )


class ModelsRegistry:
    """Queryable wrapper around the flat models dictionary in models-registry.json."""

    def __init__(self, entries: dict[str, ModelEntry], meta: dict[str, Any]) -> None:
        self._entries = entries
        self._meta = meta  # top-level keys like last_updated, model_count

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: Path | None = None) -> "ModelsRegistry":
        """Load the registry from *path*.

        Defaults to the bundled ``enchanter/runtime/data/models-registry.json``.
        Raises ``ValueError`` (with the path in the message) on malformed JSON.
        """
        resolved = Path(path) if path is not None else _DEFAULT_REGISTRY
        try:
            raw_text = resolved.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise ValueError(f"Registry file not found: {resolved}")

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSON in registry file '{resolved}': {exc}") from exc

        if "models" not in data or not isinstance(data["models"], dict):
            raise ValueError(
                f"Registry file '{resolved}' is missing a top-level 'models' dict"
            )

        entries = {
            model_id: _parse_entry(model_id, raw)
            for model_id, raw in data["models"].items()
        }
        meta = {k: v for k, v in data.items() if k != "models"}
        return cls(entries, meta)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def model(self, model_id: str) -> ModelEntry:
        """Return the entry for *model_id*.

        Raises ``UnknownModelError`` if not found.
        """
        try:
            return self._entries[model_id]
        except KeyError:
            raise UnknownModelError(model_id)

    def family(self, family_name: str) -> list[ModelEntry]:
        """Return all entries whose ``family`` field matches *family_name* (exact, case-sensitive).

        Raises ``UnknownFamilyError`` if no matches.
        """
        results = [e for e in self._entries.values() if e.family == family_name]
        if not results:
            raise UnknownFamilyError(family_name)
        return results

    def latest_in_family(self, family: str) -> ModelEntry:
        """Return the single entry with the lexicographically greatest model_id in *family*.

        The registry uses a ``<name>-<major>-<minor>`` naming convention, so
        lexicographic ordering reliably yields the latest version within a family
        (e.g., ``claude-opus-4-7`` > ``claude-opus-4-6``).

        Raises ``UnknownFamilyError`` if the family is empty.
        """
        members = self.family(family)  # raises UnknownFamilyError if empty
        return max(members, key=lambda e: e.model_id)

    def all(self) -> list[ModelEntry]:
        """Return all entries in insertion order."""
        return list(self._entries.values())

    # ------------------------------------------------------------------
    # Meta
    # ------------------------------------------------------------------

    @property
    def last_updated(self) -> str | None:
        return self._meta.get("last_updated")

    @property
    def model_count(self) -> int:
        return len(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return (
            f"ModelsRegistry(models={len(self._entries)}, "
            f"last_updated={self.last_updated!r})"
        )
