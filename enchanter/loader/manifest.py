"""Manifest schema — dataclass + strict TOML parser/validator for engine.toml files.

Schema (all fields required unless marked optional):

  name         str        kebab-case engine identifier
  description  str        human-readable description
  version      str        semver string
  phases       list[str]  lifecycle phases the engine handles
  required     bool       fail-closed (True) vs. fail-open (False)
  budget_tier  str        one of "always" | "med-or-higher" | "high-only"
  adapter      str        Python entry-point notation: "module.path:attr"

  [topics]
  subscribes   list[str]  topics the engine subscribes to
  emits        list[str]  topics the engine may emit

  # Optional:
  depends_on   list[str]  engine names that must load before this one
  tags         list[str]  free-form label list
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import ManifestSchemaError

# ──────────────────────────────────────────────────────────────────────────────
# Known fields (strict mode — extras are rejected)
# ──────────────────────────────────────────────────────────────────────────────

_REQUIRED_FIELDS: frozenset[str] = frozenset(
    {"name", "description", "version", "phases", "required", "budget_tier", "adapter", "topics"}
)
_OPTIONAL_FIELDS: frozenset[str] = frozenset({"depends_on", "tags"})
_ALL_KNOWN_FIELDS: frozenset[str] = _REQUIRED_FIELDS | _OPTIONAL_FIELDS

_REQUIRED_TOPICS_FIELDS: frozenset[str] = frozenset({"subscribes", "emits"})

_VALID_BUDGET_TIERS: frozenset[str] = frozenset({"always", "med-or-higher", "high-only"})


# ──────────────────────────────────────────────────────────────────────────────
# Dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EngineTopics:
    subscribes: tuple[str, ...]
    emits: tuple[str, ...]


@dataclass(frozen=True)
class EngineManifest:
    """Parsed, validated engine.toml manifest."""

    name: str
    description: str
    version: str
    phases: tuple[str, ...]
    required: bool
    budget_tier: str
    adapter: str
    topics: EngineTopics

    # Optional
    depends_on: tuple[str, ...] = field(default_factory=tuple)
    tags: tuple[str, ...] = field(default_factory=tuple)

    # Source path (not part of TOML; set by the loader)
    manifest_path: str = field(default="", compare=False, hash=False)


# ──────────────────────────────────────────────────────────────────────────────
# Parser / validator
# ──────────────────────────────────────────────────────────────────────────────

def _require_str(data: dict[str, Any], key: str, path: str) -> str:
    val = data[key]
    if not isinstance(val, str):
        raise ManifestSchemaError(
            f"field {key!r} must be a string, got {type(val).__name__}",
            field=key,
            manifest_path=path,
        )
    return val


def _require_bool(data: dict[str, Any], key: str, path: str) -> bool:
    val = data[key]
    if not isinstance(val, bool):
        raise ManifestSchemaError(
            f"field {key!r} must be a boolean, got {type(val).__name__}",
            field=key,
            manifest_path=path,
        )
    return val


def _require_str_list(data: dict[str, Any], key: str, path: str) -> tuple[str, ...]:
    val = data[key]
    if not isinstance(val, list) or not all(isinstance(s, str) for s in val):
        raise ManifestSchemaError(
            f"field {key!r} must be a list of strings",
            field=key,
            manifest_path=path,
        )
    return tuple(val)


def _optional_str_list(
    data: dict[str, Any], key: str, path: str
) -> tuple[str, ...]:
    if key not in data:
        return ()
    return _require_str_list(data, key, path)


def parse_manifest(toml_path: Path) -> EngineManifest:
    """Parse and strictly validate an engine.toml file.

    Raises:
        ManifestSchemaError: Any required field is missing, any field has the
                             wrong type, or any unknown field is present.
        FileNotFoundError:   *toml_path* does not exist.
        tomllib.TOMLDecodeError: The file is not valid TOML.
    """
    path_str = str(toml_path)

    with open(toml_path, "rb") as fh:
        data: dict[str, Any] = tomllib.load(fh)

    # ── Strict unknown-field check ────────────────────────────────────────────
    top_level_keys = set(data.keys())
    unknown = top_level_keys - _ALL_KNOWN_FIELDS
    if unknown:
        first = sorted(unknown)[0]
        raise ManifestSchemaError(
            f"unknown field(s) in manifest: {sorted(unknown)!r}",
            field=first,
            manifest_path=path_str,
        )

    # ── Required field presence ───────────────────────────────────────────────
    missing = _REQUIRED_FIELDS - top_level_keys
    if missing:
        first = sorted(missing)[0]
        raise ManifestSchemaError(
            f"missing required field(s): {sorted(missing)!r}",
            field=first,
            manifest_path=path_str,
        )

    # ── Type validation ───────────────────────────────────────────────────────
    name = _require_str(data, "name", path_str)
    description = _require_str(data, "description", path_str)
    version = _require_str(data, "version", path_str)
    phases = _require_str_list(data, "phases", path_str)
    required = _require_bool(data, "required", path_str)
    budget_tier = _require_str(data, "budget_tier", path_str)
    adapter = _require_str(data, "adapter", path_str)

    if budget_tier not in _VALID_BUDGET_TIERS:
        raise ManifestSchemaError(
            f"budget_tier {budget_tier!r} is not one of {sorted(_VALID_BUDGET_TIERS)}",
            field="budget_tier",
            manifest_path=path_str,
        )

    if ":" not in adapter:
        raise ManifestSchemaError(
            f"adapter {adapter!r} must use 'module.path:attribute' notation",
            field="adapter",
            manifest_path=path_str,
        )

    # ── [topics] sub-table ────────────────────────────────────────────────────
    raw_topics = data["topics"]
    if not isinstance(raw_topics, dict):
        raise ManifestSchemaError(
            "field 'topics' must be a TOML table",
            field="topics",
            manifest_path=path_str,
        )

    topics_unknown = set(raw_topics.keys()) - _REQUIRED_TOPICS_FIELDS
    if topics_unknown:
        first = sorted(topics_unknown)[0]
        raise ManifestSchemaError(
            f"unknown field(s) in [topics]: {sorted(topics_unknown)!r}",
            field=f"topics.{first}",
            manifest_path=path_str,
        )

    topics_missing = _REQUIRED_TOPICS_FIELDS - set(raw_topics.keys())
    if topics_missing:
        first = sorted(topics_missing)[0]
        raise ManifestSchemaError(
            f"missing required field(s) in [topics]: {sorted(topics_missing)!r}",
            field=f"topics.{first}",
            manifest_path=path_str,
        )

    subscribes = _require_str_list(raw_topics, "subscribes", path_str)
    emits = _require_str_list(raw_topics, "emits", path_str)

    topics = EngineTopics(subscribes=subscribes, emits=emits)

    # ── Optional fields ───────────────────────────────────────────────────────
    depends_on = _optional_str_list(data, "depends_on", path_str)
    tags = _optional_str_list(data, "tags", path_str)

    return EngineManifest(
        name=name,
        description=description,
        version=version,
        phases=phases,
        required=required,
        budget_tier=budget_tier,
        adapter=adapter,
        topics=topics,
        depends_on=depends_on,
        tags=tags,
        manifest_path=path_str,
    )
