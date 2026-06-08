"""Manifest schema — dataclass + strict TOML parser/validator for engine.toml files.

Schema (all fields required unless marked optional):

  name         str        kebab-case engine identifier
  description  str        human-readable description
  version      str        semver string
  phases       list[str]  lifecycle phases the engine handles
  required     bool       fail-closed (True) vs. fail-open (False)
  budget_tier  str        one of "always" | "med-or-higher" | "high-only"
  runtime      str        optional; "python" (default) or "sidecar"
  concurrent_safe bool    optional; default False. Engines that declare True may
                          be dispatched in parallel with other concurrent_safe
                          engines for the same phase. Engines that mutate
                          shared in-process state (posteriors, LCS history,
                          ledgers, file writes) MUST leave this False.
  topics       table      subscribes/emits string lists

  # When runtime == "python" (or absent):
  adapter      str        Python entry-point notation: "module.path:attr"

  # When runtime == "sidecar":
  command          str       executable path to spawn for the sidecar
  args             list[str] optional argv tail (default [])
  env_allowlist    list[str] optional env keys forwarded to subprocess
                            (default ["PATH"])
  adapter_metadata table     optional dict[str, str] — informational only

  [topics]
  subscribes   list[str]  topics the engine subscribes to
  emits        list[str]  topics the engine may emit

  # Optional (any runtime):
  depends_on   list[str]  engine names that must load before this one
  tags         list[str]  free-form label list

  # Optional [agent] table (audit §8 — "agent-shaped delegations"):
  [agent]
  tier         str        task-class the tier-router understands —
                          one of "orchestrator" | "executor" | "validator"
  [agent.prompts]
  <phase>      str        repo-relative path to the engine-authored prompt
                          body for that phase, e.g. post-session = "prompts/drift.md"

  When present, the [agent] table marks the engine as "agent-shaped": its
  on_phase may call a model (the prompt body lives in the referenced .md files)
  instead of, or alongside, a deterministic algorithm.  The table is optional —
  most engines have no [agent] table and parse with agent=None, fully
  backward-compatible.
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

# These are required regardless of runtime.
_RUNTIME_AGNOSTIC_REQUIRED: frozenset[str] = frozenset(
    {"name", "description", "version", "phases", "required", "budget_tier", "topics"}
)

# Runtime-specific keys.
_PYTHON_REQUIRED: frozenset[str] = frozenset({"adapter"})
_PYTHON_FORBIDDEN: frozenset[str] = frozenset({"command", "args", "env_allowlist"})

_SIDECAR_REQUIRED: frozenset[str] = frozenset({"command"})
_SIDECAR_OPTIONAL: frozenset[str] = frozenset({"args", "env_allowlist", "adapter_metadata"})
_SIDECAR_FORBIDDEN: frozenset[str] = frozenset({"adapter"})

_OPTIONAL_FIELDS: frozenset[str] = frozenset(
    {"depends_on", "tags", "runtime", "adapter", "command", "args", "env_allowlist", "adapter_metadata", "concurrent_safe", "agent"}
)
_ALL_KNOWN_FIELDS: frozenset[str] = _RUNTIME_AGNOSTIC_REQUIRED | _OPTIONAL_FIELDS

_REQUIRED_TOPICS_FIELDS: frozenset[str] = frozenset({"subscribes", "emits"})

_VALID_BUDGET_TIERS: frozenset[str] = frozenset({"always", "med-or-higher", "high-only"})
_VALID_RUNTIMES: frozenset[str] = frozenset({"python", "sidecar"})

# [agent] table — strict shape.  `tier` is a tier-router task class; `prompts`
# maps lifecycle-phase name → repo-relative prompt path.
_REQUIRED_AGENT_FIELDS: frozenset[str] = frozenset({"tier", "prompts"})
_VALID_AGENT_TIERS: frozenset[str] = frozenset({"orchestrator", "executor", "validator"})

DEFAULT_SIDECAR_ENV_ALLOWLIST: tuple[str, ...] = ("PATH",)


# ──────────────────────────────────────────────────────────────────────────────
# Dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EngineTopics:
    subscribes: tuple[str, ...]
    emits: tuple[str, ...]


@dataclass(frozen=True)
class AgentSpec:
    """Resolved `[agent]` table — marks an engine as "agent-shaped" (audit §8).

    tier:
        Tier-router task class the engine's model call routes to —
        one of "orchestrator" | "executor" | "validator".
    prompts:
        Phase name → repo-relative path to the engine-authored prompt body.
        Stored as a sorted tuple of (phase, path) pairs so the dataclass stays
        frozen/hashable; use :meth:`prompt_for` for lookup.
    """

    tier: str
    prompts: tuple[tuple[str, str], ...]

    def prompt_for(self, phase: str) -> str | None:
        """Return the prompt path declared for *phase*, or None."""
        for p, path in self.prompts:
            if p == phase:
                return path
        return None


@dataclass(frozen=True)
class EngineManifest:
    """Parsed, validated engine.toml manifest."""

    name: str
    description: str
    version: str
    phases: tuple[str, ...]
    required: bool
    budget_tier: str
    topics: EngineTopics

    # Runtime selector — "python" (default) or "sidecar".
    runtime: str = "python"

    # runtime == "python" → adapter is set; command/args/env_allowlist are empty.
    # runtime == "sidecar" → adapter is empty; command/args/env_allowlist are set.
    adapter: str = ""
    command: str = ""
    args: tuple[str, ...] = field(default_factory=tuple)
    env_allowlist: tuple[str, ...] = field(default_factory=tuple)
    adapter_metadata: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    # Optional
    depends_on: tuple[str, ...] = field(default_factory=tuple)
    tags: tuple[str, ...] = field(default_factory=tuple)

    # Optional [agent] table (audit §8).  None → not agent-shaped (the common
    # case); set → the engine may back a phase with a model call.
    agent: AgentSpec | None = None

    # Wave 13.3 — per-engine opt-in for concurrent dispatch. Default False
    # preserves the historical (post-Wave-13.3: serial) ordering for engines
    # that mutate shared state.
    concurrent_safe: bool = False

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


def _parse_agent_table(data: dict[str, Any], path: str) -> AgentSpec | None:
    """Validate and parse the optional ``[agent]`` table.

    Returns None when absent.  When present, validates strictly:
      * the table is a dict with exactly {tier, prompts}
      * ``tier`` is one of the known tier-router task classes
      * ``prompts`` is a table of phase(str) → path(str), non-empty
    """
    if "agent" not in data:
        return None

    raw_agent = data["agent"]
    if not isinstance(raw_agent, dict):
        raise ManifestSchemaError(
            "field 'agent' must be a TOML table",
            field="agent",
            manifest_path=path,
        )

    agent_unknown = set(raw_agent.keys()) - _REQUIRED_AGENT_FIELDS
    if agent_unknown:
        first = sorted(agent_unknown)[0]
        raise ManifestSchemaError(
            f"unknown field(s) in [agent]: {sorted(agent_unknown)!r}",
            field=f"agent.{first}",
            manifest_path=path,
        )

    agent_missing = _REQUIRED_AGENT_FIELDS - set(raw_agent.keys())
    if agent_missing:
        first = sorted(agent_missing)[0]
        raise ManifestSchemaError(
            f"missing required field(s) in [agent]: {sorted(agent_missing)!r}",
            field=f"agent.{first}",
            manifest_path=path,
        )

    tier = raw_agent["tier"]
    if not isinstance(tier, str):
        raise ManifestSchemaError(
            f"field 'agent.tier' must be a string, got {type(tier).__name__}",
            field="agent.tier",
            manifest_path=path,
        )
    if tier not in _VALID_AGENT_TIERS:
        raise ManifestSchemaError(
            f"agent.tier {tier!r} is not one of {sorted(_VALID_AGENT_TIERS)}",
            field="agent.tier",
            manifest_path=path,
        )

    raw_prompts = raw_agent["prompts"]
    if not isinstance(raw_prompts, dict):
        raise ManifestSchemaError(
            "field 'agent.prompts' must be a TOML table of phase→path",
            field="agent.prompts",
            manifest_path=path,
        )
    if not raw_prompts:
        raise ManifestSchemaError(
            "field 'agent.prompts' must declare at least one phase→path mapping",
            field="agent.prompts",
            manifest_path=path,
        )

    prompt_items: list[tuple[str, str]] = []
    for phase_key, prompt_path in raw_prompts.items():
        if not isinstance(phase_key, str) or not isinstance(prompt_path, str):
            raise ManifestSchemaError(
                "field 'agent.prompts' must be a TOML table of string→string (phase→path)",
                field="agent.prompts",
                manifest_path=path,
            )
        if not prompt_path:
            raise ManifestSchemaError(
                f"field 'agent.prompts.{phase_key}' must be a non-empty path",
                field=f"agent.prompts.{phase_key}",
                manifest_path=path,
            )
        prompt_items.append((phase_key, prompt_path))

    # Sort for deterministic, hashable ordering.
    return AgentSpec(tier=tier, prompts=tuple(sorted(prompt_items)))


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

    # ── Runtime-agnostic required presence ────────────────────────────────────
    missing = _RUNTIME_AGNOSTIC_REQUIRED - top_level_keys
    if missing:
        first = sorted(missing)[0]
        raise ManifestSchemaError(
            f"missing required field(s): {sorted(missing)!r}",
            field=first,
            manifest_path=path_str,
        )

    # ── Type validation (runtime-agnostic) ────────────────────────────────────
    name = _require_str(data, "name", path_str)
    description = _require_str(data, "description", path_str)
    version = _require_str(data, "version", path_str)
    phases = _require_str_list(data, "phases", path_str)
    required = _require_bool(data, "required", path_str)
    budget_tier = _require_str(data, "budget_tier", path_str)

    if budget_tier not in _VALID_BUDGET_TIERS:
        raise ManifestSchemaError(
            f"budget_tier {budget_tier!r} is not one of {sorted(_VALID_BUDGET_TIERS)}",
            field="budget_tier",
            manifest_path=path_str,
        )

    # ── Runtime branch ────────────────────────────────────────────────────────
    runtime: str = "python"
    if "runtime" in data:
        runtime = _require_str(data, "runtime", path_str)
        if runtime not in _VALID_RUNTIMES:
            raise ManifestSchemaError(
                f"runtime {runtime!r} is not one of {sorted(_VALID_RUNTIMES)}",
                field="runtime",
                manifest_path=path_str,
            )

    adapter: str = ""
    command: str = ""
    args_tuple: tuple[str, ...] = ()
    env_allowlist: tuple[str, ...] = ()
    adapter_metadata_items: tuple[tuple[str, str], ...] = ()

    if runtime == "python":
        # Required: adapter. Forbidden: command, args, env_allowlist.
        if "adapter" not in top_level_keys:
            raise ManifestSchemaError(
                "missing required field(s): ['adapter'] for runtime='python'",
                field="adapter",
                manifest_path=path_str,
            )
        forbidden_present = _PYTHON_FORBIDDEN & top_level_keys
        if forbidden_present:
            first = sorted(forbidden_present)[0]
            raise ManifestSchemaError(
                f"field(s) {sorted(forbidden_present)!r} are not allowed when runtime='python'",
                field=first,
                manifest_path=path_str,
            )
        if "adapter_metadata" in top_level_keys:
            raise ManifestSchemaError(
                "field 'adapter_metadata' is not allowed when runtime='python'",
                field="adapter_metadata",
                manifest_path=path_str,
            )
        adapter = _require_str(data, "adapter", path_str)
        if ":" not in adapter:
            raise ManifestSchemaError(
                f"adapter {adapter!r} must use 'module.path:attribute' notation",
                field="adapter",
                manifest_path=path_str,
            )
    else:  # runtime == "sidecar"
        if "command" not in top_level_keys:
            raise ManifestSchemaError(
                "missing required field(s): ['command'] for runtime='sidecar'",
                field="command",
                manifest_path=path_str,
            )
        forbidden_present = _SIDECAR_FORBIDDEN & top_level_keys
        if forbidden_present:
            first = sorted(forbidden_present)[0]
            raise ManifestSchemaError(
                f"field(s) {sorted(forbidden_present)!r} are not allowed when runtime='sidecar'",
                field=first,
                manifest_path=path_str,
            )
        command = _require_str(data, "command", path_str)
        if not command:
            raise ManifestSchemaError(
                "field 'command' must be a non-empty string",
                field="command",
                manifest_path=path_str,
            )
        args_tuple = _optional_str_list(data, "args", path_str)
        if "env_allowlist" in data:
            env_allowlist = _require_str_list(data, "env_allowlist", path_str)
        else:
            env_allowlist = DEFAULT_SIDECAR_ENV_ALLOWLIST
        if "adapter_metadata" in data:
            raw_meta = data["adapter_metadata"]
            if not isinstance(raw_meta, dict):
                raise ManifestSchemaError(
                    "field 'adapter_metadata' must be a TOML table of string→string",
                    field="adapter_metadata",
                    manifest_path=path_str,
                )
            items: list[tuple[str, str]] = []
            for k, v in raw_meta.items():
                if not isinstance(k, str) or not isinstance(v, str):
                    raise ManifestSchemaError(
                        "field 'adapter_metadata' must be a TOML table of string→string",
                        field="adapter_metadata",
                        manifest_path=path_str,
                    )
                items.append((k, v))
            adapter_metadata_items = tuple(items)

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

    # concurrent_safe — optional bool; absent → False.
    concurrent_safe: bool = False
    if "concurrent_safe" in data:
        concurrent_safe = _require_bool(data, "concurrent_safe", path_str)

    # [agent] table — optional; None when absent.
    agent = _parse_agent_table(data, path_str)

    return EngineManifest(
        name=name,
        description=description,
        version=version,
        phases=phases,
        required=required,
        budget_tier=budget_tier,
        topics=topics,
        runtime=runtime,
        adapter=adapter,
        command=command,
        args=args_tuple,
        env_allowlist=env_allowlist,
        adapter_metadata=adapter_metadata_items,
        depends_on=depends_on,
        tags=tags,
        concurrent_safe=concurrent_safe,
        agent=agent,
        manifest_path=path_str,
    )
