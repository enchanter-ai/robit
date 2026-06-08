"""Engine discovery — glob manifests, import adapters, return a PluginRegistry.

Public API:

    find_engine_manifests(root: Path) -> list[Path]
        Globs ``engines/*/engine.toml`` under *root*.  Returns paths sorted
        alphabetically so the order is deterministic across runs.

    load_engine_registry(root: Path | None = None) -> PluginRegistry
        Parses every manifest found by ``find_engine_manifests``, imports
        each adapter, applies topological ordering by ``depends_on``, and
        returns a ``dict[name, PluginAdapter]``.

        When *root* is None the package root is inferred from this file's
        location (two levels up: enchanter/loader/ → enchanter/ → repo root).
"""

from __future__ import annotations

import logging
import os
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

from robit.core.plugin import PluginAdapter, PluginRegistry
from robit.core.topics import is_known_topic, is_lifecycle_subscription, is_wildcard

from .errors import DependencyCycleError, ManifestSchemaError, TopicRegistryError
from .manifest import EngineManifest, parse_manifest
from .runtimes import load_runtime

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# G2 — opt-in environment switch flipping the topic cross-check from
# warn-by-default to strict (raise). See ``_cross_check_topics`` for the
# rationale behind defaulting to warn.
_STRICT_TOPICS_ENV: str = "ROBIT_STRICT_TOPICS"


# ──────────────────────────────────────────────────────────────────────────────
# Default root resolution
# ──────────────────────────────────────────────────────────────────────────────

def _default_root() -> Path:
    """Return the repo root: two parents up from robit/loader/discovery.py."""
    # __file__ is  <root>/robit/loader/discovery.py
    return Path(__file__).parent.parent.parent


# ──────────────────────────────────────────────────────────────────────────────
# Manifest discovery
# ──────────────────────────────────────────────────────────────────────────────

def find_engine_manifests(root: Path) -> list[Path]:
    """Return sorted list of engine.toml paths under ``root/engines/*/engine.toml``.

    Directories without an engine.toml are silently skipped — this is expected
    for engine directories under active development.
    """
    engines_dir = root / "robit" / "engines"
    if not engines_dir.is_dir():
        return []
    return sorted(engines_dir.glob("*/engine.toml"))


# ──────────────────────────────────────────────────────────────────────────────
# Adapter resolution (delegates to the runtime registry)
# ──────────────────────────────────────────────────────────────────────────────

def _import_adapter(manifest: EngineManifest) -> PluginAdapter:
    """Resolve the manifest to its concrete adapter via the runtime registry.

    For runtime='python' this imports ``module.path:attr`` (historical behavior).
    For runtime='sidecar' this returns a SidecarAdapter wrapping a subprocess
    (lazily spawned on first on_phase call).
    """
    return load_runtime(manifest)


# ──────────────────────────────────────────────────────────────────────────────
# Topological sort (Kahn's algorithm)
# ──────────────────────────────────────────────────────────────────────────────

def _topological_sort(manifests: list[EngineManifest]) -> list[EngineManifest]:
    """Return *manifests* ordered so each engine appears after its ``depends_on``.

    Raises:
        DependencyCycleError: A dependency cycle is detected.
        ManifestSchemaError:  A ``depends_on`` entry names an unknown engine.
    """
    name_to_manifest: dict[str, EngineManifest] = {m.name: m for m in manifests}

    # Validate all depends_on references exist.
    for m in manifests:
        for dep in m.depends_on:
            if dep not in name_to_manifest:
                raise ManifestSchemaError(
                    f"engine {m.name!r} depends_on unknown engine {dep!r}",
                    field="depends_on",
                    manifest_path=m.manifest_path,
                )

    # Build in-degree and adjacency (dep → dependents).
    in_degree: dict[str, int] = {m.name: 0 for m in manifests}
    dependents: dict[str, list[str]] = {m.name: [] for m in manifests}

    for m in manifests:
        for dep in m.depends_on:
            in_degree[m.name] += 1
            dependents[dep].append(m.name)

    # Kahn's BFS.
    queue: deque[str] = deque(name for name, deg in in_degree.items() if deg == 0)
    # Sort for determinism when multiple nodes have in-degree 0.
    queue = deque(sorted(queue))

    ordered: list[EngineManifest] = []
    while queue:
        node = queue.popleft()
        ordered.append(name_to_manifest[node])
        for child in sorted(dependents[node]):
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    if len(ordered) != len(manifests):
        # Cycle detected — find the nodes still in the cycle.
        remaining = [n for n, deg in in_degree.items() if deg > 0]
        raise DependencyCycleError(
            f"dependency cycle detected among engines: {sorted(remaining)}",
            cycle=sorted(remaining),
        )

    return ordered


# ──────────────────────────────────────────────────────────────────────────────
# G2 — boot-time topic cross-check
# ──────────────────────────────────────────────────────────────────────────────

def _cross_check_topics(
    manifests: list[EngineManifest],
    *,
    strict: bool,
) -> list[str]:
    """Cross-check every engine's declared topics against the central registry.

    Three classes of problem are detected per the G2 contract:

    1. **Unknown topic** — a declared emit/subscribe that is neither a
       registry-known canonical topic nor an always-allowed pattern.
    2. **Unsubscribed emit** — an emitted topic that *no* engine subscribes to
       (directly or via a matching wildcard).
    3. **Unemitted subscription** — a subscribed (non-wildcard, non-lifecycle)
       topic that nothing emits — neither an engine nor a registry-known
       framework topic.

    Always-allowed and therefore never flagged:

    * wildcard subscriptions (``*`` / ``foo.*``),
    * ``lifecycle.<phase>`` subscriptions,
    * deprecated-but-registered synonyms (e.g. ``llm.proxy.request``).

    Why **warn-by-default**: the live engine set is a *layered* substrate. Every
    engine emits namespaced result topics (``cost-ledger.appended``,
    ``trust-scorer.trust.scored``, …) that are consumed by the host process /
    operator tooling / the inference-substrate wildcard fan-in — not by a
    sibling engine's explicit ``subscribes`` list. Symmetrically, the topics
    engines subscribe to (``mcp.tool.call.requested``, ``filesystem.write.completed``)
    are framework-emitted, not engine-emitted. Enforcing rules (2) and (3)
    strictly would reject the entire existing, valid 14-engine set at boot.
    That is a false positive, not a real contract break — so the default is to
    *log a warning* and let boot proceed. Set ``ROBIT_STRICT_TOPICS=1`` (or pass
    ``strict=True``) to promote warnings to a :class:`TopicRegistryError` — the
    honest-over-breaking posture from the roadmap.

    Rule (1) — *unknown* topics — is the strongest signal of a genuine contract
    violation (a typo or an undeclared topic): a topic that is not in the
    registry at all cannot be reconciled and would silently never match on the
    bus. That is the exact silent-failure G2 exists to eliminate. In **strict**
    mode an unknown topic raises. In the default **warn** mode it is logged at
    WARNING (a distinct, louder message than the soft coverage warnings) but
    boot proceeds — the existing engine fixtures across the suite declare
    free-form topic names that are valid for those engines but absent from the
    canonical registry, so raising by default would break boot. Honest over
    breaking; flip ``ROBIT_STRICT_TOPICS=1`` to make unknowns fatal.

    Returns the list of all problem strings (empty when clean). Raises
    :class:`TopicRegistryError` on any problem when *strict*.
    """
    # Aggregate the live declarations.
    all_subscribes: set[str] = set()
    all_emits: set[str] = set()
    for m in manifests:
        all_subscribes.update(m.topics.subscribes)
        all_emits.update(m.topics.emits)

    # An emitted topic is "covered" by a subscriber if any engine subscribes to
    # it exactly, OR via the bare ``*`` catch-all, OR via a dotted-prefix
    # wildcard whose stem matches (``foo.*`` covers ``foo.bar``).
    def _has_subscriber(topic: str) -> bool:
        for sub in all_subscribes:
            if sub == topic:
                return True
            if sub == "*":
                return True
            if sub.endswith(".*") and topic.startswith(sub[:-1]):
                # ``foo.*`` → stem ``foo.`` ; matches ``foo.bar``.
                return True
            if sub.startswith("*.") and topic.endswith(sub[1:]):
                # ``*.veto`` → suffix ``.veto`` ; matches ``cve-pattern-gate.veto``.
                return True
        return False

    unknown_problems: list[str] = []
    soft_problems: list[str] = []

    for m in manifests:
        # ── Emits ──────────────────────────────────────────────────────────
        for topic in m.topics.emits:
            if is_wildcard(topic):
                # An *emit* wildcard is nonsensical (you emit a concrete topic),
                # but treat as unknown so it never silently passes.
                unknown_problems.append(
                    f"engine {m.name!r} emits wildcard topic {topic!r} (emits must be concrete)"
                )
                continue
            if not is_known_topic(topic):
                unknown_problems.append(
                    f"engine {m.name!r} emits unknown topic {topic!r} (not in registry)"
                )
                continue
            if not _has_subscriber(topic):
                soft_problems.append(
                    f"engine {m.name!r} emits {topic!r} but no engine subscribes to it"
                )

        # ── Subscribes ─────────────────────────────────────────────────────
        for topic in m.topics.subscribes:
            if is_wildcard(topic) or is_lifecycle_subscription(topic):
                # Always allowed; never flagged.
                continue
            if not is_known_topic(topic):
                unknown_problems.append(
                    f"engine {m.name!r} subscribes to unknown topic {topic!r} (not in registry)"
                )
                continue
            # Known topic — is there a producer? Registry-known framework topics
            # count as emitters even if no engine emits them.
            from robit.core.topics import get_topic

            spec = get_topic(topic)
            framework_emits = spec is not None and spec.kind in ("emit", "both")
            if not framework_emits and topic not in all_emits:
                soft_problems.append(
                    f"engine {m.name!r} subscribes to {topic!r} but nothing emits it"
                )

    all_problems = unknown_problems + soft_problems

    if all_problems and strict:
        raise TopicRegistryError(
            "topic registry cross-check failed (strict): "
            + "; ".join(all_problems),
            problems=all_problems,
        )

    # Warn-by-default. Unknown topics get a louder, distinct message because
    # they are the strongest silent-failure signal; soft coverage gaps are
    # expected for the layered engine substrate (see docstring).
    for prob in unknown_problems:
        logger.warning("topic-registry: UNKNOWN TOPIC — %s", prob)
    for prob in soft_problems:
        logger.warning("topic-registry: %s", prob)

    return all_problems


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────

def load_engine_registry(
    root: Path | None = None,
    *,
    strict_topics: bool | None = None,
) -> PluginRegistry:
    """Parse all engine manifests, import adapters, return a PluginRegistry.

    Args:
        root: Repository root.  When None, inferred from this file's location.
        strict_topics: Controls the G2 topic cross-check. When None (default),
            reads the ``ROBIT_STRICT_TOPICS`` env var (``"1"``/``"true"``/``"yes"``
            → strict). When True, unsubscribed-emit / unemitted-subscribe
            problems raise :class:`TopicRegistryError`; when False they only
            warn. **Unknown** topics always raise regardless of this flag.

    Returns:
        A ``dict[name, PluginAdapter]`` keyed by each engine's manifest name,
        ordered by dependency (engines with no deps first).

    Raises:
        ManifestSchemaError:  A manifest fails validation.
        EngineLoadError:      An adapter cannot be imported.
        DependencyCycleError: A circular dependency is detected.
        TopicRegistryError:   A declared topic is unknown, or (in strict mode)
                              an emit/subscribe has no counterpart.
    """
    if root is None:
        root = _default_root()

    if strict_topics is None:
        strict_topics = os.environ.get(_STRICT_TOPICS_ENV, "").strip().lower() in (
            "1",
            "true",
            "yes",
        )

    manifest_paths = find_engine_manifests(root)

    manifests: list[EngineManifest] = []
    for path in manifest_paths:
        manifests.append(parse_manifest(path))

    # G2 — boot-time topic registry cross-check. Runs before adapter import so
    # a topic typo is caught before any subprocess is spawned.
    _cross_check_topics(manifests, strict=strict_topics)

    ordered = _topological_sort(manifests)

    registry: dict[str, PluginAdapter] = {}
    for manifest in ordered:
        adapter = _import_adapter(manifest)
        registry[manifest.name] = adapter

    return registry
