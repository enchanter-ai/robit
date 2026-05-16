"""robit.runtime.tier_router — maps semantic task classes to concrete model IDs.

Engines say "I need an orchestrator" or "I need a validator."  TierRouter
translates that intent into a model_id sourced from the registry.  When a model
retires, update the registry (or the defaults table below) — not 50 engine call-sites.

Default routing (overrideable per-instance):
  orchestrator → latest Claude Opus  (prefers claude-opus-4-7, falls back to latest in family)
  executor     → latest Claude Sonnet (prefers claude-sonnet-4-6)
  validator    → latest Claude Haiku  (prefers claude-haiku-4-5)
  image        → latest OpenAI (Image Gen) family entry
  embed        → latest Gemini embedding family entry

Task classes follow the Wixie agent-tier definitions:
  orchestrator  Opus — judgment, intent, technique selection
  executor      Sonnet — convergence loops, translation, long tasks
  validator     Haiku — quality gate, file completeness, score freshness
  image         Any image-generation model
  embed         Any text-embedding model
"""

from __future__ import annotations

import logging
from typing import Literal, get_args

from robit.runtime.models_registry import ModelsRegistry, UnknownFamilyError

logger = logging.getLogger(__name__)

# The canonical set of task classes.
TaskClass = Literal["orchestrator", "executor", "validator", "image", "embed"]

_VALID_TASK_CLASSES: frozenset[str] = frozenset(get_args(TaskClass))

# ---------------------------------------------------------------------------
# Default family → task-class mapping.
# Ordered by preference: the first family that exists in the registry wins.
# ---------------------------------------------------------------------------

_DEFAULT_FAMILY_MAP: dict[str, list[str]] = {
    "orchestrator": ["Claude 4.x"],   # resolved to latest Opus in that family
    "executor":     ["Claude 4.x"],   # resolved to latest Sonnet in that family
    "validator":    ["Claude 4.x"],   # resolved to latest Haiku in that family
    "image":        ["OpenAI (Image Gen)", "Midjourney", "Stability AI"],
    "embed":        ["Gemini"],        # gemini-embedding-* entries
}

# Preferred model IDs per task class.  If the preferred ID is present in the
# registry, it's used directly; otherwise we fall back to latest_in_family.
_PREFERRED_MODEL: dict[str, str] = {
    "orchestrator": "claude-opus-4-7",
    "executor":     "claude-sonnet-4-6",
    "validator":    "claude-haiku-4-5",
    "image":        "gpt-image-2",
    "embed":        "gemini-embedding-2-preview",
}

# For validator/executor/orchestrator, filter by display_name substring so we
# pick the right tier within the shared "Claude 4.x" family.
_FAMILY_FILTER: dict[str, str] = {
    "orchestrator": "Opus",
    "executor":     "Sonnet",
    "validator":    "Haiku",
}


class UnknownTaskClassError(ValueError):
    """Raised when the requested task_class is not a recognised TaskClass literal."""

    def __init__(self, task_class: str) -> None:
        self.task_class = task_class
        valid = ", ".join(sorted(_VALID_TASK_CLASSES))
        super().__init__(
            f"Unknown task_class '{task_class}'. Valid values: {valid}"
        )


class MissingDefaultFamilyError(RuntimeError):
    """Raised when none of the preferred families for a task_class exist in the registry."""

    def __init__(self, task_class: str, families: list[str]) -> None:
        self.task_class = task_class
        self.families = families
        super().__init__(
            f"Cannot resolve default model for task_class '{task_class}': "
            f"none of the expected families {families!r} were found in the registry"
        )


class TierRouter:
    """Maps a TaskClass to a concrete model_id drawn from a ModelsRegistry.

    Parameters
    ----------
    registry:
        A loaded ``ModelsRegistry`` instance.
    overrides:
        Optional dict mapping TaskClass strings to explicit model_id strings.
        An override bypasses family lookup entirely — the pinned model_id is
        returned as-is (validated against the registry at construction time).

    Example
    -------
    >>> router = TierRouter(registry)
    >>> router.route("orchestrator")
    'claude-opus-4-7'
    >>> router = TierRouter(registry, overrides={"validator": "claude-haiku-3-5"})
    >>> router.route("validator")
    'claude-haiku-3-5'
    """

    def __init__(
        self,
        registry: ModelsRegistry,
        overrides: dict[str, str] | None = None,
    ) -> None:
        self._registry = registry
        self._overrides: dict[str, str] = {}

        if overrides:
            for task_class, model_id in overrides.items():
                if task_class not in _VALID_TASK_CLASSES:
                    raise UnknownTaskClassError(task_class)
                # Validate override model exists in registry.
                registry.model(model_id)  # raises UnknownModelError if absent
                self._overrides[task_class] = model_id

        # Pre-resolve defaults at construction time so route() is deterministic.
        self._defaults: dict[str, str] = {}
        for tc in _VALID_TASK_CLASSES:
            if tc not in self._overrides:
                self._defaults[tc] = self._resolve_default(tc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(self, task_class: TaskClass, size_hint: int | None = None) -> str:
        """Return the concrete model_id for *task_class*.

        Parameters
        ----------
        task_class:
            One of ``"orchestrator"``, ``"executor"``, ``"validator"``,
            ``"image"``, or ``"embed"``.
        size_hint:
            Reserved for future context-window-aware routing.  Currently
            logged and ignored — the result is the same regardless of value.

        Raises
        ------
        UnknownTaskClassError
            If *task_class* is not a recognised literal.
        """
        if task_class not in _VALID_TASK_CLASSES:
            raise UnknownTaskClassError(task_class)

        if size_hint is not None:
            logger.debug(
                "TierRouter.route: size_hint=%d received for task_class=%r — "
                "size-aware routing is not yet implemented; ignoring.",
                size_hint,
                task_class,
            )

        if task_class in self._overrides:
            return self._overrides[task_class]

        return self._defaults[task_class]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_default(self, task_class: str) -> str:
        """Resolve the default model_id for *task_class* from the registry."""
        # 1. Try the preferred model_id directly.
        preferred = _PREFERRED_MODEL.get(task_class)
        if preferred is not None:
            try:
                self._registry.model(preferred)
                return preferred
            except KeyError:
                pass  # fall through to family lookup

        # 2. Walk preferred families in order; pick the best match.
        families = _DEFAULT_FAMILY_MAP.get(task_class, [])
        subfamily_filter = _FAMILY_FILTER.get(task_class)

        for family_name in families:
            try:
                members = self._registry.family(family_name)
            except UnknownFamilyError:
                continue

            # For orchestrator/executor/validator: filter by tier keyword in
            # display_name (e.g. "Opus", "Sonnet", "Haiku").
            if subfamily_filter:
                filtered = [
                    m for m in members if subfamily_filter in m.display_name
                ]
                if filtered:
                    # Return the latest (lexicographically greatest model_id).
                    return max(filtered, key=lambda e: e.model_id).model_id
                # If the filter eliminates all members, fall through to next family.
                continue

            # For image/embed: no subfamily filter — return latest in family.
            return max(members, key=lambda e: e.model_id).model_id

        # Nothing found.
        raise MissingDefaultFamilyError(task_class, families)
