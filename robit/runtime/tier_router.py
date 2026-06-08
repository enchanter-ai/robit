"""robit.runtime.tier_router — maps semantic task classes to concrete model IDs.

Engines say "I need an orchestrator" or "I need a validator."  TierRouter
translates that intent into a model_id sourced from the registry.  When a model
retires, update the registry (or the defaults table below) — not 50 engine call-sites.

``route(task_class) -> str`` returns the single primary model_id.
``route_chain(task_class) -> tuple[str, ...]`` returns an ordered fallback chain
(primary first, then registry-known alternatives, de-duplicated) for callers that
want to retry a different model on a retryable upstream failure.  ``route()`` is
defined as ``route_chain(...)[0]`` so the two never diverge.

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

        # Pre-resolve fallback chains at construction time so route() /
        # route_chain() are deterministic. The chain head is the primary
        # model (what route() returns); _resolve_chain raises
        # MissingDefaultFamilyError if the primary cannot be resolved — same
        # contract as before.
        self._chains: dict[str, tuple[str, ...]] = {}
        for tc in _VALID_TASK_CLASSES:
            if tc not in self._overrides:
                self._chains[tc] = self._resolve_chain(tc)

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

        # route() is defined as the head of route_chain() so the two never
        # diverge — the primary model is always chain[0].
        return self.route_chain(task_class, size_hint)[0]

    def route_chain(
        self, task_class: TaskClass, size_hint: int | None = None
    ) -> tuple[str, ...]:
        """Return an ordered fallback chain of model_ids for *task_class*.

        The first element is the primary model (identical to what
        :meth:`route` returns); the remaining elements are the other
        family/preferred alternatives the registry already knows about,
        ordered by preference (latest-first) and de-duplicated.

        Callers (e.g. the proxy's ``call_upstream``) iterate the chain: try
        the primary, and on a *retryable* upstream failure fall through to
        the next entry. A single-element chain therefore behaves exactly
        like the no-fallback path did before.

        Parameters
        ----------
        task_class:
            One of ``"orchestrator"``, ``"executor"``, ``"validator"``,
            ``"image"``, or ``"embed"``.
        size_hint:
            Reserved for future context-window-aware routing. Currently
            logged and ignored — the chain is the same regardless of value.

        Raises
        ------
        UnknownTaskClassError
            If *task_class* is not a recognised literal.
        """
        if task_class not in _VALID_TASK_CLASSES:
            raise UnknownTaskClassError(task_class)

        if size_hint is not None:
            logger.debug(
                "TierRouter.route_chain: size_hint=%d received for "
                "task_class=%r — size-aware routing is not yet implemented; "
                "ignoring.",
                size_hint,
                task_class,
            )

        # An explicit override pins a single model — no fallback alternatives.
        if task_class in self._overrides:
            return (self._overrides[task_class],)

        return self._chains[task_class]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_chain(self, task_class: str) -> tuple[str, ...]:
        """Resolve the ordered, de-duped fallback chain for *task_class*.

        The chain head is the primary model (the same value the previous
        single-model ``_resolve_default`` produced):

        1. The preferred model_id, if present in the registry.
        2. Otherwise the latest member of the first preferred family that
           matches the subfamily filter (orchestrator/executor/validator) or
           simply the latest member (image/embed).

        The tail is the *remaining* candidates from the same family pool,
        ordered latest-first (lexicographically greatest model_id first),
        with the primary removed. Duplicates are dropped while preserving
        order.

        Raises ``MissingDefaultFamilyError`` when no primary can be resolved
        — identical contract to the pre-fallback ``_resolve_default``.
        """
        families = _DEFAULT_FAMILY_MAP.get(task_class, [])
        subfamily_filter = _FAMILY_FILTER.get(task_class)

        # Gather the candidate pool, ordered latest-first, walking the
        # preferred families in declared order.
        candidates: list[str] = []
        for family_name in families:
            try:
                members = self._registry.family(family_name)
            except UnknownFamilyError:
                continue

            if subfamily_filter:
                members = [
                    m for m in members if subfamily_filter in m.display_name
                ]
                if not members:
                    # Filter eliminated all members — try the next family.
                    continue

            # Latest-first within this family slice.
            for entry in sorted(
                members, key=lambda e: e.model_id, reverse=True
            ):
                candidates.append(entry.model_id)

        # Decide the primary (chain head): preferred model if it's present in
        # the registry, otherwise the latest candidate.
        primary: str | None = None
        preferred = _PREFERRED_MODEL.get(task_class)
        if preferred is not None:
            try:
                self._registry.model(preferred)
                primary = preferred
            except KeyError:
                primary = None

        if primary is None:
            if not candidates:
                raise MissingDefaultFamilyError(task_class, families)
            primary = candidates[0]

        # Assemble: primary first, then remaining candidates, de-duplicated
        # while preserving order.
        ordered = [primary] + [c for c in candidates if c != primary]
        seen: set[str] = set()
        chain: list[str] = []
        for model_id in ordered:
            if model_id not in seen:
                seen.add(model_id)
                chain.append(model_id)
        return tuple(chain)
