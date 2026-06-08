"""Loader error types — raised by manifest parsing, engine import, and dependency ordering."""

from __future__ import annotations


class ManifestSchemaError(Exception):
    """Raised when an engine.toml fails schema validation.

    Attributes:
        field: The manifest field that caused the error (e.g. "phases", "name").
              None when the error is not field-specific.
        manifest_path: Filesystem path to the offending manifest, if known.
    """

    def __init__(
        self,
        message: str,
        *,
        field: str | None = None,
        manifest_path: str | None = None,
    ) -> None:
        super().__init__(message)
        self.field = field
        self.manifest_path = manifest_path

    def __str__(self) -> str:
        parts = [super().__str__()]
        if self.field:
            parts.append(f"(field: {self.field!r})")
        if self.manifest_path:
            parts.append(f"(manifest: {self.manifest_path})")
        return " ".join(parts)


class EngineLoadError(Exception):
    """Raised when an engine adapter cannot be imported or resolved.

    Attributes:
        engine_name: The engine name from the manifest.
        adapter_path: The full ``module:attr`` string from the manifest.
    """

    def __init__(
        self,
        message: str,
        *,
        engine_name: str | None = None,
        adapter_path: str | None = None,
    ) -> None:
        super().__init__(message)
        self.engine_name = engine_name
        self.adapter_path = adapter_path

    def __str__(self) -> str:
        parts = [super().__str__()]
        if self.engine_name:
            parts.append(f"(engine: {self.engine_name!r})")
        if self.adapter_path:
            parts.append(f"(adapter: {self.adapter_path!r})")
        return " ".join(parts)


class TopicRegistryError(Exception):
    """Raised when an engine's declared topics violate the central registry.

    G2 boot-time cross-check. Fires when (in strict mode) an engine emits a
    topic no one subscribes to, subscribes to a topic no one emits, or declares
    a topic that is not registry-known. Wildcard subscriptions (``*`` /
    ``foo.*``) and ``lifecycle.*`` subscriptions are always permitted and never
    raise.

    Attributes:
        engine_name: The engine whose manifest tripped the check, if known.
        topic:       The offending topic string, if a single topic is to blame.
        kind:        ``"emit"`` or ``"subscribe"`` — which side declared it.
        problems:    Full list of human-readable problem strings for this run.
    """

    def __init__(
        self,
        message: str,
        *,
        engine_name: str | None = None,
        topic: str | None = None,
        kind: str | None = None,
        problems: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.engine_name = engine_name
        self.topic = topic
        self.kind = kind
        self.problems = problems or []

    def __str__(self) -> str:
        parts = [super().__str__()]
        if self.engine_name:
            parts.append(f"(engine: {self.engine_name!r})")
        if self.topic:
            parts.append(f"(topic: {self.topic!r}, kind: {self.kind})")
        return " ".join(parts)


class DependencyCycleError(Exception):
    """Raised when engine ``depends_on`` declarations form a cycle.

    Attributes:
        cycle: Ordered list of engine names that form the cycle.
    """

    def __init__(self, message: str, *, cycle: list[str] | None = None) -> None:
        super().__init__(message)
        self.cycle = cycle or []

    def __str__(self) -> str:
        if self.cycle:
            return f"{super().__str__()} — cycle: {' -> '.join(self.cycle)}"
        return super().__str__()
