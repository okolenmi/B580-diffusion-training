"""The fingerprint's view of a node class: which of its params make two runs "the same configuration" for peak purposes."""

from __future__ import annotations

from typing import Any

from ...application.memory_fingerprint import MemoryFieldsResolver


def memory_fields_resolver(registry: Any) -> MemoryFieldsResolver:
    """Resolve a class name to its declared ``memory_fields``.

    Reads the class-level declaration off the registry's cached classes: data, not code, so nothing is instantiated.
    A class that declares nothing -- or does not exist -- answers None, which the fingerprint reads as "this node
    contributes no field" rather than as a default.
    """

    def resolve(class_name: str) -> tuple[str, ...] | None:
        classes, _errors = registry.load()
        cls = classes.get(class_name)
        fields = getattr(cls, "memory_fields", None) if cls is not None else None
        if not fields:
            return None
        return tuple(str(field) for field in fields)

    return resolve
