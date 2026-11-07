"""The fingerprint's view of a node class: which of its params make two runs "the same configuration" for peak
purposes."""

from __future__ import annotations

from typing import Any

from ...application.memory_fingerprint import MemoryField, MemoryFieldsResolver


def memory_fields_resolver(registry: Any) -> MemoryFieldsResolver:
    """Resolve a class name to its declared ``memory_fields``, each with its port's default.

    Reads class-level declarations off the registry's cached classes: data, not code, so nothing is instantiated.
    A class that declares nothing -- or does not exist -- answers None, which the fingerprint reads as "this node
    contributes nothing" rather than as a default. A declared name that is not a port is a bug in the node
    (``next/tests`` checks it), and is skipped here rather than failing a start.
    """

    def resolve(class_name: str) -> tuple[MemoryField, ...] | None:
        classes, _errors = registry.load()
        cls = classes.get(class_name)
        names = getattr(cls, "memory_fields", None) if cls is not None else None
        if not names:
            return None
        ports = getattr(cls, "INPUTS", {})
        return tuple(
            MemoryField(str(name), ports[name].default, ports[name].required) for name in names if name in ports
        )

    return resolve
