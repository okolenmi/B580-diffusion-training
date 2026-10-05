"""Discovery: walk ``nodes/`` and collect every concrete Node subclass.

Replaces the legacy hand-maintained import list
(``server/nodegraph_registry.py::_load()`` -- drift bugs happened twice)
with a ``pkgutil`` walk. Rules:

* every module under ``nodes.`` except ``smoke_tests`` is imported;
* a module that fails to import is recorded as a ``CatalogLoadError``
  and skipped (loud absence -- its classes then fail validation as
  ``unknown_class`` if a saved graph references them), never swallowed;
* classes are ``Node`` subclasses that are concrete (not abstract) and
  *defined* under ``nodes.`` (re-exports dedupe by object identity);
* two different classes answering to the same ``__name__`` is a
  conflict: saved graphs resolve by that name, so the first wins and
  the conflict is reported as a load error.

Measured cost: ~94 modules, 0 errors, 1.3 s including torch, once per
process (or per ``refresh``).
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import threading
from collections.abc import Callable

from nodes.core import Node

from ...application.memory_fingerprint import MemoryFieldsResolver
from ...application.ports.graph_catalog import CatalogLoadError

ScanResult = tuple[dict[str, type], tuple[CatalogLoadError, ...]]


def scan_nodes() -> ScanResult:
    """Import every nodes/ module; return ``(classes_by_name, errors)``."""
    import nodes  # local: this module stays import-light until first use

    classes: dict[str, type] = {}
    errors: list[CatalogLoadError] = []
    for module_info in pkgutil.walk_packages(nodes.__path__, prefix="nodes."):
        name = module_info.name
        if "smoke_tests" in name:
            continue
        try:
            module = importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 -- one broken module must not hide the rest
            errors.append(
                CatalogLoadError(
                    module=name, message=f"{type(exc).__name__}: {exc}"
                )
            )
            continue
        for attribute in dir(module):
            candidate = getattr(module, attribute, None)
            if not _is_node_class(candidate):
                continue
            existing = classes.get(candidate.__name__)
            if existing is not None and existing is not candidate:
                errors.append(
                    CatalogLoadError(
                        module=candidate.__module__,
                        message=(
                            f"class name {candidate.__name__!r} is defined by both "
                            f"{existing.__module__} and {candidate.__module__}; "
                            f"keeping the first (saved graphs resolve by name)"
                        ),
                    )
                )
                continue
            classes[candidate.__name__] = candidate
    return classes, tuple(errors)


def _is_node_class(candidate: object) -> bool:
    return (
        isinstance(candidate, type)
        and issubclass(candidate, Node)
        and candidate is not Node
        and not inspect.isabstract(candidate)
        and str(getattr(candidate, "__module__", "")).startswith("nodes.")
    )


class NodeRegistry:
    """Instance-owned discovery cache (the legacy module-global
    ``_CACHE``'s replacement): two application containers cannot share
    state, tests inject their own ``scan``, and ``refresh`` swaps in a
    freshly scanned dict atomically (in-flight readers keep the old one).
    """

    def __init__(self, scan: Callable[[], ScanResult] | None = None) -> None:
        self._scan = scan if scan is not None else scan_nodes
        self._lock = threading.Lock()
        self._classes: dict[str, type] | None = None
        self._errors: tuple[CatalogLoadError, ...] = ()

    def load(self, *, refresh: bool = False) -> ScanResult:
        """Cached scan; ``refresh=True`` re-walks ``nodes/`` (picks up
        newly added node files without a restart)."""
        if self._classes is None or refresh:
            with self._lock:
                if self._classes is None or refresh:
                    classes, errors = self._scan()
                    self._classes, self._errors = classes, errors
        return self._classes, self._errors


def memory_fields_resolver(registry: NodeRegistry) -> MemoryFieldsResolver:
    """Resolve a class name to its declared ``memory_fields`` (MEM-01).

    Reads the class-level declaration off the registry's cached classes:
    data, not code, so nothing is instantiated and no module is imported
    that the server's catalog would not import anyway (the registry walk
    already imports every ``nodes/`` module once). A class that declares
    nothing -- or does not exist -- answers None, which the fingerprint
    reads as "this node contributes no field" rather than as a default.
    """
    def resolve(class_name: str) -> tuple[str, ...] | None:
        classes, _errors = registry.load()
        cls = classes.get(class_name)
        fields = getattr(cls, "memory_fields", None) if cls is not None else None
        if not fields:
            return None
        return tuple(str(field) for field in fields)

    return resolve
