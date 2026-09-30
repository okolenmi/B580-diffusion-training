"""GraphCatalog port -- read-side introspection over the node registry.

Everything the editor's palette needs, derived from the *real* Python
classes (reflection, never a hand-maintained second file): identity,
display name, domain grouping, declared ports with JSON-safe defaults,
presets for dynamic nodes, and whether diagnostics exist.

Deliberately separate from ``GraphRuntime``: serving the palette must
never execute anything, and a future reader (cached snapshot, alternate
source) can replace this adapter without touching execution.

Implementation lives in ``infrastructure/graph/`` behind the shared,
instance-owned ``NodeRegistry`` (auto-discovery -- see
``docs/design/backend/05-graph-runtime.md`` section 1).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PortInfo:
    """One declared input/output port, JSON-ready.

    ``default`` is the JSON-native value when the declared default is
    representable (None/bool/int/float/str/list/dict), else None;
    ``default_repr`` is always the ``repr()`` string -- consumers pick
    whichever they can use, instead of parsing reprs (the legacy
    contract) or losing non-JSON defaults entirely. For a required port
    both are None ("no default to show").
    """

    name: str
    type: str  # best-effort readable type ("float", "any", "list[str]", ...)
    required: bool
    doc: str = ""
    type_mro: tuple[str, ...] = ()  # [type_str, ...bases...] -- lets a caller
    # check "is that output's type a subclass of this input's type" client-side
    # as a UX prefilter; the server's validate() is authoritative.
    default: Any = None  # JSON-native value, or None when not representable
    default_repr: str | None = None
    path_kind: str | None = None  # UI hint: render a picker for this path-ish input
    choices: tuple[str, ...] | None = None  # UI hint: closed set -> dropdown
    visible_when: tuple | None = None  # [other_port_name, value(s)] UI hint
    widget_only: bool = False  # UI hint: no wire socket, widget only


@dataclass(frozen=True, slots=True)
class PresetInfo:
    """One NodePreset of a dynamic node (required-only shapes, per
    nodes.core.NodePreset's own docstring)."""

    name: str
    required_inputs: tuple[PortInfo, ...]
    required_outputs: tuple[PortInfo, ...]


@dataclass(frozen=True, slots=True)
class NodeInfo:
    """One node class, introspected.

    ``class_name`` (the real ``__name__``) is the stable identity saved
    graphs resolve against; ``display_name`` is presentation-only.
    """

    class_name: str
    display_name: str
    domain: str  # from the module path (nodes.optimizer.x -> "optimizer")
    module: str
    doc: str  # first docstring line, or ""
    bases: tuple[str, ...]  # real inheritance chain (Node first after self)
    inputs: tuple[PortInfo, ...]
    outputs: tuple[PortInfo, ...]
    node_kind: str  # Node.NODE_KIND: "static" | "dynamic"
    presets: tuple[PresetInfo, ...] | None  # non-None only when dynamic
    has_diagnostics: bool  # diagnostics() actually overridden, not inherited


@dataclass(frozen=True, slots=True)
class CatalogLoadError:
    """A module under nodes/ that failed to import during discovery.

    Surfaced in the catalog response rather than swallowed: an
    unimportable node module used to be *invisible* (never added to the
    hand list); now it is loudly absent AND explained, and its classes
    simply fail validation as ``unknown_class`` if referenced.
    """

    module: str
    message: str


@dataclass(frozen=True, slots=True)
class CatalogSnapshot:
    """Full palette payload: every discovered node + any load failures."""

    nodes: tuple[NodeInfo, ...]
    load_errors: tuple[CatalogLoadError, ...] = ()

    @property
    def domains(self) -> dict[str, tuple[NodeInfo, ...]]:
        """Nodes grouped by domain, domains in sorted order, nodes
        sorted by display name (stable palette, no import-order jitter)."""
        grouped: dict[str, list[NodeInfo]] = {}
        for node in self.nodes:
            grouped.setdefault(node.domain, []).append(node)
        return {
            domain: tuple(sorted(items, key=lambda n: (n.display_name, n.class_name)))
            for domain, items in sorted(grouped.items())
        }


class GraphCatalog(ABC):
    """Read-side contract over the node class registry."""

    @abstractmethod
    def snapshot(self, *, refresh: bool = False) -> CatalogSnapshot:
        """Introspect every discoverable node.

        ``refresh=True`` re-runs discovery (picks up newly added node
        files without a restart); otherwise the registry's cached scan
        is used. Import failures never raise -- they come back as
        ``load_errors``.
        """
        raise NotImplementedError

    @abstractmethod
    def diagnostics(self, class_name: str, params: dict) -> dict[str, list[str]]:
        """Live per-input diagnostic lines for one node class.

        Raises ``NodeClassNotFoundError`` for an unknown class; any
        exception the node's own ``diagnostics()`` raises propagates
        (a bad mid-edit params dict is an ordinary 400 for the use case
        to wrap, not a server fault).
        """
        raise NotImplementedError
