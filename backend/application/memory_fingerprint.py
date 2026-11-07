"""GraphMemoryFingerprint -- what makes two runs "the same configuration" for peak purposes, computed without
importing torch.

The server needs a fingerprint **before spawning** a child, and must not import torch to get it. So it is derived from:

1. **Declared fields** -- a node class lists, in a class-level ``memory_fields`` tuple, the names of the params that
   change the peak (a batch size, a rank, a resolution policy). This is data, not code, so the server reads it without
   instantiating anything.
2. **The dataset's name** -- the shapes a run trains come from the dataset, and the same configuration over a
   different dataset peaks differently.

The key is the dataset plus, for every node that declares fields, its class and the value of each declared field: the
node's own param, else the port's default. A field that another node feeds has no value the server can see, so the
fingerprint is **unknown**, not defaulted. An unknown fingerprint is not a zero peak; it means nothing has been
measured for this configuration.

It is deliberately not a hash of the whole graph: two graphs that differ in a caption or a checkpoint path but train
the same shapes at the same batch have the same peak, and folding that difference in would mean re-measuring for every
dataset shuffle.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..domain.graph import GraphDefinition


@dataclass(frozen=True, slots=True)
class MemoryField:
    """One declared field of a node class: its name, and what the port holds when the graph does not set it."""

    name: str
    default: object = None
    required: bool = False


#: A resolver takes a class name and returns that class's declared fields, or None if the class is unknown or declares
#: none. Injected so this module stays pure (no infrastructure imports) and testable.
MemoryFieldsResolver = Callable[[str], tuple[MemoryField, ...] | None]


@dataclass(frozen=True, slots=True)
class GraphMemoryFingerprint:
    """The dataset and, per declaring node, its class with the values of its declared fields (sorted, so the key does
    not depend on node order)."""

    dataset: str
    nodes: tuple[tuple[str, tuple[tuple[str, str], ...]], ...]

    def key(self) -> str:
        """Stable string key for the peak store."""
        parts = (f"{cls}({','.join(f'{name}={value}' for name, value in fields)})" for cls, fields in self.nodes)
        return "|".join((self.dataset, *parts))


@dataclass(frozen=True, slots=True)
class UnknownFingerprint:
    """Sentinel for a graph whose fingerprint cannot be computed. Not a zero peak: nothing has been measured."""

    reason: str


def _show(value: object) -> str:
    """One spelling per value: JSON hands a number over as int, float or a numeric string, and a bool as itself."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return str(int(value)) if float(value).is_integer() else str(float(value))
    return str(value)


def graph_fingerprint(
    graph: GraphDefinition,
    dataset: str,
    *,
    resolve_memory_fields: MemoryFieldsResolver,
) -> GraphMemoryFingerprint | UnknownFingerprint:
    """The fingerprint of ``graph`` trained on ``dataset``, or the reason it cannot be known."""
    nodes: list[tuple[str, tuple[tuple[str, str], ...]]] = []
    for node in graph.nodes:
        declared = resolve_memory_fields(node.class_name)
        if not declared:
            continue
        fed = {edge.to_port for edge in graph.edges_to(node.uid)}
        values: list[tuple[str, str]] = []
        for field in declared:
            if field.name in fed:
                return UnknownFingerprint(f"{node.class_name}.{field.name} is fed by another node")
            if field.name in node.params:
                value = node.params[field.name]
            elif field.required:
                return UnknownFingerprint(f"{node.class_name}.{field.name} is not set")
            else:
                value = field.default
            values.append((field.name, _show(value)))
        nodes.append((node.class_name, tuple(values)))
    if not nodes:
        return UnknownFingerprint("no node declares memory fields")
    return GraphMemoryFingerprint(dataset=dataset, nodes=tuple(sorted(nodes)))
