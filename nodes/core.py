"""Domain-independent node/port primitives.

See docs/nodes_package_design.md for the full design reasoning. Short
version: a Node declares a fixed, typed set of input and output Ports as
real class-level data (not guessed from a constructor signature), and
build() turns input values into output values. Nothing here knows anything
about optimizers, models, or training -- that's what domain-family ABCs
(nodes/optimizer/node.py, and future nodes/<domain>/node.py modules) are
for.

This package never imports from or modifies core/, manager/, or server/
except read-only, at the point a concrete node wraps an already-verified
class from one of those (see nodes/optimizer/*.py for the pattern). See
docs/nodes_package_design.md's "Course correction" section for why.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar


@dataclass(frozen=True)
class Port:
    """Declarative metadata for one named input or output slot on a Node.
    Pure data, no behavior -- see Node below for how it's used."""
    name: str
    type: type
    required: bool = True
    default: Any = None
    doc: str = ""


class Node(ABC):
    """The universal node contract.

    INPUTS/OUTPUTS are declared, not derived -- a subclass sets these as
    real class attributes. __init_subclass__ enforces that any concrete
    (fully-implemented, instantiable) subclass has actually declared a
    non-empty OUTPUTS -- a node that claims to be usable but produces
    nothing is almost certainly a forgotten declaration, not a real design.
    INPUTS may legitimately be empty (a "source" node needing no
    configuration is a coherent idea) so it isn't enforced non-empty, but
    it must exist as a dict.

    This check is deliberately in addition to (not instead of) Python's own
    abstractmethod enforcement: abc already refuses to instantiate a
    subclass that hasn't implemented build(); it has no way to know
    anything about the INPUTS/OUTPUTS *class attributes*, since those are
    just dicts as far as Python's abc machinery is concerned. This
    __init_subclass__ hook is what actually enforces that part of the
    contract.

    Abstract intermediate classes (domain-family ABCs like OptimizerNode,
    which still have their own unimplemented abstractmethods) are exempt --
    detected via inspect.isabstract(), not by name or convention, so this
    keeps working correctly for any future domain-family ABC without needing
    to know about it here.
    """

    INPUTS: ClassVar[dict[str, Port]] = {}
    OUTPUTS: ClassVar[dict[str, Port]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls):
            return  # still abstract (e.g. a domain-family ABC) -- exempt
        if not isinstance(cls.OUTPUTS, dict):
            raise TypeError(f"{cls.__name__}.OUTPUTS must be a dict, got {type(cls.OUTPUTS)!r}")
        if not isinstance(cls.INPUTS, dict):
            raise TypeError(f"{cls.__name__}.INPUTS must be a dict, got {type(cls.INPUTS)!r}")
        if not cls.OUTPUTS:
            raise TypeError(
                f"{cls.__name__} is a concrete Node (all abstract methods "
                f"implemented) but declares no OUTPUTS. A node that produces "
                f"nothing can't be used in a graph -- if this is intentional, "
                f"the class should stay abstract instead."
            )

    @abstractmethod
    def build(self, **inputs) -> dict[str, Any]:
        """Given values for (at least) the required INPUTS, produce a dict
        covering (at least) all OUTPUTS keys. Concrete implementations
        should call self.validate_inputs(inputs) before doing any real work
        and self.validate_outputs(result) before returning, so a
        contract violation fails loudly at the point it happens rather than
        surfacing as a confusing error somewhere downstream in the graph.
        """

    def validate_inputs(self, inputs: dict) -> None:
        for name, port in self.INPUTS.items():
            if port.required and name not in inputs:
                raise ValueError(
                    f"{type(self).__name__}.build() missing required input "
                    f"'{name}' ({port.type.__name__ if hasattr(port.type, '__name__') else port.type})"
                )

    def validate_outputs(self, outputs: dict) -> None:
        missing = set(self.OUTPUTS.keys()) - set(outputs.keys())
        if missing:
            raise ValueError(
                f"{type(self).__name__}.build() did not produce declared "
                f"output(s): {sorted(missing)}. This is a bug in the node's "
                f"build() implementation, not a caller error."
            )
