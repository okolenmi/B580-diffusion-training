"""Domain-independent node/port primitives.

A Node declares a fixed, typed set of input and output Ports as real
class-level data (not guessed from a constructor signature), and
build() turns input values into output values. Nothing here knows anything
about optimizers, models, or training -- that's what domain-family ABCs
(nodes/optimizer/node.py, and other nodes/<domain>/node.py modules) are
for.

This package never imports from or modifies core/, manager/, or server/
except read-only, at the point a concrete node wraps an already-verified
class from one of those (see nodes/optimizer/*.py for the pattern).
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
    path_kind: str | None = None  # e.g. "checkpoint"/"lora"/"dataset" -- tells the graph
    # editor this string/Path input should be a picker against a server-configured
    # directory (see server/asset_paths.py) instead of freeform text. None = ordinary
    # widget. Purely a UI hint; build() never reads this.


@dataclass(frozen=True)
class NodePreset:
    """One named configuration of a dynamic node (Node.NODE_KIND ==
    "dynamic") -- just enough to search/suggest against
    (docs/resources_controller_redesign_plan.md's Phase 3 suggestion-
    menu resolution): a name plus this preset's own REQUIRED-only
    inputs/outputs, pre-resolved. Optional ports are deliberately
    excluded -- they're "just helpers" for suggestion purposes, per
    that plan's own reasoning, and carry no signal either way about
    whether this preset is a relevant match for a dropped wire.

    Deliberately narrower than the eventual ResourcePreset interface
    (Phase 4 of the same plan): validators, the parameter-value
    dictionary, and the processor method are that richer, separate
    concern -- this is only what a dynamic node declares about itself
    for the purpose of being found and suggested in the editor, not how
    it actually gets built or configured."""
    name: str
    required_inputs: dict[str, Port]
    required_outputs: dict[str, Port]

    def __post_init__(self):
        for label, ports in (("required_inputs", self.required_inputs),
                              ("required_outputs", self.required_outputs)):
            for key, port in ports.items():
                if not port.required:
                    raise ValueError(
                        f"NodePreset {self.name!r}: {label}[{key!r}] is a Port with "
                        f"required=False -- self-contradictory, since this dict is "
                        f"specifically the required-only subset (see this class's "
                        f"own docstring). An optional, preset-specific port doesn't "
                        f"belong in required_inputs/required_outputs at all -- full "
                        f"shape resolution including optional ports is Phase 4/5's "
                        f"job (the real ResourcePreset/processor mechanism), out of "
                        f"scope for this class (search/suggestion only)."
                    )


class ExecutionContext:
    """Infrastructure a Node's build() may need beyond its declared Ports
    -- e.g. MonitorNode's live-data bus. Constructed once per graph run
    (see server/graph_executor.py) and passed to every node explicitly,
    not reached for as a module-level global: a node that needs
    something from the outside world takes it as an injected dependency,
    the same way it takes typed Port values, rather than importing a
    singleton. Deliberately a plain, extensible bag of optional fields --
    a node that doesn't need anything here just doesn't touch it; a
    future need (a logger, a cache, whatever) is one more field, not a
    new global to invent.
    """

    def __init__(self, monitor_bus=None, cancel_event=None):
        self.monitor_bus = monitor_bus
        # threading.Event, not asyncio -- graph execution runs in a plain
        # background thread (server/routes_nodegraph.py), and a Node's
        # build() (e.g. SupervisedLoRATrainerNode's step loop) polls this
        # cooperatively between steps, never mid-backward-pass. None means
        # "not cancellable" (e.g. a direct Python call/test, no server
        # involved) -- should_cancel() handles that without every caller
        # needing its own None-check.
        self.cancel_event = cancel_event

    def should_cancel(self) -> bool:
        return self.cancel_event is not None and self.cancel_event.is_set()


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

    # Palette label override. __name__ (e.g. "ComposedCAMEOptimizerNode")
    # is the STABLE identifier -- a saved graph's class_name resolves
    # against it (server/nodegraph_registry.py's get_registry()), so it
    # can never be renamed without breaking every already-saved graph.
    # DISPLAY_NAME is a second, independent string with none of that
    # constraint: what a person sees in the graph-editor palette instead
    # of the raw class name. None (the default) means "derive one
    # automatically" -- see server/nodegraph_introspect.py's
    # _auto_display_name(), which strips the trailing "Node" suffix and
    # splits the rest into words. Set this explicitly only for a class
    # whose auto-derived name reads badly; leaving it None is the common
    # case and does not need touching for a new node to get a reasonable
    # palette label for free. See docs/training_pipeline_design.md
    # section 11.5 for the full rationale.
    DISPLAY_NAME: ClassVar[str | None] = None

    # "static" (the default) means INPUTS/OUTPUTS above are the real,
    # complete, final port set -- true for every node in this project
    # today. "dynamic" means this class also has presets (see
    # NodePreset above) -- INPUTS/OUTPUTS still exist and still matter
    # (a dynamic node's *common* ports, present no matter which preset
    # is chosen), but list_presets() below is what a dynamic node
    # actually needs implemented for the editor's suggestion-menu
    # search to find it. See docs/resources_controller_redesign_plan.md
    # Phase 3 for the full rationale -- this exists specifically
    # because that search can't cheaply enumerate "every possible shape
    # a params-dependent node could have," but can cheaply enumerate
    # "every declared preset's own fixed, required-only shape."
    NODE_KIND: ClassVar[str] = "static"

    @classmethod
    def list_presets(cls) -> list[NodePreset]:
        """Only meaningful when NODE_KIND == "dynamic" -- see
        NodePreset's own docstring for what a preset actually needs to
        report and why it's narrower than the full ResourcePreset
        interface. A static node (the overwhelming default) never
        calls this. Raises rather than returning an empty list because
        __init_subclass__ below already refuses to define a concrete
        dynamic-kind class that hasn't overridden this -- reaching this
        base implementation at all means something bypassed that check
        (e.g. calling it directly on an abstract intermediate class),
        which should fail loudly here too, not return a silently empty,
        misleadingly-valid-looking preset list."""
        raise NotImplementedError(
            f"{cls.__name__}.list_presets() was not overridden. "
            f"NODE_KIND == 'dynamic' requires a real implementation -- "
            f"see NodePreset's docstring in this module."
        )

    def __init__(self, context: ExecutionContext | None = None):
        self.context = context or ExecutionContext()

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
        if cls.NODE_KIND not in ("static", "dynamic"):
            raise TypeError(
                f"{cls.__name__}.NODE_KIND must be 'static' or 'dynamic', "
                f"got {cls.NODE_KIND!r}."
            )
        if cls.NODE_KIND == "dynamic" and cls.list_presets.__func__ is Node.list_presets.__func__:
            # Compares the underlying function, not the bound classmethod
            # (two separately-bound classmethod objects for the exact same
            # inherited function are never `is`-equal to each other, so
            # that comparison would always look like "overridden" even
            # when it isn't) -- and walks the real MRO correctly for a
            # multi-level hierarchy: an intermediate class overriding this
            # correctly satisfies a further concrete subclass that doesn't
            # re-override it, since cls.list_presets.__func__ then resolves
            # to that intermediate override, not Node's own base version.
            raise TypeError(
                f"{cls.__name__}: NODE_KIND == 'dynamic' but list_presets() "
                f"wasn't overridden -- a dynamic node must declare its own "
                f"presets, or it can never be found by the editor's "
                f"suggestion-menu search (docs/resources_controller_redesign_plan.md, "
                f"Phase 3). If this class doesn't have real presets yet, use "
                f"NODE_KIND = 'static' until it does."
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
