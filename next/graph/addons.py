"""Addons: small classes that plug into a host (a node, or the graph) through a slot (rework/31-node-addons.md).

An addon is defined by the interface of the slot it fills, so any host class that declares that slot offers it. The host
never imports an addon: it asks ``self.addons`` and gets objects that satisfy the slot's interface.
"""

from __future__ import annotations

import inspect
import re
from abc import ABC
from dataclasses import dataclass, field
from typing import Any, ClassVar

from .events import GRAPH, EventType
from .ports import EventIn, EventOut, Group, Port

ADDON_ID = re.compile(r"^[a-z][a-z0-9_]{0,23}$")
SLOT_PREFIX = "slot:"  # a slot's socket on its host, so a wire can fill the slot like an addon would


class AddonError(ValueError):
    """An addon change the graph would refuse; the message is the validator's."""


@dataclass(frozen=True)
class AddonSlot:
    """A place in a host where addons of one interface fit. ``many`` slots hold an ordered list."""

    name: str
    interface: Any
    many: bool = False
    doc: str = ""

    @property
    def socket(self) -> str:
        return SLOT_PREFIX + self.name

    def port(self) -> Port:
        return Port(self.socket, self.interface, required=False, multi=self.many, doc=self.doc, group="addons")


# Every host can carry addons that only watch events (triggers, loggers); they fill no collaborator slot.
EVENTS_SLOT = AddonSlot("events", object, many=True, doc="Addons that act on events and need no collaborator in the host.")


@dataclass(frozen=True)
class AddonSpec:
    """One addon on a host as saved in the graph file. ``id`` is stable inside the host, not an index."""

    id: str
    kind: str
    enabled: bool = True
    params: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"id": self.id, "kind": self.kind, "enabled": self.enabled, "params": self.params}


@dataclass(frozen=True)
class HostContext:
    """What an addon may know about its host: where it sits and the run's shared infrastructure."""

    address: str
    context: Any
    class_name: str = ""


class Addon(ABC):
    """Declare parameters as class data, like a node; ``build`` returns the object that fills the slot."""

    SLOT: ClassVar[str]
    DISPLAY_NAME: ClassVar[str | None] = None
    INPUTS: ClassVar[dict[str, Port]] = {}
    GROUPS: ClassVar[tuple[Group, ...]] = ()
    EVENTS_IN: ClassVar[dict[str, EventIn]] = {}
    EVENTS_OUT: ClassVar[dict[str, EventOut]] = {}
    REQUIRES: ClassVar[tuple[str, ...]] = ()   # addon kinds that must be on the same host
    CONFLICTS: ClassVar[tuple[str, ...]] = ()  # addon kinds that may not be
    FITS: ClassVar[tuple[str, ...]] = ()       # host class names this addon is offered on; empty = every host with the slot

    def __init__(self, host: HostContext, addon_id: str):
        self.host, self.id = host, addon_id
        self.address = f"{host.address}.{addon_id}"

    def build(self, **inputs) -> Any:
        """The collaborator for the slot. Event-only addons are their own collaborator."""
        return self

    def emit(self, port: str, event: EventType) -> None:
        if port not in self.EVENTS_OUT:
            raise KeyError(f"{type(self).__name__} declares no event output {port!r}")
        hub = self.host.context.events
        if hub is not None:
            hub.emit(self.address, port, event)

    def close(self) -> None:
        """Called once when the run ends."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls) or cls.__name__.startswith("_"):
            return
        _check(cls)


def _check(cls: type[Addon]) -> None:
    name = cls.__name__
    if not isinstance(getattr(cls, "SLOT", None), str):
        raise TypeError(f"{name} must name the slot it fills in SLOT")
    for key, port in cls.INPUTS.items():
        if not isinstance(port, Port) or port.name != key:
            raise TypeError(f"{name}.INPUTS[{key!r}] must be a Port named {key!r}")
        if "." in key or key.startswith(SLOT_PREFIX):
            raise TypeError(f"{name}.INPUTS[{key!r}]: a parameter name may not contain '.' or start with {SLOT_PREFIX!r}")
        if port.group and port.group not in {g.id for g in cls.GROUPS}:
            raise TypeError(f"{name}.INPUTS[{key!r}] is in unknown group {port.group!r}")
    for key, ev in cls.EVENTS_IN.items():
        if not isinstance(ev, EventIn) or not callable(getattr(cls, ev.handler, None)):
            raise TypeError(f"{name}.EVENTS_IN[{key!r}] needs an EventIn whose handler is a method")
    for key, ev in cls.EVENTS_OUT.items():
        if not isinstance(ev, EventOut):
            raise TypeError(f"{name}.EVENTS_OUT[{key!r}] needs an EventOut")


class AddonSet:
    """The collaborators a host may use, by slot, in saved order (addons first, then objects wired to the slot)."""

    def __init__(self, slots: tuple[AddonSlot, ...] = (), filled: dict[str, list] | None = None):
        self._slots = {s.name: s for s in (*slots, EVENTS_SLOT)}
        self._filled = {name: list(items) for name, items in (filled or {}).items()}

    def all(self, slot: str) -> list:
        self._known(slot)
        return list(self._filled.get(slot, ()))

    def one(self, slot: str) -> Any | None:
        items = self.all(slot)
        if self._slots[slot].many:
            raise LookupError(f"slot {slot!r} holds several; ask for all()")
        return items[0] if items else None

    def has(self, slot: str) -> bool:
        return bool(self._filled.get(slot))

    def _known(self, slot: str) -> None:
        if slot not in self._slots:
            raise KeyError(f"the host declares no addon slot {slot!r}")


def slots_of(host_cls: Any) -> tuple[AddonSlot, ...]:
    """The slots a host class declares, plus the one every host has."""
    return (*getattr(host_cls, "ADDON_SLOTS", ()), EVENTS_SLOT)


def slot_sockets(host_cls: Any) -> dict[str, Port]:
    """The sockets that let a wire fill each declared slot (the implicit events slot has none)."""
    return {s.socket: s.port() for s in getattr(host_cls, "ADDON_SLOTS", ())}


def addon_port_name(addon_id: str, param: str) -> str:
    return f"{addon_id}.{param}"


def split_port(name: str) -> tuple[str, str] | None:
    """``a1.start`` -> (``a1``, ``start``); None for a node's own port."""
    head, dot, tail = name.partition(".")
    return (head, tail) if dot and tail else None


class AddonBuilder:
    """Builds the enabled addons of one host in saved order, hands each its parameters and registers it for events."""

    def __init__(self, classes: dict[str, type[Addon]], hub: Any):
        self.classes, self.hub = classes, hub

    def build(self, host: HostContext, host_cls: Any, specs: tuple[AddonSpec, ...], wired_params: dict[str, dict],
              wired_slots: dict[str, list]) -> tuple[AddonSet, list[Addon]]:
        filled: dict[str, list] = {}
        made: list[Addon] = []
        for spec in specs:
            if not spec.enabled:
                continue
            cls = self.classes[spec.kind]
            addon = cls(host, spec.id)
            inputs = dict(spec.params)
            inputs.update(wired_params.get(spec.id, {}))
            for name, port in cls.INPUTS.items():
                if name not in inputs and port.default is not None:
                    inputs[name] = port.default
            made.append(addon)
            if self.hub is not None:
                self.hub.place(addon.address, host.address, addon, cls.EVENTS_IN)
            filled.setdefault(cls.SLOT, []).append(addon.build(**inputs))
        for slot, objects in wired_slots.items():
            filled.setdefault(slot.removeprefix(SLOT_PREFIX), []).extend(objects)
        return AddonSet(slots_of(host_cls), filled), made


def end_of(doc: Any, address: str, classes: dict, addon_classes: dict) -> tuple[bool, Any, str]:
    """What an event wire's end names: (exists, class that declares its events or None, label). An end is a node
    (``uid``) or an addon of a node or of the graph (``uid.id``, ``graph.id``)."""
    uid, dot, addon_id = address.partition(".")
    if uid == GRAPH:
        specs, host_cls, label = doc.addons, None, GRAPH
    else:
        spec = doc.node(uid)
        if spec is None:
            return False, None, address
        specs, host_cls, label = spec.addons, classes.get(spec.class_name), spec.class_name
    if not dot:
        return uid != GRAPH, host_cls, label
    for addon in specs:
        if addon.id == addon_id:
            return True, addon_classes.get(addon.kind), addon.kind
    return False, None, address
