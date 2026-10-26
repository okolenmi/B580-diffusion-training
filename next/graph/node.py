"""The Node base class, the per-run ExecutionContext and the Completion side channel."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import inspect
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from .events import EventHub, EventType, Reporter
from .modules import DisplayModule
from .ports import DynamicShape, EventIn, EventOut, Group, Port
from .settings import Settings, SettingSpec, SettingsRegistry, SettingsView

FIELD_TYPES = (bool, int, float, str, Path, list, dict, tuple)
MAX_UNGROUPED_FIELDS = 8


def is_field(port: Port) -> bool:
    """A port a person fills in rather than wires: a primitive type."""
    return isinstance(port.type, type) and port.type in FIELD_TYPES


@dataclass(frozen=True)
class Completion:
    """How finished a node's outputs are (rework/contracts/runs.md)."""

    state: str = "complete"
    step: int | None = None
    total: int | None = None
    reason: str | None = None

    COMPLETE_STATE: ClassVar[str] = "complete"
    UNFINISHED_STATE: ClassVar[str] = "unfinished"

    def __post_init__(self) -> None:
        if self.state not in (self.COMPLETE_STATE, self.UNFINISHED_STATE):
            raise ValueError(f"Completion.state must be 'complete' or 'unfinished', got {self.state!r}")

    @property
    def unfinished(self) -> bool:
        return self.state == self.UNFINISHED_STATE

    @classmethod
    def unfinished_at(cls, step: int | None = None, total: int | None = None,
                      reason: str | None = "stopped") -> Completion:
        return cls(cls.UNFINISHED_STATE, step=step, total=total, reason=reason)

    def as_dict(self) -> dict:
        return {"state": self.state, "step": self.step, "total": self.total, "reason": self.reason}


COMPLETE = Completion()


@dataclass(frozen=True)
class Environment:
    """What the machine offers, supplied by the composition root: named services such as the model
    library. Graph code never looks inside; nodes ask for a service by name."""

    services: dict = field(default_factory=dict, hash=False, compare=False)

    def service(self, name: str) -> Any:
        try:
            return self.services[name]
        except KeyError:
            raise LookupError(f"no {name!r} service is configured") from None

    def has(self, name: str) -> bool:
        return name in self.services


class Task:
    """A stretch of work worth a name: ``with context.task("Encoding latents", total=n) as task: task.update(done)``.

    The reporter gets ``task`` frames (started, progress, finished), so any monitor can show what a long node is doing
    without the node knowing how it is shown. Progress frames are throttled; the first and last always go out.
    """

    INTERVAL_S = 0.5

    def __init__(self, context: ExecutionContext, title: str, total: int | None):
        self.context, self.title, self.total, self.done = context, title, total, 0
        self._started = self._last = 0.0

    def _frame(self, state: str, **extra: Any) -> None:
        self.context.report("task", {"node": self.context.uid, "title": self.title, "state": state, "done": self.done,
                                     "total": self.total, **extra})

    def __enter__(self) -> Task:
        self._started = self._last = time.monotonic()
        self._frame("started")
        return self

    def update(self, done: int, total: int | None = None) -> None:
        self.done = done
        if total is not None:
            self.total = total
        now = time.monotonic()
        if now - self._last >= self.INTERVAL_S:
            self._last = now
            self._frame("progress")

    def __exit__(self, kind, error, trace) -> bool:
        self._frame("finished", seconds=round(time.monotonic() - self._started, 2), ok=kind is None)
        return False


class ExecutionContext:
    """Run-wide infrastructure handed to every node. ``scoped`` gives one node its own copy."""

    def __init__(self, *, cancel_event: threading.Event | None = None, reporter: Reporter | None = None,
                 settings: Settings | None = None, events: EventHub | None = None, runtime: Any = None,
                 env: Environment | None = None):
        self.cancel_event = cancel_event
        self.reporter = reporter
        self.events = events
        self.runtime = runtime
        self.env = env or Environment()
        self._settings = settings if settings is not None else SettingsRegistry().snapshot()
        self.settings: SettingsView | Settings = self._settings
        self.uid: str = ""
        self._upstream: dict[str, Completion] = {}
        self._reported: Completion | None = None

    def should_cancel(self) -> bool:
        return self.cancel_event is not None and self.cancel_event.is_set()

    def scoped(self, uid: str, reads: tuple[str, ...], owner: str,
               upstream: dict[str, Completion] | None = None) -> ExecutionContext:
        clone = copy.copy(self)
        clone.uid = uid
        clone.settings = self._settings.view(reads, owner)
        clone._upstream = dict(upstream or {})
        clone._reported = None
        return clone

    def upstream_completion(self, port: str) -> Completion:
        return self._upstream.get(port, COMPLETE)

    def any_upstream_unfinished(self) -> Completion | None:
        for completion in self._upstream.values():
            if completion.unfinished:
                return completion
        return None

    def report_completion(self, completion: Completion) -> None:
        if not isinstance(completion, Completion):
            raise TypeError(f"report_completion needs a Completion, got {type(completion).__name__}")
        self._reported = completion

    @property
    def reported_completion(self) -> Completion | None:
        return self._reported

    def report(self, stream: str, frame: dict) -> None:
        if self.reporter is not None:
            self.reporter.report(stream, frame)

    def task(self, title: str, total: int | None = None) -> Task:
        return Task(self, title, total)


@dataclass(frozen=True)
class Resolution:
    """The answer to 'given these params, what does this node look like?'
    (rework/contracts/resolve.md)."""

    state: str
    inputs: dict[str, Port]
    outputs: dict[str, Port]
    choices: dict[str, tuple[str, ...]] = field(default_factory=dict)
    info: str = ""
    message: str | None = None
    focus: str | None = None
    detail: dict = field(default_factory=dict)

    STATES: ClassVar[tuple[str, ...]] = ("ready", "needs_choice", "error")

    def __post_init__(self) -> None:
        if self.state not in self.STATES:
            raise ValueError(f"Resolution.state must be one of {self.STATES}, got {self.state!r}")

    @property
    def ok(self) -> bool:
        return self.state == "ready"


class Node(ABC):
    """Declare ports as class data; do the work in ``build``."""

    INPUTS: ClassVar[dict[str, Port]] = {}
    OUTPUTS: ClassVar[dict[str, Port]] = {}
    GROUPS: ClassVar[tuple[Group, ...]] = ()
    EVENTS_IN: ClassVar[dict[str, EventIn]] = {}
    EVENTS_OUT: ClassVar[dict[str, EventOut]] = {}
    MODULES: ClassVar[tuple[DisplayModule, ...]] = ()
    READS_SETTINGS: ClassVar[tuple[str, ...]] = ()
    SETTINGS: ClassVar[tuple[SettingSpec, ...]] = ()
    DYNAMIC: ClassVar[DynamicShape | None] = None
    DISPLAY_NAME: ClassVar[str | None] = None
    CATEGORY: ClassVar[str] = "Other"
    ACCEPTS_UNFINISHED: ClassVar[bool] = False

    def __init__(self, context: ExecutionContext | None = None):
        self.context = context or ExecutionContext()

    @property
    def uid(self) -> str:
        return self.context.uid

    @classmethod
    def resolve(cls, params: dict, env: Environment | None = None) -> Resolution:
        """Editing-time shape. Default: the static ports. Dynamic nodes override."""
        return Resolution("ready", dict(cls.INPUTS), dict(cls.OUTPUTS))

    @abstractmethod
    def build(self, **inputs) -> dict[str, Any]:
        """Produce every output key. Multi ports arrive as lists in wire order."""

    def diagnostics(self, inputs: dict) -> dict[str, list[str]]:
        """Optional advice about the current inputs, keyed by input name."""
        return {}

    def upstream_completion(self, port: str) -> Completion:
        return self.context.upstream_completion(port)

    def derived_seed(self, value: int | None) -> int:
        """A node's own seed when set; otherwise one derived from ``seed.base`` and this node's id, so one number
        reproduces a whole graph while no two nodes share a stream. Without a base the derived seed is 0."""
        if value is not None:
            return int(value)
        base = int(self.context.settings.get("seed.base"))
        if base < 0:
            return 0
        return int.from_bytes(hashlib.sha256(f"{base}:{self.uid}".encode()).digest()[:4], "big")

    def report_completion(self, completion: Completion) -> None:
        self.context.report_completion(completion)

    def emit(self, port: str, event: EventType) -> None:
        """Emit on one of this node's declared event outputs (run thread only)."""
        if port not in self.EVENTS_OUT:
            raise KeyError(f"{type(self).__name__} declares no event output {port!r}")
        hub = self.context.events
        if hub is not None:
            hub.emit(self.uid, port, event)

    def close(self) -> None:
        """Called once when the run ends, for nodes the runtime retained (those with EVENTS_IN)."""

    def validate_inputs(self, inputs: dict) -> None:
        for name, port in self.INPUTS.items():
            if port.required and name not in inputs:
                raise ValueError(f"{type(self).__name__}.build() missing required input {name!r}")
            if port.multi and name in inputs and not isinstance(inputs[name], list):
                raise TypeError(f"{type(self).__name__}.{name} is a multi port and must arrive as a list")
            if port.choices is not None and inputs.get(name) is not None and inputs[name] not in port.choices:
                raise ValueError(f"{type(self).__name__}.build(): {name}={inputs[name]!r} is not one of {port.choices}")

    def validate_outputs(self, outputs: dict) -> None:
        missing = set(self.OUTPUTS) - set(outputs)
        if missing:
            raise ValueError(f"{type(self).__name__}.build() did not produce {sorted(missing)}")

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls):
            return
        _check_declarations(cls)


def _check_declarations(cls: type[Node]) -> None:
    name = cls.__name__
    for label, ports in (("INPUTS", cls.INPUTS), ("OUTPUTS", cls.OUTPUTS)):
        if not isinstance(ports, dict):
            raise TypeError(f"{name}.{label} must be a dict")
        for key, port in ports.items():
            if not isinstance(port, Port) or port.name != key:
                raise TypeError(f"{name}.{label}[{key!r}] must be a Port named {key!r}")
    if not cls.OUTPUTS and cls.DYNAMIC is None:
        raise TypeError(f"{name} is concrete but declares no OUTPUTS (and is not DYNAMIC)")
    group_ids = [g.id for g in cls.GROUPS]
    if not all(isinstance(g, Group) for g in cls.GROUPS) or len(set(group_ids)) != len(group_ids):
        raise TypeError(f"{name}.GROUPS must be unique Group objects")
    for group in cls.GROUPS:
        if group.visible_when is not None and group.visible_when[0] not in cls.INPUTS:
            raise TypeError(f"{name}: group {group.id!r} visible_when names unknown port {group.visible_when[0]!r}")
    fields_ = [p for p in cls.INPUTS.values() if is_field(p)]
    for key, port in cls.INPUTS.items():
        if port.visible_when is not None and port.visible_when[0] not in cls.INPUTS:
            raise TypeError(f"{name}.INPUTS[{key!r}].visible_when names unknown port {port.visible_when[0]!r}")
        if port.group and port.group not in group_ids:
            raise TypeError(f"{name}.INPUTS[{key!r}] is in unknown group {port.group!r}")
    if len(fields_) > MAX_UNGROUPED_FIELDS and not cls.GROUPS:
        raise TypeError(f"{name} has {len(fields_)} fields; declare GROUPS (limit {MAX_UNGROUPED_FIELDS} without)")
    if cls.DYNAMIC is not None:
        for dep in cls.DYNAMIC.depends_on:
            if dep not in cls.INPUTS:
                raise TypeError(f"{name}.DYNAMIC depends on unknown port {dep!r}")
        if cls.resolve.__func__ is Node.resolve.__func__:
            raise TypeError(f"{name} is DYNAMIC but does not override resolve()")
    for key, ev in cls.EVENTS_IN.items():
        if not isinstance(ev, EventIn) or not (isinstance(ev.payload, type) and issubclass(ev.payload, EventType)):
            raise TypeError(f"{name}.EVENTS_IN[{key!r}] needs an EventIn with an EventType payload")
        if not callable(getattr(cls, ev.handler, None)):
            raise TypeError(f"{name}.EVENTS_IN[{key!r}] handler {ev.handler!r} is not a method")
    for key, ev in cls.EVENTS_OUT.items():
        if not isinstance(ev, EventOut) or not (isinstance(ev.payload, type) and issubclass(ev.payload, EventType)):
            raise TypeError(f"{name}.EVENTS_OUT[{key!r}] needs an EventOut with an EventType payload")
    if not isinstance(cls.MODULES, tuple) or not all(isinstance(m, DisplayModule) for m in cls.MODULES):
        raise TypeError(f"{name}.MODULES must be a tuple of DisplayModule")
    if len({m.name for m in cls.MODULES}) != len(cls.MODULES):
        raise TypeError(f"{name}.MODULES has duplicate names")
    if not isinstance(cls.READS_SETTINGS, tuple) or not all(isinstance(k, str) for k in cls.READS_SETTINGS):
        raise TypeError(f"{name}.READS_SETTINGS must be a tuple of keys")
    if cls.SETTINGS:
        cls.SETTINGS = tuple(dataclasses.replace(s, contributed_by=name) for s in cls.SETTINGS)
