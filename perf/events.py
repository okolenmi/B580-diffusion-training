"""The profiler's raw events, and which operator and module launched each device kernel.

``torch.profiler``'s own tables build an object and a call tree per event in Python; here events are read once into plain tuples.
A device kernel's ``linked_correlation_id`` is the correlation id of the operator it was launched from (the innermost one, e.g.
``aten::mm`` under ``aten::matmul``); the operator's ancestors come from the call tree per host thread, and its module from the
``perf|<module>|<phase>`` ranges the module hooks open around it. The driver and runtime events carry ids of their own that collide
with operator ids, so they are not used."""

from __future__ import annotations

from dataclasses import dataclass, field

from torch.autograd import DeviceType

ANNOTATION = "perf|"  # module ranges are named perf|<path>|<forward|recompute|backward>


@dataclass
class Op:
    thread: int
    start: int
    end: int
    name: str
    corr: int = 0
    raw: object = None   # the profiler event, for shapes and dtypes on demand


@dataclass
class EventTable:
    kernels: list[tuple[str, int, float, int]] = field(default_factory=list)       # name, start ns, duration us, linked correlation id
    ops: list[Op] = field(default_factory=list)
    ranges: list[tuple[int, int, int, str]] = field(default_factory=list)          # thread, start, end, name
    kinds: dict[str, int] = field(default_factory=dict)                            # "device/activity" -> count, for diagnostics


def read_events(events, *, keep_raw: bool = False) -> EventTable:
    table = EventTable()
    for event in events:
        on_cpu = event.device_type() == DeviceType.CPU
        activity = str(event.activity_type())
        key = f"{'cpu' if on_cpu else 'device'}/{activity}"
        table.kinds[key] = table.kinds.get(key, 0) + 1
        if not on_cpu:
            table.kernels.append((event.name(), event.start_ns(), event.duration_ns() / 1000.0, event.linked_correlation_id()))
        elif activity == "cpu_op":
            if not event.is_hidden_event():
                table.ops.append(Op(event.start_thread_id(), event.start_ns(), event.end_ns(), event.name(), event.correlation_id(), event if keep_raw else None))
        elif activity == "user_annotation":
            if event.name().startswith(ANNOTATION):
                table.ranges.append((event.start_thread_id(), event.start_ns(), event.end_ns(), event.name()))
    return table


class OpTree:
    """Operators sorted by thread and start, each with its parent (the operator it ran inside, -1 for a frontend op) and its own
    time, its duration minus that of the operators called inside it."""

    def __init__(self, ops: list[Op]):
        self.ops = sorted(ops, key=lambda o: (o.thread, o.start, -o.end))
        self.parent = [-1] * len(self.ops)
        self.own_us = [(o.end - o.start) / 1000.0 for o in self.ops]
        stack: list[int] = []
        thread = None
        for i, op in enumerate(self.ops):
            if op.thread != thread:
                stack, thread = [], op.thread
            while stack and self.ops[stack[-1]].end <= op.start:
                stack.pop()
            if stack:
                self.parent[i] = stack[-1]
                self.own_us[stack[-1]] -= (op.end - op.start) / 1000.0
            stack.append(i)
        self.by_corr = {op.corr: i for i, op in enumerate(self.ops) if op.corr}

    def chain(self, index: int) -> list[int]:
        """The operator and its ancestors, innermost first."""
        out = []
        while index >= 0:
            out.append(index)
            index = self.parent[index]
        return out


@dataclass
class Launch:
    name: str
    duration_us: float
    op: int = -1                 # innermost operator, index into the tree, -1 when unknown
    link: int = 0                # the kernel's linked correlation id (0: the profiler gave none)
    ranges: tuple[str, ...] = ()  # module ranges the launch happened in, innermost first


class Attribution:
    """Ties every kernel to its launching operator and module ranges."""

    def __init__(self, table: EventTable, tree: OpTree):
        self.table, self.tree = table, tree

    def launches(self) -> list[Launch]:
        out = [Launch(name, us, link=link) for name, _, us, link in self.table.kernels]
        by_thread: dict[int, list[tuple[int, int]]] = {}  # thread -> (launch time, kernel index)
        for i, (_, _, _, link) in enumerate(self.table.kernels):
            op = self.tree.by_corr.get(link, -1) if link else -1
            out[i].op = op
            if op >= 0:
                by_thread.setdefault(self.tree.ops[op].thread, []).append((self.tree.ops[op].start, i))
        ranges: dict[int, list[tuple[int, int, str]]] = {}
        for thread, start, end, name in self.table.ranges:
            ranges.setdefault(thread, []).append((start, end, name))
        for thread, items in by_thread.items():
            items.sort()
            pending = sorted(ranges.get(thread, ()))
            cursor, active = 0, []
            for time, i in items:
                while cursor < len(pending) and pending[cursor][0] <= time:
                    active.append(pending[cursor])
                    cursor += 1
                if active:
                    active = [r for r in active if r[1] > time]
                    out[i].ranges = tuple(name for _, _, name in sorted(active, reverse=True))
        return out
