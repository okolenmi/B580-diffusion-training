"""Where device time comes from: kernels grouped by the operator, shapes, dtypes and module that launched them."""

from __future__ import annotations

from dataclasses import dataclass, field

from .categories import Categorizer
from .events import Attribution, EventTable, OpTree
from .modules import region_of, role_of

DTYPES = {"float": "f32", "double": "f64", "c10::BFloat16": "bf16", "c10::Half": "f16", "long int": "i64", "int": "i32", "bool": "bool",
          "signed char": "i8", "unsigned char": "u8", "c10::Float8_e4m3fn": "f8"}
MATMULS = {"aten::mm": (0, 1), "aten::addmm": (1, 2), "aten::bmm": (0, 1), "aten::baddbmm": (1, 2)}  # operator -> positions of the two factors


def matmul_flops(op: str, shapes: list) -> float | None:
    """FLOPs of one matmul operator call from its input shapes, None for anything else."""
    where = MATMULS.get(op)
    if where is None:
        return None
    try:
        a, b = shapes[where[0]], shapes[where[1]]
        batch = 1
        for d in a[:-2]:
            batch *= d
        return 2.0 * batch * a[-2] * a[-1] * b[-1]
    except (IndexError, TypeError):
        return None


def compact(shapes: list) -> str:
    return ", ".join("x".join(str(d) for d in s) if s else "-" for s in shapes)


@dataclass
class Provenance:
    rows: list[dict] = field(default_factory=list)                              # one per (kernel, operator, shapes, dtypes, module, phase), totals over the window
    summary: list[dict] = field(default_factory=list)                           # the same without kernel names and shapes: what a kind of work costs in all
    modules: dict[str, dict[str, list[float]]] = field(default_factory=dict)  # path -> phase -> [own us, inclusive us]
    coverage: dict[str, float] = field(default_factory=dict)                  # shares of kernel time with a known operator / module
    kinds: dict[str, int] = field(default_factory=dict)


class ProvenanceBuilder:
    def __init__(self, table: EventTable, tree: OpTree, categorizer: Categorizer):
        self.table, self.tree, self.categorizer = table, tree, categorizer
        self.shape_errors = 0

    def _shapes(self, event) -> tuple[str, str, list]:
        """Input shapes and dtypes of an operator event; a build whose shape metadata fails (seen as a MemoryError from the profiler's C++) gives none."""
        if event is None:
            return "", "", []
        try:
            raw = event.shapes()
            return compact(raw), ",".join(DTYPES.get(d, d) for d in event.dtypes()), raw
        except (MemoryError, RuntimeError, ValueError, TypeError):
            self.shape_errors += 1
            return "", "", []

    def build(self) -> Provenance:
        launches = Attribution(self.table, self.tree).launches()
        groups: dict[tuple, dict] = {}
        kinds: dict[tuple, dict] = {}
        modules: dict[str, dict[str, list[float]]] = {}
        total = with_op = with_module = no_link = lost = 0.0
        shape_cache: dict[int, tuple[str, str, list]] = {}
        for launch in launches:
            total += launch.duration_us
            op = op_name = chain = shapes = dtypes = ""
            raw_shapes: list = []
            if launch.op >= 0:
                with_op += launch.duration_us
                if launch.op not in shape_cache:
                    shape_cache[launch.op] = self._shapes(self.tree.ops[launch.op].raw)
                shapes, dtypes, raw_shapes = shape_cache[launch.op]
                ancestors = self.tree.chain(launch.op)
                op_name = self.tree.ops[launch.op].name
                root = self.tree.ops[ancestors[-1]].name
                chain = op_name if root == op_name else f"{root} > {op_name}"
            if launch.op < 0:
                if launch.link:
                    lost += launch.duration_us
                else:
                    no_link += launch.duration_us
            module = phase = ""
            for depth, name in enumerate(launch.ranges):
                _, path, phase_name = name.split("|", 2)
                entry = modules.setdefault(path, {}).setdefault(phase_name, [0.0, 0.0])
                entry[1] += launch.duration_us
                if depth == 0:
                    entry[0] += launch.duration_us
                    module, phase = path, phase_name
                    with_module += launch.duration_us
            key = (launch.name, chain, shapes, dtypes, role_of(module) if module else "", phase)
            row = groups.get(key)
            if row is None:
                flops = matmul_flops(op_name, raw_shapes) if launch.op >= 0 else None
                row = groups[key] = {"category": self.categorizer.category(launch.name), "kernel": launch.name, "op": chain, "shapes": shapes, "dtypes": dtypes,
                                     "module": key[4], "phase": phase, "region": region_of(module) if module else "", "calls": 0, "device_us": 0.0, "flops": flops}
            row["calls"] += 1
            row["device_us"] += launch.duration_us
            kind_key = (row["category"], op_name or "?", dtypes, key[4], phase)
            kind = kinds.get(kind_key)
            if kind is None:
                kind = kinds[kind_key] = {"category": row["category"], "op": op_name or "?", "dtypes": dtypes, "module": key[4], "phase": phase, "calls": 0, "device_us": 0.0}
            kind["calls"] += 1
            kind["device_us"] += launch.duration_us
        rows = sorted(groups.values(), key=lambda r: -r["device_us"])
        summary = sorted(kinds.values(), key=lambda r: -r["device_us"])
        share = (lambda part: part / total if total else 0.0)
        return Provenance(rows, summary, modules, {"operator": share(with_op), "module": share(with_module), "kernel_ms": total / 1000.0, "shape_errors": self.shape_errors,
                                                                      "no_link": share(no_link), "link_to_unknown_operator": share(lost)}, dict(self.table.kinds))

