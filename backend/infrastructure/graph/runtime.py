"""ReflectedGraphRuntime -- validate + execute graphs over real Node classes.

The authoritative half of the graph subsystem (``GraphRuntime`` port):

* ``validate`` computes one complete, deterministic issue list using
  real Python types (``issubclass`` for edges, declared param types and
  ``Port.choices`` for params) with the node's own params-aware
  ``resolve_inputs``/``resolve_outputs`` hooks -- see doc 05 section 4
  for the issue-code table;
* ``execute`` runs topologically, wires edge outputs into inputs
  (edge wins over a param, as legacy did), times each ``build()``,
  JSON-describes outputs for reporting while keeping the real objects
  for downstream wiring, stops between nodes when the cancel event is
  set, and reports every node through ``on_node_done``;
* ``release_memory`` is the legacy worker's ``finally``: gc first
  (cycles the refcount cannot free), then the caching allocator (which
  is what actually returns device memory to the driver). Injected, so
  tests never touch ``core/`` or the GPU.

Defensive posture: ``execute`` re-validates first and never raises for
graph/node problems -- they come back as an outcome the supervisor
turns into a status.
"""

from __future__ import annotations

import gc
import logging
import threading
import time
from pathlib import Path
from typing import Any

from nodes.core import ExecutionContext

from ...application.ports.graph_runtime import (
    ISSUE_ERROR,
    GraphIssue,
    GraphOutcome,
    GraphRuntime,
    NodeDoneCallback,
)
from ...domain.graph import GraphDefinition, NodeResult
from .discovery import NodeRegistry

logger = logging.getLogger(__name__)

# Types a JSON literal can meaningfully be checked against. Ports typed
# with a class/protocol (handles, callables, model wrappers) or a
# typing generic are deliberately skipped: no JSON value could ever
# satisfy them, so an editor always "fails" -- presence of required
# ports is covered by missing_required_input instead.
_WIRE_SAFE_TYPES = (bool, int, float, str, list, dict, tuple)


def _default_memory_releaser() -> None:
    """gc, then the device caching allocator (lazy bridge: importing
    ``core.comfy_setup`` pulls the torch/ComfyUI chain, so it happens
    on the first *run*, never at startup)."""
    gc.collect()
    try:
        from core.comfy_setup import xpu_empty_cache  # noqa: PLC0415 -- lazy bridge

        xpu_empty_cache()
    except Exception as exc:  # noqa: BLE001 -- best-effort, never masks a result
        logger.info("xpu cache release unavailable (non-fatal): %s", exc)


def _topological_order(node_ids: list[str], edges) -> list[str]:
    """Kahn's algorithm with stable tie-breaking (legacy port). Raises
    ValueError listing the stuck nodes when a cycle exists."""
    depends_on: dict[str, set[str]] = {nid: set() for nid in node_ids}
    known = set(node_ids)
    for edge in edges:
        if edge.to_node in known and edge.from_node in known:
            depends_on[edge.to_node].add(edge.from_node)

    ordered: list[str] = []
    remaining = set(node_ids)
    while remaining:
        ready = [nid for nid in remaining if depends_on[nid] <= set(ordered)]
        if not ready:
            raise ValueError(f"cycle detected among nodes: {sorted(remaining)}")
        ready.sort()
        ordered.extend(ready)
        remaining -= set(ready)
    return ordered


def _type_mismatch(port_type: Any, value: Any) -> bool:
    """True when a literal param value cannot satisfy the declared port
    type. ``None`` always passes (it means "absent/use the default" for
    an optional port, per Node.validate_inputs); non-wire-safe types are
    never judged (see ``_WIRE_SAFE_TYPES``)."""
    if value is None or port_type is Any or not isinstance(port_type, type):
        return False
    if port_type not in _WIRE_SAFE_TYPES and port_type is not Path:
        return False
    if port_type is float:
        # ints are fine in a float field (JSON has one number type);
        # bools are not, and isinstance(True, int) is True -- guard it.
        return isinstance(value, bool) or not isinstance(value, (int, float))
    if port_type is int:
        return isinstance(value, bool) or not isinstance(value, int)
    if port_type is bool:
        return not isinstance(value, bool)
    if port_type is str:
        return not isinstance(value, str)
    if port_type is Path:
        return not isinstance(value, (str, Path))  # editors send paths as strings
    if port_type is list:
        return not isinstance(value, list)
    if port_type is dict:
        return not isinstance(value, dict)
    if port_type is tuple:
        return not isinstance(value, (list, tuple))  # JSON arrays arrive as lists
    return not isinstance(value, port_type)


def _incompatible_types(out_type: Any, in_type: Any) -> bool:
    """Real issubclass against actual Port.type objects (a
    FusedOptimizerHandle output satisfies an OptimizerHandle input
    because it *is* one) -- never string comparison. ``Any`` or a
    non-class annotation on either side passes (legacy posture)."""
    if in_type is Any or out_type is Any:
        return False
    if isinstance(out_type, type) and isinstance(in_type, type):
        return not issubclass(out_type, in_type)
    return False


def _describe(value: Any) -> Any:
    """JSON-safe view of a build() output: plain JSON passes through,
    containers recurse (capped), everything else becomes a short
    type+repr summary -- the real object only ever feeds the next
    node's build(), never serialization."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_describe(v) for v in value[:20]]
    if isinstance(value, dict):
        return {str(k): _describe(v) for k, v in list(value.items())[:20]}
    return {"_type": type(value).__name__, "_repr": repr(value)[:200]}


class ReflectedGraphRuntime(GraphRuntime):
    def __init__(
        self,
        registry: NodeRegistry,
        *,
        memory_releaser=_default_memory_releaser,
    ) -> None:
        self._registry = registry
        self._memory_releaser = memory_releaser

    # ------------------------------------------------------------------
    # GraphRuntime port
    # ------------------------------------------------------------------

    def validate(self, graph: GraphDefinition) -> tuple[GraphIssue, ...]:
        issues: list[GraphIssue] = []
        classes, _ = self._registry.load()

        # -- pass 1: node identity + params-aware shape resolution ------
        seen: set[str] = set()
        resolved_in: dict[str, dict] = {}
        resolved_out: dict[str, dict] = {}
        for node in graph.nodes:
            if not node.id:
                issues.append(
                    GraphIssue(ISSUE_ERROR, "invalid_node_id", "node id must not be empty")
                )
            elif node.id in seen:
                issues.append(
                    GraphIssue(
                        ISSUE_ERROR,
                        "duplicate_node_id",
                        f"node id {node.id!r} appears more than once",
                        node_id=node.id,
                    )
                )
            seen.add(node.id)

            cls = classes.get(node.class_name)
            if cls is None:
                issues.append(
                    GraphIssue(
                        ISSUE_ERROR,
                        "unknown_class",
                        f"no node class named {node.class_name!r} is registered",
                        node_id=node.id,
                    )
                )
                continue
            resolved_in[node.id] = self._resolve(
                cls, cls.resolve_inputs, node, issues, "inputs"
            )
            resolved_out[node.id] = self._resolve(
                cls, cls.resolve_outputs, node, issues, "outputs"
            )

        # -- pass 2: params vs the resolved input shape -----------------
        for node in graph.nodes:
            inputs = resolved_in.get(node.id)
            if inputs is None:
                continue  # unknown class or a failed shape resolution
            fed = {e.to_port for e in graph.edges_to(node.id)}
            for name, port in inputs.items():
                provided = name in node.params
                if port.required and not provided and name not in fed:
                    issues.append(
                        GraphIssue(
                            ISSUE_ERROR,
                            "missing_required_input",
                            f"{node.class_name}.{name} is required but neither a "
                            f"param nor an edge provides it",
                            node_id=node.id,
                            param=name,
                        )
                    )
                if provided:
                    value = node.params[name]
                    if (
                        port.choices is not None
                        and value is not None
                        and value not in port.choices
                    ):
                        issues.append(
                            GraphIssue(
                                ISSUE_ERROR,
                                "invalid_choice",
                                f"{node.class_name}.{name}={value!r} is not one of "
                                f"{list(port.choices)}",
                                node_id=node.id,
                                param=name,
                            )
                        )
                    if _type_mismatch(port.type, value):
                        type_name = getattr(port.type, "__name__", str(port.type))
                        issues.append(
                            GraphIssue(
                                ISSUE_ERROR,
                                "type_mismatch",
                                f"{node.class_name}.{name} expects {type_name}, "
                                f"got {type(value).__name__}",
                                node_id=node.id,
                                param=name,
                            )
                        )
            for key in node.params:
                if key not in inputs:
                    issues.append(
                        GraphIssue(
                            ISSUE_ERROR,
                            "unknown_param",
                            f"{node.class_name} has no input named {key!r}",
                            node_id=node.id,
                            param=key,
                        )
                    )

        # -- pass 3: edges ----------------------------------------------
        for index, edge in enumerate(graph.edges):
            from_spec = graph.node(edge.from_node)
            to_spec = graph.node(edge.to_node)
            if from_spec is None or to_spec is None:
                missing = edge.from_node if from_spec is None else edge.to_node
                issues.append(
                    GraphIssue(
                        ISSUE_ERROR,
                        "edge_unknown_node",
                        f"edge references node {missing!r}, which is not in the graph",
                        node_id=missing,
                        edge_index=index,
                    )
                )
                continue
            outputs = resolved_out.get(from_spec.id)
            inputs = resolved_in.get(to_spec.id)
            out_port = outputs.get(edge.from_port) if outputs is not None else None
            in_port = inputs.get(edge.to_port) if inputs is not None else None
            if outputs is not None and out_port is None:
                issues.append(
                    GraphIssue(
                        ISSUE_ERROR,
                        "unknown_output_port",
                        f"{from_spec.class_name} has no output port {edge.from_port!r}",
                        node_id=from_spec.id,
                        edge_index=index,
                    )
                )
            if inputs is not None and in_port is None:
                issues.append(
                    GraphIssue(
                        ISSUE_ERROR,
                        "unknown_input_port",
                        f"{to_spec.class_name} has no input port {edge.to_port!r}",
                        node_id=to_spec.id,
                        edge_index=index,
                    )
                )
            if out_port is not None and in_port is not None and _incompatible_types(
                out_port.type, in_port.type
            ):
                issues.append(
                    GraphIssue(
                        ISSUE_ERROR,
                        "incompatible_types",
                        f"{from_spec.class_name}.{edge.from_port} "
                        f"({getattr(out_port.type, '__name__', out_port.type)}) does "
                        f"not satisfy {to_spec.class_name}.{edge.to_port} "
                        f"({getattr(in_port.type, '__name__', in_port.type)})",
                        node_id=to_spec.id,
                        edge_index=index,
                    )
                )

        # -- pass 4: acyclicity (only meaningful with unique ids) -------
        if len(seen) == len(graph.nodes) and "" not in seen:
            try:
                _topological_order(list(dict.fromkeys(graph.node_ids)), graph.edges)
            except ValueError as exc:
                issues.append(
                    GraphIssue(
                        ISSUE_ERROR,
                        "cycle",
                        str(exc).replace("ValueError: ", ""),
                    )
                )
        return tuple(issues)

    def execute(
        self,
        graph: GraphDefinition,
        *,
        cancel_event: threading.Event,
        on_node_done: NodeDoneCallback | None = None,
    ) -> GraphOutcome:
        errors = [i for i in self.validate(graph) if i.severity == ISSUE_ERROR]
        if errors:
            extra = f" (+{len(errors) - 1} more)" if len(errors) > 1 else ""
            return GraphOutcome(results=(), error=f"graph invalid: {errors[0].message}{extra}")

        classes, _ = self._registry.load()
        try:
            order = _topological_order(list(dict.fromkeys(graph.node_ids)), graph.edges)
        except ValueError as exc:  # unreachable post-validate; defensive
            return GraphOutcome(results=(), error=str(exc))

        # monitor_bus=None: live-monitor streaming is M5's frontend
        # decision (nodes handle None -- see nodes/monitor/training_progress).
        context = ExecutionContext(monitor_bus=None, cancel_event=cancel_event)
        outputs_by_node: dict[str, dict] = {}
        results: list[NodeResult] = []

        for node_id in order:
            if cancel_event.is_set():
                break  # stop requested; whatever ran so far is reported
            spec = graph.node(node_id)
            cls = classes[spec.class_name]
            inputs = dict(spec.params)
            for edge in graph.edges_to(node_id):
                inputs[edge.to_port] = outputs_by_node[edge.from_node][edge.from_port]

            started = time.monotonic()
            try:
                outputs = cls(context).build(**inputs)
                described = {k: _describe(v) for k, v in outputs.items()}
            except Exception as exc:  # noqa: BLE001 -- a failing node is a normal outcome
                result = NodeResult(
                    node_id=node_id,
                    ok=False,
                    outputs={},
                    error=f"{type(exc).__name__}: {exc}",
                    duration_ms=(time.monotonic() - started) * 1000.0,
                )
                results.append(result)
                self._notify(on_node_done, result)
                return GraphOutcome(results=tuple(results), error=result.error)

            outputs_by_node[node_id] = outputs  # real objects feed the next node
            result = NodeResult(
                node_id=node_id,
                ok=True,
                outputs=described,
                duration_ms=(time.monotonic() - started) * 1000.0,
            )
            results.append(result)
            self._notify(on_node_done, result)
        return GraphOutcome(results=tuple(results), error=None)

    def release_memory(self) -> None:
        self._memory_releaser()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve(self, cls: type, hook, node, issues: list[GraphIssue], label: str):
        """Call a node's params-aware shape hook; a raise is a validation
        issue, never an exception escaping validate()."""
        try:
            return hook(dict(node.params))
        except Exception as exc:  # noqa: BLE001 -- node's own bug, reported not raised
            issues.append(
                GraphIssue(
                    ISSUE_ERROR,
                    "shape_resolution_failed",
                    f"{node.class_name}.resolve_{label} raised "
                    f"{type(exc).__name__}: {exc}",
                    node_id=node.id,
                )
            )
            return None

    @staticmethod
    def _notify(callback: NodeDoneCallback | None, result: NodeResult) -> None:
        if callback is None:
            return
        try:
            callback(result)
        except Exception:  # noqa: BLE001 -- progress reporting must not abort a run
            logger.exception(
                "on_node_done callback failed for node %r", result.node_id
            )
