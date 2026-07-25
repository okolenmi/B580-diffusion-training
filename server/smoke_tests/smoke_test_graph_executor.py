"""Checks server/graph_executor.py and server/nodegraph_registry.py.

Two kinds of check, deliberately kept separate: (1) real registry, real
node classes, real execution -- but limited to the domains that don't
need torch (schedule/loss), so this runs in any environment; (2)
synthetic Node subclasses for cycle detection and subclass-aware type
compatibility, since those don't need a real domain to exercise.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nodes.core import Node, Port
from server import graph_executor as ge
from server import nodegraph_registry


def check(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def test_real_registry_torch_free_execution():
    nodes = [
        ge.NodeSpec(id="n1", class_name="CosineLRScheduleNode",
                    params={"lr": 1e-4, "total_steps": 50}),
        ge.NodeSpec(id="n2", class_name="MinSNRLossWeightingNode", params={"gamma": 5.0}),
    ]
    results = ge.GraphExecutor(nodes, []).run()
    check(all(r.ok for r in results), f"expected both to succeed: {results}")
    check(results[0].outputs["schedule"]["_type"] == "CosineLRSchedule", "wrong output type")


def test_real_registry_missing_input_reported_not_crashed():
    nodes = [ge.NodeSpec(id="n1", class_name="ManagedDatasetSourceNode", params={})]
    results = ge.GraphExecutor(nodes, []).run()
    check(not results[0].ok, "missing required input should fail the node, not the process")
    check("dataset_root" in results[0].error, f"error should name the missing port: {results[0].error}")


def test_unknown_class_raises_graph_error():
    nodes = [ge.NodeSpec(id="n1", class_name="NotARealNode", params={})]
    try:
        ge.GraphExecutor(nodes, []).run()
        raise AssertionError("should have raised GraphError")
    except ge.GraphError:
        pass


class _HandleA:
    pass


class _HandleB(_HandleA):
    pass


class _SourceNode(Node):
    OUTPUTS = {"out": Port(name="out", type=_HandleB, required=True)}
    INPUTS = {"value": Port(name="value", type=int, required=True)}

    def build(self, **inputs):
        self.validate_inputs(inputs)
        result = {"out": _HandleB()}
        self.validate_outputs(result)
        return result


class _SinkNode(Node):
    OUTPUTS = {"result": Port(name="result", type=str, required=True)}
    INPUTS = {"thing": Port(name="thing", type=_HandleA, required=True)}  # base type, fed a subclass

    def build(self, **inputs):
        self.validate_inputs(inputs)
        result = {"result": type(inputs["thing"]).__name__}
        self.validate_outputs(result)
        return result


def test_subclass_aware_connection_allowed():
    fake_registry = {"_SourceNode": _SourceNode, "_SinkNode": _SinkNode}
    original = nodegraph_registry.get_registry
    nodegraph_registry.get_registry = lambda: fake_registry
    try:
        nodes = [ge.NodeSpec(id="a", class_name="_SourceNode", params={"value": 1}),
                 ge.NodeSpec(id="b", class_name="_SinkNode", params={})]
        edges = [ge.EdgeSpec(from_node="a", from_port="out", to_node="b", to_port="thing")]
        results = ge.GraphExecutor(nodes, edges).run()
        check(results[1].ok and results[1].outputs["result"] == "_HandleB",
              f"subclass output should satisfy base-typed input: {results}")
    finally:
        nodegraph_registry.get_registry = original


def test_cycle_detected():
    try:
        ge._topological_order(
            ["a", "b"],
            [ge.EdgeSpec("a", "result", "b", "thing"), ge.EdgeSpec("b", "result", "a", "thing")],
        )
        raise AssertionError("should have detected the cycle")
    except ge.GraphError:
        pass


def main():
    test_real_registry_torch_free_execution()
    test_real_registry_missing_input_reported_not_crashed()
    test_unknown_class_raises_graph_error()
    test_subclass_aware_connection_allowed()
    test_cycle_detected()
    print("All graph_executor checks passed.")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"FAIL: {e}")
        sys.exit(1)
