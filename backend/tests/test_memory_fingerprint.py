"""graph_fingerprint: what makes two runs "the same configuration" for peak purposes.

It is computed before a child is spawned, so it must not import torch. The key is the dataset plus, for every node
that declares ``memory_fields``, its class and the declared values (the node's param, else the port's default). A field
another node feeds, a required one that is not set, or no declaring node at all is *unknown*, never defaulted.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.application.memory_fingerprint import (
    GraphMemoryFingerprint,
    MemoryField,
    UnknownFingerprint,
    graph_fingerprint,
)
from backend.domain.graph import Edge, GraphDefinition, NodeSpec
from next.graph.document import GraphDoc

_FIELDS = {
    "Trainer": (MemoryField("accumulation", 1), MemoryField("graph", False)),
    "Batcher": (MemoryField("batch_size", 1),),
    "Adapter": (MemoryField("rank", required=True), MemoryField("checkpointing", False)),
}


def _resolve(class_name: str):
    return _FIELDS.get(class_name)


def _node(uid: str, class_name: str, **params) -> NodeSpec:
    return NodeSpec(uid=f"n_{uid * 6}", class_name=class_name, params=params)


def _graph(*nodes: NodeSpec, edges=()) -> GraphDefinition:
    return GraphDefinition(GraphDoc(nodes=nodes, edges=tuple(edges)))


def _fingerprint(graph: GraphDefinition, dataset: str = "shapes"):
    return graph_fingerprint(graph, dataset, resolve_memory_fields=_resolve)


def _full() -> GraphDefinition:
    return _graph(_node("a", "Adapter", rank=64), _node("b", "Batcher", batch_size=2), _node("t", "Trainer"))


def test_fingerprint_computed_without_torch() -> None:
    """A clean interpreter imports the module and computes a key without loading torch."""
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "from backend.application.memory_fingerprint import MemoryField, graph_fingerprint\n"
        "from backend.domain.graph import GraphDefinition\n"
        "graph_fingerprint(GraphDefinition(), 'd', resolve_memory_fields=lambda c: None)\n"
        "assert 'torch' not in sys.modules\n"
    ) % str(Path(__file__).resolve().parents[2])
    done = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr


def test_key_is_dataset_then_each_declaring_node_sorted() -> None:
    result = _fingerprint(_full())
    assert isinstance(result, GraphMemoryFingerprint), result
    assert result.key() == "shapes|Adapter(rank=64,checkpointing=False)|Batcher(batch_size=2)|Trainer(accumulation=1,graph=False)"


def test_default_applies_when_the_param_is_unset() -> None:
    """The port's default is the value the run uses, so it is the value the key records."""
    explicit = _fingerprint(_graph(_node("a", "Adapter", rank=8, checkpointing=False)))
    implicit = _fingerprint(_graph(_node("a", "Adapter", rank=8)))
    assert explicit == implicit


def test_spelling_of_a_number_does_not_split_the_key() -> None:
    """JSON hands a number over as int, float or text; 2, 2.0 must be one configuration."""
    assert _fingerprint(_graph(_node("b", "Batcher", batch_size=2))).key() == \
        _fingerprint(_graph(_node("b", "Batcher", batch_size=2.0))).key()


def test_changing_a_declared_value_changes_the_key() -> None:
    base = _fingerprint(_full()).key()
    bigger = _graph(_node("a", "Adapter", rank=64), _node("b", "Batcher", batch_size=4), _node("t", "Trainer"))
    checkpointed = _graph(_node("a", "Adapter", rank=64, checkpointing=True), _node("b", "Batcher", batch_size=2),
                          _node("t", "Trainer"))
    assert _fingerprint(bigger).key() != base and _fingerprint(checkpointed).key() != base
    assert _fingerprint(_full(), "other").key() != base, "the dataset is part of the key"


def test_undeclared_params_and_node_order_do_not_matter() -> None:
    """A caption or a node's position changes nothing about the peak."""
    shuffled = _graph(_node("t", "Trainer"), _node("b", "Batcher", batch_size=2, note="x"), _node("a", "Adapter", rank=64))
    assert _fingerprint(shuffled).key() == _fingerprint(_full()).key()


def test_required_field_missing_is_unknown() -> None:
    result = _fingerprint(_graph(_node("a", "Adapter")))
    assert isinstance(result, UnknownFingerprint) and "rank" in result.reason


def test_a_wired_field_is_unknown() -> None:
    """The value comes from another node, which the server cannot evaluate."""
    wired = _graph(_node("a", "Adapter", rank=8), _node("b", "Batcher"),
                   edges=(Edge("n_aaaaaa", "out", "n_bbbbbb", "batch_size"),))
    result = _fingerprint(wired)
    assert isinstance(result, UnknownFingerprint) and "batch_size" in result.reason


def test_no_declaring_node_is_unknown() -> None:
    result = _fingerprint(_graph(_node("x", "Unrelated")))
    assert isinstance(result, UnknownFingerprint), result


def main() -> None:
    """Listed, not discovered: `scripts/check_test_wiring.py` fails this file when a `def test_*` is left out."""
    tests = [
        test_fingerprint_computed_without_torch,
        test_key_is_dataset_then_each_declaring_node_sorted,
        test_default_applies_when_the_param_is_unset,
        test_spelling_of_a_number_does_not_split_the_key,
        test_changing_a_declared_value_changes_the_key,
        test_undeclared_params_and_node_order_do_not_matter,
        test_required_field_missing_is_unknown,
        test_a_wired_field_is_unknown,
        test_no_declaring_node_is_unknown,
    ]
    for test in tests:
        test()
    print()
    print("=" * 60)
    print(f"SMOKE TEST: ALL {len(tests)} CHECKS PASSED")


if __name__ == "__main__":
    main()
