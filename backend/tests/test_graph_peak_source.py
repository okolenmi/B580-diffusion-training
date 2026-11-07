"""GraphPeakSource: what admission reads before it claims (MEM-04 #2).

The read half of the peak loop: fingerprint the graph (MEM-01), then look that key up in the peak store. Every failure
has to land on the same square -- *unknown*, never zero -- because the number that comes out is what the run will claim
(or, if unknown, whether it claims exclusively). The last test reads the declarations of the real node classes.

Run directly: python backend/tests/test_graph_peak_source.py
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.application.graph_peak_source import GraphPeakSource, ObservedPeak, dataset_name_of, unknown_peaks
from backend.application.memory_fingerprint import MemoryField
from backend.domain.graph import GraphDefinition, NodeSpec
from backend.infrastructure.graph import tree
from backend.infrastructure.graph.memory_fields import memory_fields_resolver
from backend.infrastructure.memory_peak_store import SqlitePeakStore
from backend.tests.support import check, finish
from next.graph.document import GraphDoc

_KEY = "shapes|PeakTrainer(batch_size=2,rank=64)"


def _resolver(class_name: str):
    if class_name == "PeakTrainer":
        return (MemoryField("batch_size", 1), MemoryField("rank", required=True))
    return None


def _graph(dataset: str | None = "shapes", **params) -> GraphDefinition:
    values = {"batch_size": 2, "rank": 64, **params}
    nodes = [NodeSpec("n_tttttt", "PeakTrainer", values)]
    if dataset is not None:
        nodes.append(NodeSpec("n_dddddd", "Images", {"dataset": dataset}))
    return GraphDefinition(GraphDoc(nodes=tuple(nodes)))


class _ExplodingStore:
    def peak_mb(self, key):
        raise RuntimeError("database is locked")


def _source(peaks) -> GraphPeakSource:
    return GraphPeakSource(peaks=peaks, resolve_memory_fields=_resolver)


def test_no_dataset_reference_is_unknown() -> None:
    print("\n== a graph that names no dataset: unknown ==")
    result = _source(_ExplodingStore()).observed(_graph(dataset=None))
    check(result == ObservedPeak(None, None), f"no fingerprint, no peak (got {result})")
    check(result.peak_record() is None, "and nothing to feed effective_memory")


def test_known_fingerprint_reads_the_remembered_peak() -> None:
    print("\n== known fingerprint + measured store: the remembered peak ==")
    with tempfile.TemporaryDirectory() as tmp:
        store = SqlitePeakStore(Path(tmp) / "peaks.db")
        store.record(_KEY, 7000.0)
        result = _source(store).observed(_graph())
        check(result == ObservedPeak(_KEY, 7000.0), f"key and peak come back together (got {result})")
        check(result.peak_record() == {_KEY: 7000.0}, "and it feeds effective_memory as a one-entry record")


def test_unmeasured_store_is_unknown_not_zero() -> None:
    print("\n== a known key nobody measured: the peak is unknown, not 0 ==")
    with tempfile.TemporaryDirectory() as tmp:
        result = _source(SqlitePeakStore(Path(tmp) / "peaks.db")).observed(_graph())
        check(result == ObservedPeak(_KEY, None), f"the key is known, the peak is not (got {result})")
        check(result.peak_record() is None, "so admission sees no record")


def test_no_store_still_carries_the_key() -> None:
    print("\n== no store wired: the row still gets its key ==")
    result = _source(None).observed(_graph())
    check(result == ObservedPeak(_KEY, None), f"key, no peak (got {result})")


def test_different_configuration_is_a_different_key() -> None:
    print("\n== another batch size or dataset is not the remembered configuration ==")
    with tempfile.TemporaryDirectory() as tmp:
        store = SqlitePeakStore(Path(tmp) / "peaks.db")
        store.record(_KEY, 7000.0)
        check(_source(store).observed(_graph(batch_size=4)).peak_mb is None, "a bigger batch is unmeasured")
        check(_source(store).observed(_graph(dataset="other")).peak_mb is None, "another dataset is unmeasured")


def test_unusable_fingerprint_is_unknown() -> None:
    print("\n== a required field that is not set: unknown ==")
    graph = GraphDefinition(GraphDoc(nodes=(
        NodeSpec("n_tttttt", "PeakTrainer", {}), NodeSpec("n_dddddd", "Images", {"dataset": "shapes"}))))
    check(_source(None).observed(graph) == ObservedPeak(None, None), "no rank, no fingerprint")


def test_store_errors_log_and_stay_unknown() -> None:
    print("\n== a store that raises: logged, unknown, admission goes on ==")

    class _Capture(logging.Handler):
        def __init__(self) -> None:
            super().__init__()
            self.records: list[logging.LogRecord] = []

        def emit(self, record: logging.LogRecord) -> None:
            self.records.append(record)

    handler = _Capture()
    logger = logging.getLogger("backend.application.graph_peak_source")
    logger.addHandler(handler)
    try:
        result = _source(_ExplodingStore()).observed(_graph())
    finally:
        logger.removeHandler(handler)
    check(result == ObservedPeak(None, None), f"unknown, never a zero (got {result})")
    check(any(r.levelno >= logging.WARNING and r.exc_info for r in handler.records), "the failure was logged with its traceback")


def test_dataset_name_and_unknown_peaks() -> None:
    print("\n== the dataset name is the first node's `dataset` param ==")
    odd = GraphDefinition(GraphDoc(nodes=(
        NodeSpec("n_aaaaaa", "X", {"dataset": 7}),
        NodeSpec("n_bbbbbb", "Y", {"dataset": ""}),
        NodeSpec("n_cccccc", "Z", {"dataset": "late"}),
    )))
    check(dataset_name_of(odd) == "late", "a non-string and an empty name are skipped, not trusted")
    check(dataset_name_of(_graph(dataset=None)) is None, "no name at all is None")
    check(unknown_peaks(_graph()) == ObservedPeak(None, None), "the no-store source is unknown")


def test_real_nodes_declare_what_the_peak_depends_on() -> None:
    print("\n== the real node classes: declared fields resolve with their defaults ==")
    resolve = memory_fields_resolver(tree.NodeRegistry.default())
    batcher = {f.name: f for f in resolve("BatcherNode") or ()}
    check("batch_size" in batcher and batcher["batch_size"].default == 1, f"the batcher declares its batch size (got {batcher})")
    check(resolve("LoadModelNode") is not None and resolve("NoSuchNode") is None, "declaring and unknown classes resolve distinctly")


def main() -> None:
    tests = [
        test_no_dataset_reference_is_unknown,
        test_known_fingerprint_reads_the_remembered_peak,
        test_unmeasured_store_is_unknown_not_zero,
        test_no_store_still_carries_the_key,
        test_different_configuration_is_a_different_key,
        test_unusable_fingerprint_is_unknown,
        test_store_errors_log_and_stay_unknown,
        test_dataset_name_and_unknown_peaks,
        test_real_nodes_declare_what_the_peak_depends_on,
    ]
    for test in tests:
        test()
    finish()


if __name__ == "__main__":
    main()
