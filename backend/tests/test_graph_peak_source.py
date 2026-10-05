"""GraphPeakSource: what admission reads before it claims (MEM-04 #2).

The read half of the peak loop: fingerprint the graph (MEM-01) against
the dataset's latent shapes, then look that key up in the peak store.
Every failure has to land on the same square -- *unknown*, never zero --
because the number that comes out is what the run will claim (or, if
unknown, whether it claims exclusively).

Run directly: python backend/tests/test_graph_peak_source.py
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.application.graph_peak_source import (
    GraphPeakSource,
    ObservedPeak,
    dataset_name_of,
    unknown_peaks,
)
from backend.application.memory_fingerprint import (
    GraphMemoryFingerprint,
    UnknownFingerprint,
    graph_fingerprint,
)
from backend.application.ports.dataset_library import LatentBucket
from backend.domain.graph import GraphDefinition, GraphNodeSpec
from backend.infrastructure.graph.discovery import NodeRegistry, memory_fields_resolver
from backend.infrastructure.memory_peak_store import SqlitePeakStore
from backend.tests.support import check, finish

# The five fields the fingerprint requires, declared by one node the way
# the real trainer node declares them.
_FIELDS = ("model", "batch_size", "rank", "checkpointing", "optimizer")
_RESOLVER_MAP = {"PeakTrainerNode": _FIELDS}
_KEY = "sdxl|2|1024|1024|64|True|adamw"

_BUCKETS = (
    LatentBucket(height=512, width=512, count=50),
    LatentBucket(height=1024, width=1024, count=10),
)


def _resolver(class_name: str) -> tuple[str, ...] | None:
    return _RESOLVER_MAP.get(class_name)


def _graph(*, dataset: str | None = "shapes") -> GraphDefinition:
    trainer = GraphNodeSpec(
        id="t",
        class_name="PeakTrainerNode",
        params={
            "model": "sdxl",
            "batch_size": 2,
            "rank": 64,
            "checkpointing": True,
            "optimizer": "adamw",
        },
    )
    nodes = [trainer]
    if dataset is not None:
        nodes.append(
            GraphNodeSpec(
                id="d",
                class_name="PeakDatasetNode",
                params={"dataset_root": dataset},
            )
        )
    return GraphDefinition(nodes=tuple(nodes), edges=())


class _Shapes:
    """LatentShapes stub: the buckets its dataset answers with."""

    def __init__(
        self,
        buckets: tuple[LatentBucket, ...] = _BUCKETS,
        error: Exception | None = None,
    ) -> None:
        self.calls: list[str] = []
        self._buckets = buckets
        self._error = error

    def latent_buckets(self, name: str) -> tuple[LatentBucket, ...]:
        self.calls.append(name)
        if self._error is not None:
            raise self._error
        return self._buckets


def _source(shapes, peaks, resolver=_resolver) -> GraphPeakSource:
    return GraphPeakSource(
        datasets=shapes, peaks=peaks, resolve_memory_fields=resolver
    )


def test_no_dataset_reference_is_unknown_and_touches_nothing() -> None:
    print("\n== a graph that names no dataset: unknown, and no lookup ==")
    exploding = _Shapes(error=AssertionError("must not be asked"))
    result = _source(exploding, peaks=None).observed(_graph(dataset=None))
    check(
        result == ObservedPeak(None, None),
        f"no fingerprint, no peak (got {result})",
    )
    check(exploding.calls == [], "the dataset source was never asked")
    check(result.peak_record() is None, "and nothing to feed effective_memory")


def test_known_fingerprint_reads_the_remembered_peak() -> None:
    print("\n== known fingerprint + measured store: the remembered peak ==")
    with tempfile.TemporaryDirectory() as tmp:
        store = SqlitePeakStore(Path(tmp) / "peaks.db")
        store.record(_KEY, 7000.0)
        shapes = _Shapes()
        result = _source(shapes, store).observed(_graph())
        check(
            result == ObservedPeak(_KEY, 7000.0),
            f"key and peak come back together (got {result})",
        )
        check(shapes.calls == ["shapes"], "the dataset was asked once, by name")
        check(
            result.peak_record() == {_KEY: 7000.0},
            "and it feeds effective_memory as a one-entry record",
        )


def test_unmeasured_store_is_unknown_not_zero() -> None:
    print("\n== known fingerprint, never measured: None, never 0.0 ==")
    with tempfile.TemporaryDirectory() as tmp:
        store = SqlitePeakStore(Path(tmp) / "peaks.db")
        result = _source(_Shapes(), store).observed(_graph())
        check(
            result == ObservedPeak(_KEY, None),
            f"the key is known and the peak is None (got {result})",
        )
        check(result.peak_record() is None, "so there is no record to read")
        check(result.peak_mb is None and result.peak_mb != 0,
              "None is not 0.0 -- zero would claim the run needs nothing")


def test_no_store_still_carries_the_key() -> None:
    print("\n== no store wired: the row still gets its fingerprint key ==")
    result = _source(_Shapes(), peaks=None).observed(_graph())
    check(
        result == ObservedPeak(_KEY, None),
        f"key computed, peak unknown (got {result})",
    )


def test_shape_errors_log_and_stay_unknown() -> None:
    print("\n== a store that will not answer: unknown, loudly ==")
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    logger = logging.getLogger("backend.application.graph_peak_source")
    handler = _Capture()
    logger.addHandler(handler)
    try:
        result = _source(
            _Shapes(error=RuntimeError("disk on fire")), peaks=None
        ).observed(_graph())
    finally:
        logger.removeHandler(handler)
    check(
        result == ObservedPeak(None, None),
        f"the lookup failed to unknown, not to zero (got {result})",
    )
    warnings = [r for r in records if r.levelno == logging.WARNING]
    check(bool(warnings), "the failure was logged as a warning")
    check(
        any("shapes" in (r.getMessage() or "") for r in warnings),
        "and the warning names the dataset it could not look up",
    )
    check(
        any(r.exc_info for r in warnings),
        "and carries the traceback (every except logs, with cause)",
    )


def test_shapeless_dataset_is_unknown() -> None:
    print("\n== rows with no latent size: shapeless -> unknown ==")
    shapes = _Shapes(buckets=(LatentBucket(height=0, width=0, count=9),))
    result = _source(shapes, peaks=None).observed(_graph())
    check(
        result == ObservedPeak(None, None),
        f"a 0x0 bucket is not a configuration (got {result})",
    )


def test_missing_declared_fields_is_unknown() -> None:
    print("\n== a graph whose node declares nothing: unknown ==")
    result = _source(_Shapes(), peaks=None, resolver=lambda name: None).observed(
        _graph()
    )
    check(
        result == ObservedPeak(None, None),
        f"no declared fields, no fingerprint (got {result})",
    )


def test_dataset_name_of() -> None:
    print("\n== the dataset name comes from the first node that names one ==")
    check(dataset_name_of(_graph()) == "shapes", "read off dataset_root")
    check(dataset_name_of(_graph(dataset=None)) is None, "absent -> None")
    odd = GraphDefinition(
        nodes=(
            GraphNodeSpec(id="a", class_name="X", params={"dataset_root": 7}),
            GraphNodeSpec(id="b", class_name="Y", params={"dataset_root": ""}),
            GraphNodeSpec(id="c", class_name="Z", params={"dataset_root": "late"}),
        ),
        edges=(),
    )
    check(
        dataset_name_of(odd) == "late",
        "non-strings and empty names are skipped, not taken",
    )


def test_unknown_peaks_helper() -> None:
    print("\n== unknown_peaks is the deliberate 'nothing measured' source ==")
    result = unknown_peaks(_graph())
    check(result == ObservedPeak(None, None), f"always unknown (got {result})")
    check(result.peak_record() is None, "and never a zero record")


def test_memory_fields_resolver_reads_declarations() -> None:
    print("\n== the resolver reads class-level declarations off the registry ==")
    with_fields = type(
        "Declaring",
        (),
        {"memory_fields": ("batch_size", "rank")},
    )
    without_fields = type("Silent", (), {})
    registry = NodeRegistry(
        scan=lambda: ({"Declaring": with_fields, "Silent": without_fields}, ())
    )
    resolve = memory_fields_resolver(registry)
    check(
        resolve("Declaring") == ("batch_size", "rank"),
        "a declaring class answers its tuple",
    )
    check(resolve("Silent") is None, "a class that declares nothing answers None")
    check(resolve("NoSuchNode") is None, "an unknown class answers None")


def test_fingerprint_matches_the_pure_function() -> None:
    print("\n== the key the source computes is graph_fingerprint's key ==")
    pure = graph_fingerprint(
        _graph(),
        {"buckets": [{"height": b.height, "width": b.width, "count": b.count}
                     for b in _BUCKETS]},
        resolve_memory_fields=_resolver,
    )
    check(isinstance(pure, GraphMemoryFingerprint), "the pure function is known")
    check(not isinstance(pure, UnknownFingerprint), "and not the unknown sentinel")
    if isinstance(pure, GraphMemoryFingerprint):
        with tempfile.TemporaryDirectory() as tmp:
            store = SqlitePeakStore(Path(tmp) / "peaks.db")
            result = _source(_Shapes(), store).observed(_graph())
        check(
            result.fingerprint_key == pure.key(),
            "source and pure function agree on the key",
        )


def main() -> None:
    """Run every test in this file, listed by name.

    Listed, not discovered: a `def test_*` nothing calls is a comment
    shaped like a safety net, and `scripts/check_test_wiring.py` fails
    this file when one is defined and left out here.
    """
    tests = [
        test_no_dataset_reference_is_unknown_and_touches_nothing,
        test_known_fingerprint_reads_the_remembered_peak,
        test_unmeasured_store_is_unknown_not_zero,
        test_no_store_still_carries_the_key,
        test_shape_errors_log_and_stay_unknown,
        test_shapeless_dataset_is_unknown,
        test_missing_declared_fields_is_unknown,
        test_dataset_name_of,
        test_unknown_peaks_helper,
        test_memory_fields_resolver_reads_declarations,
        test_fingerprint_matches_the_pure_function,
    ]
    for test in tests:
        test()
    finish()


if __name__ == "__main__":
    main()
