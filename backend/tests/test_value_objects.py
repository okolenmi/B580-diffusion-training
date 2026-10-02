"""Value objects and services the application layer extracted from
duplicated rules.

One test file for the small collaborators that replaced copies of a rule
(docs 08 S-06..S-09, S-13, S-14). Each section pins behaviour that used
to live in six ``_resolve`` methods, nine publish loops, three bound
constants, nine validation guards and twelve hand-rolled
"compare-and-swap then announce" sequences -- so the deduplication
cannot be undone one call site at a time without a red test.

Run:  python backend/tests/test_value_objects.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.application.errors import InvalidQueryError  # noqa: E402
from backend.application.event_publisher import EventPublisher  # noqa: E402
from backend.application.limits import (  # noqa: E402
    DEFAULT_EXECUTION_PAGE_SIZE,
    DEFAULT_DATASET_ITEM_PAGE_SIZE,
    MAX_GRAPH_DESCRIPTION,
    MAX_PAGE_SIZE,
)
from backend.application.project_paths import ProjectPaths  # noqa: E402
from backend.application.ports.dataset_library import (  # noqa: E402
    BulkItemChanges,
    ItemChanges,
)
from backend.application.requests import (  # noqa: E402
    ITEM_TYPES,
    TEXT_MODES,
    AssetRequest,
    ItemChangesRequest,
    ItemSelection,
)
from backend.application.lifecycle_writer import LifecycleWriter  # noqa: E402
from backend.application.ports.dataset_tasks import TaskKind, TaskStatus  # noqa: E402
from backend.application.ports.graph_runtime import IssueSeverity  # noqa: E402
from backend.domain.events import DomainEvent, GraphExecutionStarted  # noqa: E402
from backend.tests.support import check, finish  # noqa: E402


class _Bus:
    def __init__(self) -> None:
        self.published: list[DomainEvent] = []

    def publish(self, event: DomainEvent) -> None:
        self.published.append(event)


class _Aggregate:
    """Anything with an event buffer -- the shape the publisher needs."""

    def __init__(self, *events: DomainEvent) -> None:
        self._events = list(events)

    def collect_events(self) -> list[DomainEvent]:
        drained, self._events = self._events, []
        return drained


def _expect_error(fn, exc_type, fragment: str, label: str) -> None:
    try:
        fn()
    except exc_type as exc:
        check(fragment in str(exc), f"{label} (got {exc})")
    else:
        check(False, f"{label} -- nothing raised")


# ---------------------------------------------------------------------------
# S-06: ProjectPaths -- "a path the client named"
# ---------------------------------------------------------------------------


def test_project_paths() -> None:
    print("\n== ProjectPaths: one definition of 'relative means the root' ==")
    root = Path(tempfile.mkdtemp(prefix="paths-test-"))
    paths = ProjectPaths(root=root)

    check(
        paths.resolve("cfg/a.toml") == root / "cfg/a.toml",
        "a relative path hangs off the project root",
    )
    check(
        paths.resolve("/etc/hosts") == Path("/etc/hosts"),
        "an absolute path is taken as sent",
    )
    check(paths.config("cfg/a.toml") == root / "cfg/a.toml", "config() resolves")
    _expect_error(
        lambda: paths.config(""),
        InvalidQueryError,
        "config path is required",
        "a missing config path names the field",
    )
    _expect_error(
        lambda: paths.require("", "image dir"),
        InvalidQueryError,
        "image dir is required",
        "require() names whichever field asked",
    )


# ---------------------------------------------------------------------------
# S-07: EventPublisher -- "drain the buffer once, in order"
# ---------------------------------------------------------------------------


def test_event_publisher() -> None:
    print("\n== EventPublisher: buffers drained once, telemetry emitted direct ==")
    bus = _Bus()
    publisher = EventPublisher(events=bus)
    first = _Aggregate(GraphExecutionStarted(execution_id=1), GraphExecutionStarted(execution_id=1))

    published = publisher.publish(first)
    check(published == 2, "publish reports how many events went out")
    check(len(bus.published) == 2, "both buffered events reached the bus")
    check(publisher.publish(first) == 0, "a drained buffer publishes nothing twice")
    check(len(bus.published) == 2, "and the bus stayed untouched on the second call")

    publisher.emit(GraphExecutionStarted(execution_id=2))
    check(len(bus.published) == 3, "emit() sends one unbuffered event")

    second = _Aggregate(GraphExecutionStarted(execution_id=3))
    total = publisher.publish_all(first, second)
    check(total == 1, "publish_all sums the buffers it drained")
    check(bus.published[-1].execution_id == 3, "in argument order")


# ---------------------------------------------------------------------------
# S-08: one limits module, imported by use cases and routes alike
# ---------------------------------------------------------------------------


def test_limits() -> None:
    print("\n== limits: the numbers API and clients agree on ==")
    check(MAX_PAGE_SIZE == 500, "one page ceiling for every list endpoint")
    check(
        DEFAULT_DATASET_ITEM_PAGE_SIZE == 500,
        "dataset item page size -- the one unbounded collection",
    )
    check(DEFAULT_EXECUTION_PAGE_SIZE == 50, "executions default page size")
    check(MAX_GRAPH_DESCRIPTION == 1000, "graph description ceiling")

    # The point of the module: the route's default *is* the use case's
    # default, not a second number that happens to match today.
    from backend.application.use_cases import list_graph_executions as lge
    from backend.application.use_cases.list_dataset_items import ListDatasetItems

    check(
        lge.MAX_LIMIT == MAX_PAGE_SIZE and lge.DEFAULT_LIMIT == DEFAULT_EXECUTION_PAGE_SIZE,
        "ListGraphExecutions enforces the shared ceiling, and its own "
        "default is the shared constant",
    )
    check(
        ListDatasetItems.MAX_LIMIT == MAX_PAGE_SIZE,
        "ListDatasetItems enforces the same one",
    )

    import backend.presentation.api.graphs as graphs_route
    from backend.application.limits import DEFAULT_EXECUTION_PAGE_SIZE as SHARED

    annotations = graphs_route.list_executions.__annotations__
    check(
        "limit" in annotations and graphs_route.list_executions.__defaults__,
        "the executions route still takes a limit",
    )
    check(
        any(getattr(d, "default", None) == SHARED
            for d in graphs_route.list_executions.__defaults__),
        "and its default is the shared constant rather than a second "
        "number that happens to match today",
    )


# ---------------------------------------------------------------------------
# S-09: request value objects -- guards that used to be copied
# ---------------------------------------------------------------------------


def test_asset_request() -> None:
    print("\n== AssetRequest: one 'asset kind is required' ==")
    check(AssetRequest.of("checkpoints").kind == "checkpoints", "kind kept")
    check(
        AssetRequest.of("lora", "sub/dir").relative_path == "sub/dir",
        "path kept when given",
    )
    _expect_error(
        lambda: AssetRequest.of(""),
        InvalidQueryError,
        "asset kind is required",
        "empty kind refused",
    )
    _expect_error(
        lambda: AssetRequest.of("lora", "", path_required=True),
        InvalidQueryError,
        "relative_path is required",
        "an operation that needs a path says so, by that name",
    )
    check(
        AssetRequest.of("lora", "").relative_path == "",
        "listing a kind needs no path",
    )


def test_item_selection() -> None:
    print("\n== ItemSelection: one 'item_ids must not be empty' ==")
    selection = ItemSelection.of([3, 1, 3])
    check(selection.as_list() == [3, 1, 3], "the ids are kept as sent (no dedup)")
    check(len(selection) == 3, "length is the id count")
    check(list(selection) == [3, 1, 3], "iterable for the adapters that want a list")
    _expect_error(
        lambda: ItemSelection.of([]),
        InvalidQueryError,
        "item_ids must not be empty",
        "an empty selection refused",
    )


def test_item_changes_request() -> None:
    print("\n== ItemChangesRequest: single and bulk share the two rules ==")
    changes = ItemChangesRequest.single(prompt="a cat")
    check(isinstance(changes, ItemChanges) and changes.prompt == "a cat", "single patch")
    bulk = ItemChangesRequest.bulk(prompt="a cat", prompt_mode="prepend")
    check(
        isinstance(bulk, BulkItemChanges) and bulk.prompt_mode == "prepend",
        "bulk patch keeps its mode",
    )

    for label, fn in (
        ("single", lambda: ItemChangesRequest.single()),
        ("bulk", lambda: ItemChangesRequest.bulk()),
    ):
        _expect_error(
            fn,
            InvalidQueryError,
            "no changes provided",
            f"{label} refuses a patch that changes nothing",
        )
    for label, fn in (
        ("single", lambda: ItemChangesRequest.single(prompt="x", type="maybe")),
        ("bulk", lambda: ItemChangesRequest.bulk(prompt="x", type="maybe")),
    ):
        _expect_error(
            fn,
            InvalidQueryError,
            "unknown type",
            f"{label} refuses an unknown verdict",
        )
    for mode_field, fn in (
        ("prompt_mode", lambda: ItemChangesRequest.bulk(prompt="x", prompt_mode="upend")),
        ("neg_prompt_mode", lambda: ItemChangesRequest.bulk(prompt="x", neg_prompt_mode="upend")),
    ):
        _expect_error(
            fn, InvalidQueryError, f"unknown {mode_field}", f"{mode_field} refused"
        )

    check(ITEM_TYPES == ("good", "bad"), "verdict vocabulary")
    check(TEXT_MODES == ("set", "prepend", "append"), "text-mode vocabulary")
    # Order matters: an *unknown* verdict is refused even when nothing
    # else was sent, so the vocabulary check cannot be masked by the
    # "no changes" rule running first.
    _expect_error(
        lambda: ItemChangesRequest.single(type="maybe"),
        InvalidQueryError,
        "unknown type",
        "the verdict is checked before the no-changes rule",
    )
    verdict_only = ItemChangesRequest.single(type="good")
    check(verdict_only.type == "good", "a verdict on its own is a real change")


# ---------------------------------------------------------------------------
# S-13b: LifecycleWriter -- compare-and-swap, then announce, winner only
# ---------------------------------------------------------------------------


class _Repo:
    """A repository that answers the CAS however the test says."""

    def __init__(self, *, wins: bool = True) -> None:
        self.wins = wins
        self.calls: list[str] = []

    def add(self, aggregate):
        self.calls.append("add")
        return aggregate

    def update_if_status(self, aggregate, *, expected) -> bool:
        self.calls.append(f"cas:{expected}")
        return self.wins


def test_lifecycle_writer() -> None:
    print("\n== LifecycleWriter: CAS first, announce only if you won ==")
    bus = _Bus()
    writer = LifecycleWriter(repository=_Repo(wins=True), events=EventPublisher(events=bus))

    first = _Aggregate(GraphExecutionStarted(execution_id=1))
    check(writer.commit(first, expected="running") is True, "a won CAS reports success")
    check(len(bus.published) == 1, "and the buffered events are announced")

    loser = LifecycleWriter(
        repository=_Repo(wins=False), events=EventPublisher(events=bus)
    )
    second = _Aggregate(GraphExecutionStarted(execution_id=2))
    check(
        loser.commit(second, expected="running") is False,
        "a lost CAS reports failure",
    )
    check(len(bus.published) == 1, "and announces nothing -- the row is not ours")

    # The order is the whole point: nothing is announced before the row
    # is actually written, so the stream can never show a state the
    # database does not hold.
    bus2 = _Bus()
    repo = _Repo(wins=True)
    writer2 = LifecycleWriter(repository=repo, events=EventPublisher(events=bus2))
    third = _Aggregate(GraphExecutionStarted(execution_id=3))
    writer2.commit(third, expected="running")
    check(
        repo.calls == ["cas:running"] and len(bus2.published) == 1,
        "the row was written before it was announced",
    )

    # ``prior`` exists for the start-that-failed repair: the launch
    # entity still buffers RunCreated, and the stream must read
    # created -> failed rather than failed -> created.
    bus3 = _Bus()
    writer3 = LifecycleWriter(repository=_Repo(), events=EventPublisher(events=bus3))
    created = _Aggregate(GraphExecutionStarted(execution_id=4))
    failed = _Aggregate(GraphExecutionStarted(execution_id=4))
    writer3.commit(failed, expected="created", prior=(created,))
    check(
        [e.execution_id for e in bus3.published] == [4, 4],
        "prior buffers are announced before the aggregate's own",
    )

    stored = writer3.insert(_Aggregate(GraphExecutionStarted(execution_id=5)))
    check(
        isinstance(stored, _Aggregate) and len(bus3.published) == 3,
        "insert persists and announces (nothing to race against)",
    )


# ---------------------------------------------------------------------------


    print("\n== TaskStatus: active and terminal derived (S-24) ==")
    check(TaskStatus.PENDING.is_active and TaskStatus.RUNNING.is_active,
          "pending/running are active")
    check(
        all(
            not s.is_active and s.is_terminal
            for s in (TaskStatus.FINISHED, TaskStatus.FAILED, TaskStatus.KILLED)
        ),
        "finished/failed/killed are terminal",
    )
    check(
        all(not s.is_terminal for s in TaskStatus if s.is_active),
        "an active status is never terminal",
    )
    check(len(list(TaskStatus)) == 5, "the vocabulary is closed")
    check(
        [k.value for k in TaskKind] == ["ingest_lora", "generate_teacher"],
        "two task kinds",
    )


def test_task_status() -> None:
    print("\n== TaskStatus: active and terminal derived (S-24) ==")
    check(TaskStatus.PENDING.is_active and TaskStatus.RUNNING.is_active,
          "pending/running are active")
    check(
        all(
            not s.is_active and s.is_terminal
            for s in (TaskStatus.FINISHED, TaskStatus.FAILED, TaskStatus.KILLED)
        ),
        "finished/failed/killed are terminal",
    )
    check(
        all(not s.is_terminal for s in TaskStatus if s.is_active),
        "an active status is never terminal",
    )
    check(len(list(TaskStatus)) == 5, "the vocabulary is closed")
    check(
        [k.value for k in TaskKind] == ["ingest_lora", "generate_teacher"],
        "two task kinds",
    )


def test_issue_severity() -> None:
    print("\n== IssueSeverity: what blocks a run ==")
    check(IssueSeverity.ERROR.blocks, "an error blocks")
    check(not IssueSeverity.WARNING.blocks, "a warning does not")
    check(IssueSeverity.ERROR == "error", "str-valued for the wire")
    check(
        IssueSeverity("error") is IssueSeverity.ERROR,
        "the stored word resolves to the member",
    )


def main() -> None:
    test_project_paths()
    test_event_publisher()
    test_limits()
    test_asset_request()
    test_item_selection()
    test_item_changes_request()
    test_lifecycle_writer()
    test_task_status()
    test_issue_severity()
    finish()


if __name__ == "__main__":
    main()