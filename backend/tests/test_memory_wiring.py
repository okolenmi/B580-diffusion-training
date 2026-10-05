"""MEM-03 wiring: admission in the start paths, release on every path.

The ledger's own concurrency and breakdown are pinned in
``test_memory_ledger.py``; what is pinned *here* is the wiring around it,
at the seams where a claim can be taken and never given back:

* a refused start holds nothing and writes no row, and the refusal
  carries the whole breakdown naming the holders (task rules 6 + "a
  refused run holds nothing");
* a crash between the claim and the child leaves a row whose claim a
  restart rebuilds, and reconcile then releases -- the convergence
  property that makes "reserve before spawn" safe (rule: cross-process
  properties are tested with real processes, so the last test races two
  *actual* HTTP client processes for the one device);
* every release path is idempotent, and ``None``-ledger containers
  refuse explicitly instead of admitting unchecked (rule 2);
* a dataset task and a graph run share the one ledger, in both
  directions: an unknown-demand task is exclusive, a stated graph run
  fits beside nothing;
* the probe is admitted like any other child and degrades to "not
  checked" -- never a claim recorded unchecked -- when refused;
* ``/health`` reports the live snapshot, or ``null`` while no ledger
  exists (never a fabricated zero).

Run directly: python backend/tests/test_memory_wiring.py
"""

from __future__ import annotations

import json
import multiprocessing
import socket
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import uvicorn

from backend.application.dataset_task_sweeper import DatasetTaskSweeper
from backend.application.dto import StartDatasetTaskCommand
from backend.application.errors import MemoryUnavailableError
from backend.application.event_publisher import EventPublisher
from backend.application.graph_peak_source import (
    GraphPeakSource,
    ObservedPeak,
    unknown_peaks,
)
from backend.application.graph_supervisor import GraphExecutionSupervisor
from backend.application.lifecycle_writer import ExecutionLifecycleWriter
from backend.application.memory_admission import (
    LedgerProvider,
    graph_owner,
    release,
    task_owner,
)
from backend.application.memory_ledger import (
    DEFAULT_FOREIGN_RESERVE_MB,
    DEFAULT_PROCESS_OVERHEAD_MB,
    Grant,
    MemoryLedger,
)
from backend.application.ports.dataset_tasks import TaskKind
from backend.application.ports.environment import DeviceReport
from backend.application.ports.execution_launcher import ExecutionLauncher
from backend.application.ports.graph_task_gateway import (
    GraphTaskGateway,
    GraphTaskLaunch,
)
from backend.application.ports.peak_store import PeakStore
from backend.application.use_cases import (
    ReconcileDatasetTasks,
    ReconcileGraphExecutions,
    StartDatasetTask,
    StartGraphExecution,
)
from backend.domain.entities.graph_execution import GraphExecution
from backend.domain.graph import GraphDefinition, GraphEdgeSpec, GraphNodeSpec
from backend.domain.memory_settings import MemorySettings, effective_memory
from backend.infrastructure.dataset_library import SqliteDatasetLibrary
from backend.infrastructure.dataset_tasks import SqliteDatasetTasks
from backend.infrastructure.graph.discovery import NodeRegistry, memory_fields_resolver
from backend.infrastructure.graph.runtime import ReflectedGraphRuntime
from backend.infrastructure.graph_event_stream import (
    ExecutionEventTail,
    ExecutionEventWriter,
)
from backend.infrastructure.memory_peak_store import SqlitePeakStore
from backend.infrastructure.persistence.graph_execution_repository import (
    SqliteGraphExecutionRepository,
)
from backend.infrastructure.persistence.sqlite import SqliteDatabase
from backend.infrastructure.workspace import WorkspaceLayout
from backend.presentation.app import create_app
from backend.tests.support import (
    FakeClock,
    FakeDatasetTaskGateway,
    FakeDeviceProbe,
    RecordingEventBus,
    asgi_request,
    build_services,
    check,
    finish,
    fixture_graph_registry,
    make_v2_dataset,
    wait_until,
)
from nodes.core import Node, Port

NOW = datetime(2026, 3, 1, 9, 0, 0, tzinfo=UTC)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def valid_graph() -> GraphDefinition:
    """The palette graph every other suite runs (Scale -> Sum)."""
    return GraphDefinition(
        nodes=(
            GraphNodeSpec(
                id="v", class_name="ScaleNode",
                params={"value": 3.0, "factor": 5.0},
            ),
            GraphNodeSpec(id="s", class_name="SumNode", params={"b": 1.0}),
        ),
        edges=(
            GraphEdgeSpec(
                from_node="v", from_port="scaled",
                to_node="s", to_port="a",
            ),
        ),
    )


def slow_graph(seconds: float) -> GraphDefinition:
    """A run that stays alive long enough for a second start to race it."""
    return GraphDefinition(
        nodes=(
            GraphNodeSpec(
                id="w", class_name="SlowNode", params={"seconds": seconds},
            ),
        ),
        edges=(),
    )


class _NoChildren(ExecutionLauncher):
    """Adoptable nothing: for tests about rows that must not spawn."""

    def launch(self, execution_id, graph) -> None:
        raise AssertionError("this test never launches a run")

    def cancel(self, execution_id) -> None:
        raise AssertionError("this test never cancels a run")

    def adopt(self, execution_id):
        return None

    def has_running_child(self, execution_id) -> bool:
        return False

    def recorded_outcome(self, execution_id):
        return None


class _BoomLauncher(_NoChildren):
    """The spawn itself dies: the claim must not survive it."""

    def launch(self, execution_id, graph) -> None:
        raise RuntimeError("spawn died")


class _AdmittedOnly(_NoChildren):
    """Accepts the start and never runs it: admission-only tests.

    What such a test asserts is what admission *read* (the remembered
    peak, the fingerprint key) and what it *granted* -- the row keeps
    both without a child ever existing.
    """

    def launch(self, execution_id, graph) -> None:
        return None


def _provider(probe, graph_executions, dataset_tasks) -> LedgerProvider:
    return LedgerProvider(
        probe=probe,
        graph_executions=graph_executions,
        dataset_tasks=dataset_tasks,
        foreign_reserve_mb=DEFAULT_FOREIGN_RESERVE_MB,
        process_overhead_mb=DEFAULT_PROCESS_OVERHEAD_MB,
    )


def _writer(repo, clock) -> ExecutionLifecycleWriter:
    return ExecutionLifecycleWriter(
        clock=clock,
        repository=repo,
        events=EventPublisher(events=RecordingEventBus()),
    )


def _graph_start(repo, ledger_source, *, launcher=None, clock=None,
                 runtime=None, peak_source=None):
    clock = clock if clock is not None else FakeClock()
    return StartGraphExecution(
        executions=repo,
        writer=_writer(repo, clock),
        runtime=(runtime if runtime is not None
                 else ReflectedGraphRuntime(fixture_graph_registry())),
        launcher=launcher if launcher is not None else _NoChildren(),
        clock=clock,
        memory_ledger=ledger_source,
        # ``unknown_peaks`` keeps the default exactly what this suite
        # claimed before the read half existed: no fingerprint inputs,
        # every demand unknown, never a zero peak (MEM-04 #2).
        peak_source=peak_source if peak_source is not None else unknown_peaks,
    )


def _task_repo(root: Path, clock: FakeClock | None = None) -> SqliteDatasetTasks:
    db = SqliteDatabase(root / "tasks.db")
    db.initialize()
    return SqliteDatasetTasks(db, clock if clock is not None else FakeClock())


def _graph_repo(root: Path) -> SqliteGraphExecutionRepository:
    db = SqliteDatabase(root / "graphs.db")
    db.initialize()
    return SqliteGraphExecutionRepository(db)


def _own_checkpoints(app, root: Path) -> Path:
    """Point the wired layout's checkpoints at a directory this test owns.

    The layout's default resolves through ComfyUI's own layout (or
    raises), so a task test that did not take possession of the knob
    would be reading the machine's real checkpoints directory.
    """
    ckpt = root / "models"
    ckpt.mkdir(parents=True, exist_ok=True)
    status, _, body = asgi_request(
        app, "/api/v1/settings",
        method="POST", json_body={"checkpoints_dir": str(ckpt)},
    )
    check(
        status == 200 and isinstance(body, dict)
        and body.get("stored", {}).get("checkpoints_dir") == str(ckpt),
        f"fixture: checkpoints dir owned by this test ({status})",
    )
    (ckpt / "m.safetensors").write_bytes(b"st")
    return ckpt


def _image_dir(root: Path) -> Path:
    imgs = root / "imgs"
    imgs.mkdir(parents=True, exist_ok=True)
    for i in range(2):
        (imgs / f"{i}.png").write_bytes(b"png")
    return imgs


def _task_cmd(dataset: str, imgs: Path) -> StartDatasetTaskCommand:
    return StartDatasetTaskCommand(
        dataset=dataset, image_dir=str(imgs), model="m.safetensors"
    )


def _refusal(fn, label: str) -> MemoryUnavailableError | None:
    """Run ``fn``, expecting a memory refusal; record and return it."""
    try:
        fn()
    except MemoryUnavailableError as exc:
        check(True, label)
        return exc
    check(False, f"{label} (got no refusal)")
    return None


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def test_refused_run_holds_nothing_and_names_the_holder() -> None:
    print("-- refused start: holds nothing, names the holder --")
    services = build_services()
    ledger = services.memory_ledger()
    check(ledger is not None, "fixture: the provider builds a ledger")
    assert ledger is not None

    held_before = ledger.reserve(
        "fixture:blocker", ledger.capacity_mb, exploratory=False
    )
    check(isinstance(held_before, Grant), "fixture: the card is fully held")
    held0 = ledger.held_mb()

    exc = _refusal(
        lambda: services.graphs.start_execution.execute(valid_graph()),
        "a run that cannot fit is refused 409",
    )
    if exc is not None:
        check(
            exc.code == "memory_unavailable" and exc.status_code == 409,
            f"409 memory_unavailable (got {exc.status_code} {exc.code})",
        )
        details = exc.details or {}
        check(
            "fixture:blocker" in str(exc),
            f"the refusal's message names the holder (got {str(exc)!r})",
        )
        check(
            details.get("holders", {}).get("fixture:blocker")
            == ledger.capacity_mb,
            f"the breakdown lists the holder and its size ({details})",
        )
        check(
            {"capacity_mb", "foreign_reserve_mb", "free_mb", "requested_mb"}
            <= set(details),
            f"the breakdown carries the whole arithmetic ({sorted(details)})",
        )
    check(ledger.held_mb() == held0, "a refused run holds nothing")
    check(
        services.graphs.list_executions.execute(limit=50).count == 0,
        "a refused run writes no row",
    )
    # Release the fixture claim so this services object ends clean.
    ledger.release("fixture:blocker")


def test_crash_between_reserve_and_spawn_releases_then_converges() -> None:
    print("-- crash between reserve and spawn: held 0, restart + reconcile --")
    root = Path(tempfile.mkdtemp(prefix="backend-mem-crash-"))
    repo = _graph_repo(root)
    tasks = _task_repo(root)
    clock = FakeClock()
    provider = _provider(FakeDeviceProbe(), repo, tasks)
    ledger = provider()
    check(ledger is not None, "fixture: the provider builds a ledger")
    assert ledger is not None

    start = _graph_start(repo, provider, launcher=_BoomLauncher(), clock=clock)
    try:
        start.execute(valid_graph())
        check(False, "a launcher that dies must surface to the caller")
    except RuntimeError:
        check(True, "the spawn failure surfaces to the caller")
    check(
        ledger.held_mb() == 0.0,
        f"a crash between the claim and the child holds nothing "
        f"(held {ledger.held_mb()})",
    )
    unfinished = repo.list_unfinished()
    check(
        len(unfinished) == 1 and unfinished[0].reserved_mb,
        "the row it wrote still carries its claim for the next start",
    )

    # A restart: a fresh provider rebuilds the same held total from the
    # rows, and reconcile -- which finds no child -- releases it. Either
    # build/reconcile order converges on held 0.
    restarted = _provider(FakeDeviceProbe(), repo, tasks)
    ledger2 = restarted()
    check(ledger2 is not None, "fixture: the restarted provider builds")
    assert ledger2 is not None
    check(
        ledger2.held_mb() == unfinished[0].reserved_mb,
        f"a restart reproduces the held total from the row "
        f"({ledger2.held_mb()} vs {unfinished[0].reserved_mb})",
    )
    sweep = ReconcileGraphExecutions(
        executions=repo,
        writer=_writer(repo, clock),
        launcher=_BoomLauncher(),
        clock=clock,
        memory_ledger=restarted,
    )
    result = sweep.execute()
    check(result.cleaned == 1, f"reconcile fails the childless row ({result})")
    check(
        ledger2.held_mb() == 0.0,
        "reconcile hands the rebuilt claim back",
    )


def test_restart_reproduces_the_held_total_from_both_kinds_of_row() -> None:
    print("-- rebuild: held total from graph and task rows, released by both --")
    root = Path(tempfile.mkdtemp(prefix="backend-mem-rebuild-"))
    repo = _graph_repo(root)
    clock = FakeClock()
    tasks = _task_repo(root, clock)

    execution = repo.add(
        GraphExecution.create(
            graph=valid_graph(), created_at=NOW, reserved_mb=7000.0
        )
    )
    task = tasks.add(
        dataset="d", kind=TaskKind.INGEST_LORA, total=1, params={},
        reserved_mb=3000.0,
    )

    provider = _provider(FakeDeviceProbe(), repo, tasks)
    ledger = provider()
    check(ledger is not None, "fixture: the provider builds a ledger")
    assert ledger is not None
    check(
        ledger.held_mb() == 10000.0,
        f"restart reproduces the total from both kinds of row "
        f"(held {ledger.held_mb()})",
    )
    check(
        set(ledger.snapshot()["holders"])
        == {graph_owner(execution.require_id()), task_owner(task.id)},
        f"owners are the row-derived ones ({ledger.snapshot()['holders']})",
    )

    # Graph side: queued with no child -> reconcile cleans -> release.
    swept = ReconcileGraphExecutions(
        executions=repo,
        writer=_writer(repo, clock),
        launcher=_NoChildren(),
        clock=clock,
        memory_ledger=provider,
    ).execute()
    check(swept.cleaned == 1, f"the graph row is reconciled ({swept})")
    check(
        ledger.held_mb() == 3000.0,
        "the graph claim is back; the task claim still stands",
    )

    # Task side: pending with no pid, past the stuck threshold -> the
    # sweeper fails the row -> release.
    clock.advance(120)
    reconciled = ReconcileDatasetTasks(
        sweeper=DatasetTaskSweeper(
            tasks=tasks,
            gateway=FakeDatasetTaskGateway(),
            clock=clock,
            memory_ledger=provider,
        )
    ).execute()
    check(reconciled.cleaned == 1, f"the task row is swept ({reconciled})")
    check(
        ledger.held_mb() == 0.0,
        "both release paths hand the rebuilt claims back",
    )


def test_release_is_idempotent_on_every_path() -> None:
    print("-- release: idempotent, and safe with no source at all --")
    ledger = MemoryLedger(total_mb=12216.0)
    grant = ledger.reserve("graph:7", 4000.0, exploratory=False)
    check(isinstance(grant, Grant), "fixture: a claim is taken")
    source = lambda: ledger  # noqa: E731 -- a LedgerSource in one line
    release(source, "graph:7")
    release(source, "graph:7")  # watcher racing a stop, or both paths
    release(source, "graph:7")
    check(
        ledger.held_mb() == 0.0,
        f"three releases hand back exactly one claim (held "
        f"{ledger.held_mb()})",
    )
    release(source, "graph:never-existed")
    check(ledger.held_mb() == 0.0, "releasing an owner that never existed is a no-op")
    release(None, "graph:7")
    check(ledger.held_mb() == 0.0, "a container with no source releases nothing")


def test_a_task_blocks_a_graph_and_a_graph_blocks_a_task() -> None:
    print("-- cross-kind: task blocks graph, graph blocks task --")
    root = Path(tempfile.mkdtemp(prefix="backend-mem-cross-"))
    services = build_services(project_root=root)
    app = create_app(services)
    _own_checkpoints(app, root)
    imgs = _image_dir(root)
    make_v2_dataset(root, "dev")
    ledger = services.memory_ledger()
    check(ledger is not None, "fixture: the provider builds a ledger")
    assert ledger is not None
    cmd = _task_cmd("dev", imgs)

    # A task with no measured default is an unknown demand, and an
    # unknown demand is an exclusive claim: it takes the whole card.
    task = services.datasets.start_task.execute(cmd)
    check(
        task.status == "running",
        f"the task starts (status {task.status})",
    )
    check(
        ledger.held_mb() == ledger.capacity_mb,
        f"an unknown-demand task claims the whole usable card "
        f"(held {ledger.held_mb()})",
    )
    exc = _refusal(
        lambda: services.graphs.start_execution.execute(
            valid_graph(), memory_overrides={"vram_max_mb": 8000.0}
        ),
        "a graph cannot start beside an exclusive task",
    )
    if exc is not None:
        check(
            task_owner(task.id) in str(exc),
            f"the refusal names the task (got {str(exc)!r})",
        )
    check(
        ledger.held_mb() == ledger.capacity_mb,
        "the refused graph added nothing to what the task holds",
    )

    # Stop the task: its exit path is what hands the claim back.
    services.datasets.stop_task.execute(task.id)
    check(
        wait_until(lambda: ledger.held_mb() == 0.0, timeout=5.0),
        "the stopped task's claim is back (exit path)",
    )

    # The other direction: a stated graph run holds allocator + overhead,
    # and a task -- which would claim exclusively -- cannot start beside it.
    run = services.graphs.start_execution.execute(
        slow_graph(0.8), memory_overrides={"vram_max_mb": 8000.0}
    )
    check(
        ledger.held_mb() == 8600.0,
        f"a stated claim is allocator MB + process overhead "
        f"({ledger.held_mb()} = 8000 + 600)",
    )
    exc = _refusal(
        lambda: services.datasets.start_task.execute(cmd),
        "a task cannot start beside a graph run",
    )
    if exc is not None:
        check(
            graph_owner(run.execution_id) in str(exc),
            f"the refusal names the graph (got {str(exc)!r})",
        )
    check(
        wait_until(
            lambda: ledger.held_mb() == 0.0,
            timeout=5.0,
        ),
        "the finished run's claim is back (watcher path)",
    )
    task2 = services.datasets.start_task.execute(cmd)
    check(
        task2.status == "running",
        "with the card free the task admits again",
    )
    services.datasets.stop_task.execute(task2.id)
    check(ledger.held_mb() == 0.0, "and its claim is back on exit")


def test_parallel_task_starts_admit_exactly_one() -> None:
    print("-- four datasets, four parallel starts, one admitted --")
    root = Path(tempfile.mkdtemp(prefix="backend-mem-parallel-"))
    services = build_services(project_root=root)
    app = create_app(services)
    _own_checkpoints(app, root)
    imgs = _image_dir(root)
    for i in range(4):
        make_v2_dataset(root, f"p{i}")
    ledger = services.memory_ledger()
    check(ledger is not None, "fixture: the provider builds a ledger")
    assert ledger is not None

    outcomes: list = []
    lock = threading.Lock()

    def attempt(i: int) -> None:
        try:
            started = services.datasets.start_task.execute(
                _task_cmd(f"p{i}", imgs)
            )
            result = ("ok", started)
        except MemoryUnavailableError as exc:
            result = ("refused", exc)
        except Exception as exc:  # noqa: BLE001 -- recorded, never swallowed
            result = ("error", exc)
        with lock:
            outcomes.append((i, result))

    threads = [
        threading.Thread(target=attempt, args=(i,), daemon=True)
        for i in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15.0)
    check(
        not any(thread.is_alive() for thread in threads),
        "every parallel start answered",
    )
    check(len(outcomes) == 4, f"four outcomes recorded ({len(outcomes)})")

    ok = [(i, r) for i, r in outcomes if r[0] == "ok"]
    refused = [(i, r) for i, r in outcomes if r[0] == "refused"]
    errored = [(i, r) for i, r in outcomes if r[0] == "error"]
    check(
        not errored,
        f"no start failed with an unexpected error "
        f"({[str(r) for _, r in errored]})",
    )
    check(
        len(ok) == 1 and len(refused) == 3,
        f"exactly one of four parallel starts admits "
        f"({len(ok)} admitted, {len(refused)} refused)",
    )
    if len(ok) == 1 and len(refused) == 3:
        winner = ok[0][1][1]
        for _, (_, exc) in refused:
            check(
                task_owner(winner.id) in str(exc),
                f"each refusal names the winner (got {str(exc)!r})",
            )
        check(
            ledger.held_mb() == ledger.capacity_mb,
            "the winner holds exactly the one claim",
        )
        services.datasets.stop_task.execute(winner.id)
        check(
            wait_until(lambda: ledger.held_mb() == 0.0, timeout=5.0),
            "stopping the winner hands the card back",
        )


def test_without_a_ledger_every_start_refuses_explicitly() -> None:
    print("-- no ledger: device-total-unknown, no row, never a zero claim --")
    root = Path(tempfile.mkdtemp(prefix="backend-mem-noleger-"))

    # Graph path.
    repo = _graph_repo(root)
    start = _graph_start(repo, lambda: None)
    exc = _refusal(
        lambda: start.execute(valid_graph()),
        "a graph start with no ledger is refused",
    )
    if exc is not None:
        details = exc.details or {}
        check(
            details.get("reason") == "device_total_unknown",
            f"the refusal says the device total is unknown ({details})",
        )
        check(
            "device total is unknown" in str(exc),
            f"the message says it in words (got {str(exc)!r})",
        )
        check(
            details.get("holders") == {},
            f"a source with no rows wired to it names no holders "
            f"({details})",
        )
    check(
        len(repo.list_unfinished()) == 0,
        "an unknown-device refusal writes no graph row",
    )

    # Task path: assembled by hand, because the wired container's
    # provider is exactly what this case is the absence of.
    layout = WorkspaceLayout(root)
    library = SqliteDatasetLibrary(layout)
    tasks = _task_repo(root)
    ckpt = root / "ckpt"
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "m.safetensors").write_bytes(b"st")
    imgs = _image_dir(root)
    make_v2_dataset(root, "nol")
    start_task = StartDatasetTask(
        library=library,
        tasks=tasks,
        gateway=FakeDatasetTaskGateway(),
        checkpoints_dir=lambda: ckpt,
        memory_ledger=lambda: None,
    )
    exc = _refusal(
        lambda: start_task.execute(_task_cmd("nol", imgs)),
        "a task start with no ledger is refused",
    )
    if exc is not None:
        check(
            (exc.details or {}).get("reason") == "device_total_unknown",
            f"the refusal says the device total is unknown ({exc.details})",
        )
    check(
        len(tasks.list_unfinished()) == 0,
        "an unknown-device refusal writes no task row",
    )


def test_health_reports_the_live_ledger_snapshot() -> None:
    print("-- /health: live snapshot, or null while no ledger exists --")
    services = build_services()
    app = create_app(services)
    status, _, body = asgi_request(app, "/api/v1/health")
    memory = body.get("memory") if isinstance(body, dict) else None
    check(
        status == 200 and isinstance(memory, dict),
        f"health carries a memory snapshot ({status})",
    )
    if isinstance(memory, dict):
        check(
            memory.get("capacity_mb") == 11192.0
            and memory.get("held_mb") == 0.0
            and memory.get("free_mb") == 11192.0
            and memory.get("holders") == {},
            f"empty-card arithmetic: 12216 - 1024 = 11192 free "
            f"(got {memory})",
        )

    # A live run shows up as a holder with its exact size.
    ledger = services.memory_ledger()
    assert ledger is not None
    run = services.graphs.start_execution.execute(
        slow_graph(0.6), memory_overrides={"vram_max_mb": 8000.0}
    )
    status, _, body = asgi_request(app, "/api/v1/health")
    memory = body.get("memory") if isinstance(body, dict) else None
    check(
        status == 200
        and isinstance(memory, dict)
        and memory.get("held_mb") == 8600.0
        and memory.get("holders", {}).get(graph_owner(run.execution_id), {}).get(
            "mb"
        )
        == 8600.0,
        f"a running hold is visible with its owner and size (got {memory})",
    )
    check(
        wait_until(lambda: ledger.held_mb() == 0.0, timeout=5.0),
        "the run finishes and the snapshot empties again",
    )
    status, _, body = asgi_request(app, "/api/v1/health")
    memory = body.get("memory") if isinstance(body, dict) else None
    check(
        isinstance(memory, dict) and memory.get("held_mb") == 0.0,
        f"after the run the card reads free (got {memory})",
    )

    # No card: an explicit unknown -- null total, never zero -- with
    # empty holders, because this root has no rows to name (MEM-03H-03).
    no_card = build_services(
        project_root=Path(tempfile.mkdtemp(prefix="backend-mem-nocard-")),
        device_probe=FakeDeviceProbe(
            DeviceReport(present=False, backend="xpu", name=None)
        ),
    )
    status, _, body = asgi_request(create_app(no_card), "/api/v1/health")
    memory = body.get("memory") if isinstance(body, dict) else None
    check(
        status == 200
        and isinstance(memory, dict)
        and memory.get("total_mb") is None
        and memory.get("holders") == {},
        f"an unknown device total reports a null total and no holders "
        f"it could name, never a zero ({body})",
    )


def test_the_probe_is_admitted_and_degrades_when_refused() -> None:
    print("-- the probe: admitted around report(), not checked when refused --")
    # Success path: the probe's claim is taken and released around the
    # report, so readiness costs exactly nothing.
    probe = FakeDeviceProbe()
    services = build_services(device_probe=probe)
    ledger = services.memory_ledger()
    check(ledger is not None, "fixture: the provider builds a ledger")
    assert ledger is not None
    before = ledger.held_mb()
    report = services.installer.check.execute()
    check(
        report.device_checked is True and report.device_present,
        "an admitted probe is asked and answers",
    )
    check(
        report.device_total_memory_mb == 12216.0,
        f"the readiness report carries the total ({report.device_total_memory_mb})",
    )
    check(
        ledger.held_mb() == before,
        f"the probe's claim is released when it returns "
        f"(held {ledger.held_mb()} before {before})",
    )
    check(
        probe.calls == 2,
        f"one report to size the ledger, one for readiness ({probe.calls})",
    )

    # Refusal path: the probe is NOT asked, and the degraded report
    # carries the breakdown rather than asserting anything about the
    # device (rule 2: no unchecked claim; rule 6: the refusal carries
    # the whole detail).
    probe2 = FakeDeviceProbe()
    services2 = build_services(
        project_root=Path(tempfile.mkdtemp(prefix="backend-mem-probe-")),
        device_probe=probe2,
    )
    ledger2 = services2.memory_ledger()
    assert ledger2 is not None
    blocker = ledger2.reserve(
        "probe-blocker", ledger2.capacity_mb, exploratory=False
    )
    check(isinstance(blocker, Grant), "fixture: the card is fully held")
    degraded = services2.installer.check.execute()
    check(
        degraded.device_checked is False and degraded.device_present is False,
        "a refused probe is not asked (device_checked False)",
    )
    check(
        degraded.device_reason is None,
        "a memory refusal is not stated as a fact about the device",
    )
    detail = degraded.device_detail or ""
    check(
        "probe not run" in detail and "probe-blocker" in detail,
        f"the degraded report carries the breakdown ({detail!r})",
    )
    check(
        probe2.calls == 1,
        f"the probe itself ran only for sizing, not while refused "
        f"({probe2.calls})",
    )
    check(
        ledger2.held_mb() == ledger2.capacity_mb,
        "the refused probe held nothing",
    )


# -- the cross-process property, with real processes ------------------------

def _http_racer(url: str, payload: dict, who: str, ready, go, results) -> None:
    """Child process: announce readiness, wait for the gun, POST, report.

    Both racers are *actual* processes talking to an *actual* HTTP
    server, so the admission that decides between them is the same
    cross-process authority production has -- not two threads sharing
    one interpreter's locks.
    """
    ready.put(who)
    if not go.wait(timeout=30.0):
        results.put((who, -1, {"error": "go signal never arrived"}))
        return
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30.0) as response:
            raw = response.read()
            status = response.status
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        status = exc.code
    except Exception as exc:  # noqa: BLE001 -- the parent asserts on it
        results.put((who, -1, {"error": repr(exc)}))
        return
    try:
        body = json.loads(raw or b"{}")
    except ValueError:
        body = {"raw": raw.decode("utf-8", "replace")}
    results.put((who, status, body))


def _drain(queue, count: int, label: str) -> list:
    """Read up to ``count`` items, reporting (never hiding) a timeout."""
    items = []
    for _ in range(count):
        try:
            items.append(queue.get(timeout=60.0))
        except Exception as exc:  # noqa: BLE001 -- recorded by the caller
            print(f"  {label} queue: {exc!r} (got {items})")
            break
    return items


def _start_server(app):
    """A real socket and a real uvicorn thread; None when it never rose."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.listen(64)
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(
        target=server.run, kwargs={"sockets": [sock]}, daemon=True
    )
    thread.start()
    if not wait_until(lambda: server.started, timeout=20.0, interval=0.02):
        server.should_exit = True
        thread.join(timeout=5.0)
        return None
    return server, thread, port


def _race_over_http(port: int, graph_payload: dict, task_payload: dict):
    """Spawn both children, fire the gun, return ``{who: (who, st, body)}``.

    ``None`` when the race could not be judged (a child never showed).
    Either way, no child is left running behind us.
    """
    ctx = multiprocessing.get_context("spawn")
    ready, go, results = ctx.Queue(), ctx.Event(), ctx.Queue()
    racers = [
        ctx.Process(
            target=_http_racer,
            args=(
                f"http://127.0.0.1:{port}/api/v1/graphs/run",
                graph_payload, "graph", ready, go, results,
            ),
            daemon=True,
        ),
        ctx.Process(
            target=_http_racer,
            args=(
                f"http://127.0.0.1:{port}/api/v1/datasets/race/tasks",
                task_payload, "task", ready, go, results,
            ),
            daemon=True,
        ),
    ]
    try:
        for process in racers:
            process.start()
        announced = _drain(ready, 2, "ready")
        check(
            sorted(announced) == ["graph", "task"],
            f"both children reached their start line ({announced})",
        )
        go.set()
        collected = _drain(results, 2, "results")
        for process in racers:
            process.join(timeout=10.0)
        check(
            len(collected) == 2
            and all(process.exitcode == 0 for process in racers),
            f"both children exited cleanly "
            f"({[process.exitcode for process in racers]}, "
            f"{len(collected)} answers)",
        )
        if len(collected) != 2:
            return None
        return {who: (who, status, body) for who, status, body in collected}
    finally:
        for process in racers:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)


def _admitted_and_refused(outcomes: dict) -> tuple[list, list]:
    """Exactly one start admits; return the two name-lists."""
    graph, task = outcomes["graph"], outcomes["task"]
    admitted = [
        name
        for name, result in (("graph", graph), ("task", task))
        if result[1] in (200, 201)
    ]
    refused = [
        name
        for name, result in (("graph", graph), ("task", task))
        if result[1] == 409
    ]
    check(
        len(admitted) == 1 and len(refused) == 1,
        f"exactly one start is admitted (graph {graph[1]}, task {task[1]})",
    )
    return admitted, refused


def _assert_loser_told_everything(loser, ledger) -> None:
    """The whole truth to the loser: code, breakdown, holders (rule 6)."""
    body = loser[2] if isinstance(loser[2], dict) else {}
    error = body.get("error", {}) if isinstance(body, dict) else {}
    check(
        error.get("code") == "memory_unavailable",
        f"the loser is 409 memory_unavailable ({error.get('code')})",
    )
    details = error.get("details", {})
    holders = details.get("holders", {})
    check(
        isinstance(holders, dict) and bool(holders),
        f"the breakdown names holders ({holders})",
    )
    names = list(holders) if isinstance(holders, dict) else []
    check(
        all(name.split(":")[0] in ("graph", "task") for name in names),
        f"every holder is a row-derived claim key ({names})",
    )
    message = str(error.get("message", ""))
    check(
        all(name in message for name in names),
        f"the message names every holder ({message!r})",
    )
    check(
        details.get("free_mb") == ledger.free_mb(),
        f"the breakdown says what was free, still true while the winner "
        f"runs ({details.get('free_mb')} vs {ledger.free_mb()})",
    )


def _assert_health_agrees(app, expected: float) -> None:
    status, _, body = asgi_request(app, "/api/v1/health")
    memory = body.get("memory") if isinstance(body, dict) else None
    check(
        status == 200
        and isinstance(memory, dict)
        and memory.get("held_mb") == expected,
        f"the health snapshot agrees with the ledger ({memory})",
    )


def _stop_winner_and_wait(app, admitted: list, outcomes: dict, ledger) -> None:
    """Cleanup through the real release path: stop, watch the card come back."""
    if admitted == ["graph"]:
        execution_id = outcomes["graph"][2].get("execution_id")
        asgi_request(
            app, f"/api/v1/graphs/executions/{execution_id}/stop",
            method="POST",
        )
    elif admitted == ["task"]:
        task_id = outcomes["task"][2].get("id")
        asgi_request(
            app, f"/api/v1/datasets/race/tasks/{task_id}/stop",
            method="POST",
        )
    check(
        wait_until(lambda: ledger.held_mb() == 0.0, timeout=15.0),
        f"stopping the winner hands the card back "
        f"(held {ledger.held_mb()})",
    )


def test_real_http_children_race_for_one_device() -> None:
    print("-- real processes: a graph start and a task start race over HTTP --")
    root = Path(tempfile.mkdtemp(prefix="backend-mem-http-"))
    services = build_services(project_root=root)
    app = create_app(services)
    _own_checkpoints(app, root)
    imgs = _image_dir(root)
    make_v2_dataset(root, "race")
    ledger = services.memory_ledger()
    check(ledger is not None, "fixture: the provider builds a ledger")
    assert ledger is not None

    served = _start_server(app)
    check(served is not None, "the server is up on a real socket")
    if served is None:
        return
    server, server_thread, port = served
    try:
        # The graph states 8000 (allocator) -> 8600 claimed on the
        # device; the task has no measured default -> exploratory,
        # exclusive. Either fits alone, neither fits with the other.
        graph_payload = {
            "nodes": [
                {"id": "w", "class_name": "SlowNode",
                 "params": {"seconds": 2.0}}
            ],
            "edges": [],
            "memory_overrides": {"vram_max_mb": 8000.0, "strict": False},
        }
        task_payload = {
            "kind": "ingest_lora",
            "image_dir": str(imgs),
            "model": "m.safetensors",
        }
        outcomes = _race_over_http(port, graph_payload, task_payload)
        if outcomes is None:
            check(False, "without both answers the race is unjudgeable")
            return
        admitted, _refused = _admitted_and_refused(outcomes)

        loser = None
        if outcomes["graph"][1] == 409:
            loser = outcomes["graph"]
        elif outcomes["task"][1] == 409:
            loser = outcomes["task"]
        if loser is not None:
            _assert_loser_told_everything(loser, ledger)

        # The winner's claim is live, exact, and within capacity.
        held = ledger.held_mb()
        expected = 8600.0 if admitted == ["graph"] else ledger.capacity_mb
        check(
            bool(admitted) and 0.0 < held <= ledger.capacity_mb,
            f"the winner's claim is held within capacity (held {held})",
        )
        check(
            held == expected,
            f"the winner holds exactly its claim ({held} vs {expected})",
        )
        _assert_health_agrees(app, expected)
        _stop_winner_and_wait(app, admitted, outcomes, ledger)
    finally:
        server.should_exit = True
        server_thread.join(timeout=10.0)
        check(not server_thread.is_alive(), "the server shut down cleanly")


# ==========================================================================
# MEM-04 #2: admission reads the remembered peak, the watcher files what a
# child reports -- under the same fingerprint key on the row
# ==========================================================================

#: What one graph's fingerprint looks like end to end: model, batch,
#: latent h/w (the largest good bucket of the ``shapes`` fixture, 64x64),
#: rank, checkpointing, optimizer.
_OBSERVED_KEY = "sdxl|2|64|64|64|True|adamw"


class _PeakProbeNode(Node):
    """A node that declares every fingerprint field and names a dataset.

    The fixture palette declares no ``memory_fields`` and no
    ``dataset_root`` input, so a graph that can actually be *fingerprinted*
    (MEM-01) needs its own node class: the observed path is only real if
    the start use case computed the key itself, from the registry's
    declarations and the dataset's latent buckets.
    """

    memory_fields = ("model", "batch_size", "rank", "checkpointing", "optimizer")

    INPUTS = {
        "model": Port(name="model", type=str, required=False, default="sdxl"),
        "batch_size": Port(name="batch_size", type=int, required=False, default=1),
        "rank": Port(name="rank", type=int, required=False, default=8),
        "checkpointing": Port(
            name="checkpointing", type=bool, required=False, default=False
        ),
        "optimizer": Port(name="optimizer", type=str, required=False, default="adamw"),
        "dataset_root": Port(
            name="dataset_root", type=str, required=False, default=None
        ),
    }
    OUTPUTS = {"ok": Port(name="ok", type=bool, doc="ran")}

    def build(
        self,
        model="sdxl",
        batch_size=1,
        rank=8,
        checkpointing=False,
        optimizer="adamw",
        dataset_root=None,
    ):
        return {"ok": True}


def _observed_graph(memory: MemorySettings | None = None) -> GraphDefinition:
    return GraphDefinition(
        nodes=(
            GraphNodeSpec(
                id="p",
                class_name="PeakProbeNode",
                params={
                    "model": "sdxl",
                    "batch_size": 2,
                    "rank": 64,
                    "checkpointing": True,
                    "optimizer": "adamw",
                    "dataset_root": "shapes",
                },
            ),
        ),
        edges=(),
        memory=memory if memory is not None else MemorySettings(),
    )


def _observed_setup(prefix: str, *, seed_peak: float | None):
    """Everything `_observed_case` does except the start itself.

    Returns ``(start, repo, provider)`` so a test can decide what to
    execute -- the floor tests (MEM-03H-02) need to run the same
    observed-demand setup with different memory settings, including one
    that must be refused before a row exists.
    """
    root = Path(tempfile.mkdtemp(prefix=prefix))
    make_v2_dataset(root, "shapes", items=3)
    registry = NodeRegistry(scan=lambda: ({"PeakProbeNode": _PeakProbeNode}, ()))
    store = SqlitePeakStore(root / "memory_peaks.db")
    if seed_peak is not None:
        store.record(_OBSERVED_KEY, seed_peak)
    repo = _graph_repo(root)
    provider = _provider(FakeDeviceProbe(), repo, _task_repo(root))
    peak_source = GraphPeakSource(
        datasets=SqliteDatasetLibrary(WorkspaceLayout(root)),
        peaks=store,
        resolve_memory_fields=memory_fields_resolver(registry),
    ).observed
    start = _graph_start(
        repo,
        provider,
        launcher=_AdmittedOnly(),
        clock=FakeClock(),
        runtime=ReflectedGraphRuntime(registry),
        peak_source=peak_source,
    )
    return start, repo, provider


def _observed_case(prefix: str, *, seed_peak: float | None):
    """Admit the fingerprintable graph once; return (row, ledger).

    The store is either seeded with ``seed_peak`` under the expected key
    or left empty -- the two worlds this package has to keep apart:
    measured and never-measured (the second must stay *unknown*, never
    become a zero).
    """
    start, repo, provider = _observed_setup(prefix, seed_peak=seed_peak)
    summary = start.execute(_observed_graph())
    return repo.get(summary.execution_id), provider()


def test_admission_reads_the_remembered_peak_and_keys_the_row() -> None:
    print("-- observed demand: admission reads the peak store, keys the row --")
    row, ledger = _observed_case("backend-mem-unmeasured-", seed_peak=None)
    memory = row.memory
    check(
        memory is not None and memory.fingerprint_key == _OBSERVED_KEY,
        f"the row carries the fingerprint admission computed "
        f"(got {memory.fingerprint_key if memory else None!r})",
    )
    check(
        memory is not None and memory.demand_source == "unknown",
        "a known fingerprint that was never measured is an unknown "
        "demand, not a zero claim",
    )
    check(ledger is not None, "fixture: the provider builds a ledger")
    assert ledger is not None and memory is not None
    check(
        memory.demand_mb == ledger.capacity_mb,
        f"unknown falls back to the exclusive capacity claim "
        f"({memory.demand_mb} vs {ledger.capacity_mb})",
    )
    check(
        ledger.held_mb() == row.reserved_mb,
        f"and the ledger holds exactly the claim the row recorded "
        f"({ledger.held_mb()} vs {row.reserved_mb})",
    )

    row2, ledger2 = _observed_case("backend-mem-remembered-", seed_peak=7000.0)
    memory2 = row2.memory
    check(
        memory2 is not None and memory2.demand_mb == 7150.0,
        f"observed demand = remembered peak + 150 pillow "
        f"(got {memory2.demand_mb if memory2 else None})",
    )
    check(memory2 is not None and memory2.demand_source == "observed",
          "and the demand is labeled observed")
    check(
        row2.reserved_mb == 7750.0,
        f"a grant is device MB: demand + 600 process overhead "
        f"(got {row2.reserved_mb})",
    )
    assert ledger2 is not None
    check(
        ledger2.held_mb() == 7750.0,
        f"the ledger holds exactly that (got {ledger2.held_mb()})",
    )


# MEM-03H-02: vram_min_mb was validated, stored and shown, but never
# answered for. These five tests are the floor it now has to cover.


def _floor_case(prefix: str, memory: MemorySettings):
    """A start wired like the other admission tests, with this memory.

    Unknown peaks (the default) plus a non-stated maximum leave the
    demand unknown -- the exploratory path where the declared floor is
    the only thing between the graph and a claim it cannot run inside.
    """
    root = Path(tempfile.mkdtemp(prefix=prefix))
    repo = _graph_repo(root)
    provider = _provider(FakeDeviceProbe(), repo, _task_repo(root))
    start = _graph_start(repo, provider, launcher=_AdmittedOnly())
    graph = GraphDefinition(
        nodes=valid_graph().nodes,
        edges=valid_graph().edges,
        memory=memory,
    )
    return start, repo, provider(), graph


def test_vram_min_floor_refuses_an_exploratory_start_it_cannot_cover() -> None:
    print("-- vram_min_mb: an exploratory start below the floor is refused --")
    start, repo, ledger, graph = _floor_case(
        "backend-mem-floor-high-", MemorySettings(vram_min_mb=11000)
    )
    check(ledger is not None, "fixture: the provider builds a ledger")
    assert ledger is not None
    check(
        ledger.capacity_mb == 11192.0,
        f"fixture: capacity 12216 - 1024 (got {ledger.capacity_mb})",
    )

    # 11000 + 600 process overhead = 11600 > 11192 free: even the
    # empty card cannot cover the floor, and the demand is unknown,
    # so this is an exploratory start.
    exc = _refusal(
        lambda: start.execute(graph),
        "a floor the empty card cannot cover is refused 409",
    )
    if exc is not None:
        check(
            exc.code == "memory_unavailable" and exc.status_code == 409,
            f"409 memory_unavailable (got {exc.status_code} {exc.code})",
        )
        check(
            "11000" in str(exc),
            f"the message names the floor (got {str(exc)!r})",
        )
        check(
            "11192" in str(exc),
            f"the message names what is free (got {str(exc)!r})",
        )
        details = exc.details or {}
        check(
            {
                "owner", "requested_mb", "capacity_mb", "foreign_reserve_mb",
                "free_mb", "holders", "reason", "what_would_fit",
            } <= set(details),
            f"the refusal carries the whole breakdown (got {sorted(details)})",
        )
        check(
            details.get("requested_mb") == 11600.0,
            f"requested = vram_min_mb + process overhead "
            f"(got {details.get('requested_mb')})",
        )
        reason = str(details.get("reason", ""))
        check(
            "vram_min_mb" in reason and "11192" in reason,
            f"the reason names the setting and the free space "
            f"(got {reason!r})",
        )
    check(
        ledger.held_mb() == 0.0,
        f"a refused floor holds nothing (got {ledger.held_mb()})",
    )
    check(len(repo.list_unfinished()) == 0, "and writes no row")


def test_vram_min_floor_admits_an_exploratory_start_it_can_cover() -> None:
    print("-- vram_min_mb: an exploratory start above the floor admits --")
    start, repo, ledger, graph = _floor_case(
        "backend-mem-floor-low-", MemorySettings(vram_min_mb=500)
    )
    assert ledger is not None

    summary = start.execute(graph)
    row = repo.get(summary.execution_id)
    check(
        ledger.held_mb() == ledger.capacity_mb,
        f"admitted as the exploratory claim of all free space "
        f"({ledger.held_mb()} vs {ledger.capacity_mb})",
    )
    check(
        row.reserved_mb == ledger.capacity_mb,
        f"the row records the same claim (got {row.reserved_mb})",
    )
    check(len(repo.list_unfinished()) == 1, "and the run has its row")


def test_vram_min_is_no_extra_condition_for_a_stated_demand_that_fits() -> None:
    print("-- vram_min_mb: a stated demand that fits is admitted as asked --")
    start, repo, ledger, graph = _floor_case(
        "backend-mem-floor-stated-",
        MemorySettings(vram_min_mb=1000, vram_max_mb=4096),
    )
    assert ledger is not None

    summary = start.execute(graph)
    row = repo.get(summary.execution_id)
    check(
        row.reserved_mb == 4696.0,
        f"the claim is the stated demand + overhead, neither inflated "
        f"nor refused by the floor (got {row.reserved_mb})",
    )
    check(
        ledger.held_mb() == 4696.0,
        f"the ledger holds exactly that (got {ledger.held_mb()})",
    )


def test_vram_min_above_the_remembered_demand_is_refused() -> None:
    print("-- vram_min_mb: a floor above the observed demand is refused --")
    start, repo, provider = _observed_setup(
        "backend-mem-floor-peak-", seed_peak=5000.0
    )
    ledger = provider()
    check(ledger is not None, "fixture: the provider builds a ledger")
    assert ledger is not None

    # The remembered peak says 5150 (5000 + 150 pillow), so the claim
    # would be 5750 device MB -- under a floor of 8000 + 600: a run
    # started below the minimum it declares.
    exc = _refusal(
        lambda: start.execute(_observed_graph(memory=MemorySettings(vram_min_mb=8000))),
        "a floor above the remembered demand is refused 409",
    )
    if exc is not None:
        check(
            exc.code == "memory_unavailable" and exc.status_code == 409,
            f"409 memory_unavailable (got {exc.status_code} {exc.code})",
        )
        check(
            "8000" in str(exc),
            f"the message names the floor (got {str(exc)!r})",
        )
        check(
            "5750" in str(exc),
            f"the message names the demand (got {str(exc)!r})",
        )
        details = exc.details or {}
        check(
            details.get("requested_mb") == 8600.0,
            f"requested = vram_min_mb + process overhead "
            f"(got {details.get('requested_mb')})",
        )
        check(
            "vram_min_mb" in str(details.get("reason", "")),
            f"the reason names the setting (got {details.get('reason')!r})",
        )
    check(
        ledger.held_mb() == 0.0,
        f"a refused floor holds nothing (got {ledger.held_mb()})",
    )
    check(len(repo.list_unfinished()) == 0, "and writes no row")


def test_vram_min_under_the_remembered_demand_admits() -> None:
    print("-- vram_min_mb: a floor under the observed demand admits --")
    start, repo, provider = _observed_setup(
        "backend-mem-floor-fit-", seed_peak=5000.0
    )
    ledger = provider()
    assert ledger is not None

    summary = start.execute(
        _observed_graph(memory=MemorySettings(vram_min_mb=4000))
    )
    row = repo.get(summary.execution_id)
    check(
        row.reserved_mb == 5750.0,
        f"the claim is the observed demand + overhead, untouched by "
        f"the floor (got {row.reserved_mb})",
    )
    check(
        ledger.held_mb() == 5750.0,
        f"the ledger holds exactly that (got {ledger.held_mb()})",
    )


def test_a_floor_never_answers_before_the_unknown_device_total() -> None:
    print("-- vram_min_mb: an unknown device total still answers first --")
    root = Path(tempfile.mkdtemp(prefix="backend-mem-floor-nocard-"))
    repo = _graph_repo(root)
    start = _graph_start(repo, lambda: None)
    graph = GraphDefinition(
        nodes=valid_graph().nodes,
        edges=valid_graph().edges,
        memory=MemorySettings(vram_min_mb=11000),
    )
    exc = _refusal(
        lambda: start.execute(graph),
        "no ledger is refused before any floor question is asked",
    )
    if exc is not None:
        details = exc.details or {}
        check(
            details.get("reason") == "device_total_unknown",
            f"rule 2: the explicit unknown, not the floor ({details})",
        )
        check(
            "vram_min_mb" not in str(exc),
            f"it does not pretend to have measured a floor "
            f"(got {str(exc)!r})",
        )
    check(len(repo.list_unfinished()) == 0, "and writes no row")


# MEM-03H-03: "unknown total" hides the restart-while-busy case, where
# the rows still say who holds the device. Both surfaces name them.


def _no_total_probe() -> FakeDeviceProbe:
    """A card whose properties cannot be read: total stays None.

    The busy case, not the absent one: `present` is true (the probe
    saw a device) but no total ever arrives, so the provider answers
    no ledger -- exactly what a restart while an adopted child holds
    the card produces.
    """
    return FakeDeviceProbe(
        DeviceReport(present=True, backend="xpu", name=None)
    )


def test_unknown_total_refusal_names_what_the_rows_still_claim() -> None:
    print("-- unknown total: the refusal names what the rows still claim --")
    root = Path(tempfile.mkdtemp(prefix="backend-mem-unknowntot-"))
    repo = _graph_repo(root)
    tasks = _task_repo(root)
    task = tasks.add(
        dataset="d", kind=TaskKind.INGEST_LORA, total=1, params={},
        reserved_mb=3000.0,
    )
    busy = _provider(_no_total_probe(), repo, tasks)
    check(busy() is None, "fixture: a card with no total builds no ledger")

    start = _graph_start(repo, busy, launcher=_AdmittedOnly())
    exc = _refusal(
        lambda: start.execute(valid_graph()),
        "a start with an unknown total is refused 409",
    )
    if exc is not None:
        holder = task_owner(task.id)
        check(
            holder in str(exc),
            f"the refusal names the adopted row's holder "
            f"(got {str(exc)!r})",
        )
        details = exc.details or {}
        check(
            details.get("reason") == "device_total_unknown",
            f"the reason is still the explicit unknown ({details})",
        )
        check(
            details.get("holders") == {holder: 3000.0},
            f"details.holders carries owner and size from the row "
            f"({details.get('holders')})",
        )
    check(
        len(repo.list_unfinished()) == 0,
        "the refused start writes no graph row",
    )


def test_task_start_with_an_unknown_total_names_the_graph_row() -> None:
    print("-- unknown total: a task start names the adopted graph row --")
    root = Path(tempfile.mkdtemp(prefix="backend-mem-unknowntot-task-"))
    repo = _graph_repo(root)
    tasks = _task_repo(root)
    execution = repo.add(
        GraphExecution.create(
            graph=valid_graph(), created_at=NOW, reserved_mb=7000.0
        )
    )
    busy = _provider(_no_total_probe(), repo, tasks)
    check(busy() is None, "fixture: a card with no total builds no ledger")

    # Assembled by hand, like the other no-ledger task start: the
    # wired container's provider is exactly what this case replaces.
    library = SqliteDatasetLibrary(WorkspaceLayout(root))
    ckpt = root / "ckpt"
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "m.safetensors").write_bytes(b"st")
    imgs = _image_dir(root)
    make_v2_dataset(root, "unknowntot")
    start_task = StartDatasetTask(
        library=library,
        tasks=tasks,
        gateway=FakeDatasetTaskGateway(),
        checkpoints_dir=lambda: ckpt,
        memory_ledger=busy,
    )
    exc = _refusal(
        lambda: start_task.execute(_task_cmd("unknowntot", imgs)),
        "a task start with an unknown total is refused 409",
    )
    if exc is not None:
        holder = graph_owner(execution.require_id())
        check(
            holder in str(exc),
            f"the refusal names the adopted graph row "
            f"(got {str(exc)!r})",
        )
        details = exc.details or {}
        check(
            details.get("holders") == {holder: 7000.0},
            f"details.holders carries owner and size from the row "
            f"({details.get('holders')})",
        )
    check(
        len(tasks.list_unfinished()) == 0,
        "the refused task writes no row",
    )


def test_health_names_row_holders_while_the_total_is_unknown() -> None:
    print("-- health: unknown total, holders named from the rows --")
    root = Path(tempfile.mkdtemp(prefix="backend-mem-unknowntot-health-"))
    graphs_db = SqliteDatabase(root / "test-graphs.db")
    graphs_db.initialize()
    repo = SqliteGraphExecutionRepository(graphs_db)
    execution = repo.add(
        GraphExecution.create(
            graph=valid_graph(), created_at=NOW, reserved_mb=7000.0
        )
    )
    services = build_services(
        project_root=root, device_probe=_no_total_probe(),
    )
    status, _, body = asgi_request(create_app(services), "/api/v1/health")
    memory = body.get("memory") if isinstance(body, dict) else None
    check(
        status == 200
        and isinstance(memory, dict)
        and memory.get("total_mb") is None,
        f"an unknown total stays an explicit unknown, never a zero "
        f"({memory})",
    )
    check(
        isinstance(memory, dict)
        and memory.get("holders")
        == {graph_owner(execution.require_id()): 7000.0},
        f"health names what the rows still claim ({memory})",
    )


class _RecordingPeakStore(PeakStore):
    """An in-memory PeakStore that remembers the order of the writes.

    The SQLite store's ``MAX`` semantics would hide *which* frames
    reached it: filed-then-overtaken is indistinguishable from never
    filed. What MEM-04 #2 promises is one statement **only when the peak
    rose**, and only a recording double can see that.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, float]] = []
        self._peaks: dict[str, float] = {}

    def record(self, fingerprint: str, peak_mb: float) -> float:
        self.calls.append((fingerprint, peak_mb))
        stored = max(self._peaks.get(fingerprint, peak_mb), peak_mb)
        self._peaks[fingerprint] = stored
        return stored

    def peak_mb(self, fingerprint: str) -> float | None:
        return self._peaks.get(fingerprint)


_REPO_ROOT = Path(__file__).resolve().parents[2]


class _MemoryReportChild(GraphTaskGateway):
    """A gateway whose child is a *real* process writing real records.

    The writer, the event file, the tail and the watcher under test are
    all the production ones; only the content is pinned, because the
    fixed-interval telemetry that will produce these frames does not
    exist yet (MEM-05 #4). Cross-process property, real process: three
    memory frames -- 6000, a rise to 7200, a dip back to 6500 -- and a
    clean outcome.
    """

    _SCRIPT = (
        "import sys\n"
        "sys.path.insert(0, {root!r})\n"
        "from pathlib import Path\n"
        "from backend.infrastructure.graph_event_stream import ExecutionEventWriter\n"
        "w = ExecutionEventWriter(Path({event!r}))\n"
        "w.memory(reserved_mb=5000.0, allocated_mb=4000.0, peak_mb=6000.0, budget_mb=None)\n"
        "w.memory(reserved_mb=5000.0, allocated_mb=4500.0, peak_mb=7200.0, budget_mb=None)\n"
        "w.memory(reserved_mb=5000.0, allocated_mb=4500.0, peak_mb=6500.0, budget_mb=None)\n"
        "w.outcome(error=None, results_count=0)\n"
        "w.close()\n"
    )

    def __init__(self) -> None:
        self._proc: subprocess.Popen[bytes] | None = None

    def spawn(self, launch: GraphTaskLaunch) -> int:
        script = self._SCRIPT.format(
            root=str(_REPO_ROOT), event=str(launch.event_path)
        )
        self._proc = subprocess.Popen([sys.executable, "-c", script])
        return self._proc.pid

    def request_stop(self, pid: int) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()

    def kill(self, pid: int) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.kill()

    def is_alive(self, pid: int) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def find_running_all(self, execution_id) -> list[int]:
        return []

    def find_running(self, execution_id) -> int | None:
        return None


def test_child_reported_peaks_land_in_the_store() -> None:
    print("-- a real child process reports peaks; the watcher files the rises --")
    root = Path(tempfile.mkdtemp(prefix="backend-mem-childpeak-"))
    repo = _graph_repo(root)
    clock = FakeClock()
    provider = _provider(FakeDeviceProbe(), repo, _task_repo(root))
    store = _RecordingPeakStore()
    gateway = _MemoryReportChild()
    supervisor = GraphExecutionSupervisor(
        executions=repo,
        writer=_writer(repo, clock),
        gateway=gateway,
        events=EventPublisher(events=RecordingEventBus()),
        clock=clock,
        peak_store=store,
        scratch_dir=root / "scratch",
        make_tail=ExecutionEventTail,
        poll_interval=0.02,
    )
    start = _graph_start(
        repo,
        provider,
        launcher=supervisor,
        clock=clock,
        peak_source=lambda _graph: ObservedPeak("fp-e2e", None),
    )
    summary = start.execute(valid_graph())
    row = repo.get(summary.execution_id)
    check(
        row.memory is not None and row.memory.fingerprint_key == "fp-e2e",
        "the row carries the key the watcher files this run's peaks under",
    )
    check(
        wait_until(
            lambda: repo.get(summary.execution_id).status.is_terminal,
            timeout=30.0,
        ),
        "the run reaches a terminal state",
    )
    check(
        gateway._proc is not None and gateway._proc.returncode == 0,
        f"the child wrote its records cleanly (rc="
        f"{gateway._proc.returncode if gateway._proc else 'no proc'})",
    )
    check(
        store.peak_mb("fp-e2e") == 7200.0,
        f"the child's peak landed in the store (got {store.peak_mb('fp-e2e')})",
    )
    check(
        store.calls == [("fp-e2e", 6000.0), ("fp-e2e", 7200.0)],
        f"only rises cost a statement -- the 6500 tail frame was skipped "
        f"(got {store.calls})",
    )
    check(
        supervisor._peak_seen == {},
        "the per-run rise bookkeeping is released with the run",
    )


def test_reconcile_drain_files_peaks() -> None:
    print("-- the post-restart drain files a peak no live watcher saw --")
    root = Path(tempfile.mkdtemp(prefix="backend-mem-drain-"))
    repo = _graph_repo(root)
    clock = FakeClock()
    store = SqlitePeakStore(root / "memory_peaks.db")
    supervisor = GraphExecutionSupervisor(
        executions=repo,
        writer=_writer(repo, clock),
        gateway=None,
        events=EventPublisher(events=RecordingEventBus()),
        clock=clock,
        peak_store=store,
        scratch_dir=root / "scratch",
        make_tail=ExecutionEventTail,
    )
    row = GraphExecution.create(
        graph=valid_graph(),
        created_at=NOW,
        memory=effective_memory(MemorySettings(), None, None, "fp-drain", 11192.0),
    )
    repo.add(row)
    events_path = supervisor._paths_for(row.id)["events"]
    events_path.parent.mkdir(parents=True, exist_ok=True)
    writer = ExecutionEventWriter(events_path)
    writer.memory(
        reserved_mb=4000.0, allocated_mb=3000.0, peak_mb=5000.0, budget_mb=None
    )
    writer.outcome(error=None, results_count=0)
    writer.close()
    outcome = supervisor.recorded_outcome(row.id)
    check(
        outcome is not None and outcome.error is None,
        "the drain reads the run's own outcome",
    )
    check(
        store.peak_mb("fp-drain") == 5000.0,
        f"and files the peak the run reported while no one was watching "
        f"(got {store.peak_mb('fp-drain')})",
    )


def test_peak_filing_guards() -> None:
    print("-- what never reaches the store: no key, no usable number, no rise --")
    root = Path(tempfile.mkdtemp(prefix="backend-mem-peaksafe-"))
    repo = _graph_repo(root)
    clock = FakeClock()
    store = _RecordingPeakStore()
    supervisor = GraphExecutionSupervisor(
        executions=repo,
        writer=_writer(repo, clock),
        gateway=None,
        events=EventPublisher(events=RecordingEventBus()),
        clock=clock,
        peak_store=store,
        scratch_dir=root / "scratch",
        make_tail=ExecutionEventTail,
    )
    # A row whose fingerprint admission could not compute (also the shape
    # of every row written before this package existed).
    keyless = GraphExecution.create(graph=valid_graph(), created_at=NOW)
    repo.add(keyless)
    supervisor._record_peak(keyless.id, {"peak_mb": 9000.0})
    check(
        store.calls == [],
        "unknown fingerprint -> nothing filed (the watcher has no key, "
        "exactly like admission had no past)",
    )

    keyed = GraphExecution.create(
        graph=valid_graph(),
        created_at=NOW,
        memory=effective_memory(MemorySettings(), None, None, "fp-guards", 11192.0),
    )
    repo.add(keyed)
    for payload in (
        {},
        {"peak_mb": None},
        {"peak_mb": "lots"},
        {"peak_mb": 0.0},
        {"peak_mb": -4.0},
        {"peak_mb": float("nan")},
    ):
        supervisor._record_peak(keyed.id, dict(payload))
    check(
        store.calls == [],
        f"malformed / non-positive peaks are refused, not stored "
        f"(got {store.calls})",
    )
    supervisor._record_peak(keyed.id, {"peak_mb": 1000.0})
    check(store.calls == [("fp-guards", 1000.0)], "a measured peak is filed")
    supervisor._record_peak(keyed.id, {"peak_mb": 900.0})
    check(
        store.calls == [("fp-guards", 1000.0)],
        "a lower frame for the same run costs no statement",
    )
    supervisor._record_peak(keyed.id, {"peak_mb": 1200.0})
    check(
        store.calls == [("fp-guards", 1000.0), ("fp-guards", 1200.0)],
        "and the next rise is filed again",
    )

    # A store that raises must cost the record, never the supervision.
    class _Exploding(PeakStore):
        def record(self, fingerprint: str, peak_mb: float) -> float:
            raise RuntimeError("database is locked")

        def peak_mb(self, fingerprint: str) -> float | None:
            return None

    boom = GraphExecutionSupervisor(
        executions=repo,
        writer=_writer(repo, clock),
        gateway=None,
        events=EventPublisher(events=RecordingEventBus()),
        clock=clock,
        peak_store=_Exploding(),
        scratch_dir=root / "scratch2",
        make_tail=ExecutionEventTail,
    )
    boom._record_peak(keyed.id, {"peak_mb": 2000.0})
    check(True, "a store that raises cannot abort the watcher (it logged)")


def test_bad_budget_numbers_never_reach_the_ledger() -> None:
    print("-- 422 at the edge: a bad budget claims nothing, writes no row --")
    root = Path(tempfile.mkdtemp(prefix="backend-mem-badnum-"))
    repo = _graph_repo(root)
    services = build_services(project_root=root, graph_executions=repo)
    app = create_app(services)
    ledger = services.memory_ledger()
    check(ledger is not None, "fixture: the provider builds a ledger")
    assert ledger is not None
    check(
        ledger.free_mb() == ledger.capacity_mb,
        "fixture: the card starts free",
    )

    # The reproduction seeds, through the real endpoint (MEM-03H-01):
    # -9000 used to be a 201 that made the ledger report 19,592 MB free
    # on an 11,192 MB card. The inverted pair is the model-level rule,
    # whose 422 loc is the block -- the field names ride in the message.
    for memory, field in (
        ({"vram_max_mb": -9000}, "vram_max_mb"),
        ({"vram_min_mb": -1}, "vram_min_mb"),
        ({"vram_max_mb": True}, "vram_max_mb"),
        ({"vram_max_mb": float("nan")}, "vram_max_mb"),
        ({"vram_max_mb": 0}, "vram_max_mb"),
        ({"vram_max_mb": 10**400}, "vram_max_mb"),
        ({"vram_min_mb": 5000, "vram_max_mb": 4096}, "vram_min_mb"),
    ):
        status, _, body = asgi_request(
            app,
            "/api/v1/graphs/run",
            method="POST",
            json_body={"nodes": [], "memory": memory},
        )
        check(
            status == 422 and field in json.dumps(body),
            f"{memory} -> 422 naming {field} (got {status}: {body})",
        )

    # The override path is the same door (the seed used it too).
    status, _, body = asgi_request(
        app,
        "/api/v1/graphs/run",
        method="POST",
        json_body={"nodes": [], "memory_overrides": {"vram_max_mb": -9000}},
    )
    check(
        status == 422 and "vram_max_mb" in json.dumps(body),
        f"overrides vram_max_mb=-9000 -> 422 (got {status}: {body})",
    )

    # What a rejection must leave behind: nothing.
    check(
        ledger.free_mb() == ledger.capacity_mb,
        f"every rejected submission claimed nothing: free {ledger.free_mb()} "
        f"== capacity {ledger.capacity_mb}",
    )
    check(
        len(repo.list_unfinished()) == 0,
        "and not one execution row was written",
    )


def main() -> None:
    test_refused_run_holds_nothing_and_names_the_holder()
    test_crash_between_reserve_and_spawn_releases_then_converges()
    test_restart_reproduces_the_held_total_from_both_kinds_of_row()
    test_release_is_idempotent_on_every_path()
    test_a_task_blocks_a_graph_and_a_graph_blocks_a_task()
    test_parallel_task_starts_admit_exactly_one()
    test_without_a_ledger_every_start_refuses_explicitly()
    test_health_reports_the_live_ledger_snapshot()
    test_the_probe_is_admitted_and_degrades_when_refused()
    test_real_http_children_race_for_one_device()
    test_admission_reads_the_remembered_peak_and_keys_the_row()
    test_child_reported_peaks_land_in_the_store()
    test_reconcile_drain_files_peaks()
    test_peak_filing_guards()
    test_bad_budget_numbers_never_reach_the_ledger()
    test_vram_min_floor_refuses_an_exploratory_start_it_cannot_cover()
    test_vram_min_floor_admits_an_exploratory_start_it_can_cover()
    test_vram_min_is_no_extra_condition_for_a_stated_demand_that_fits()
    test_vram_min_above_the_remembered_demand_is_refused()
    test_vram_min_under_the_remembered_demand_admits()
    test_a_floor_never_answers_before_the_unknown_device_total()
    test_unknown_total_refusal_names_what_the_rows_still_claim()
    test_task_start_with_an_unknown_total_names_the_graph_row()
    test_health_names_row_holders_while_the_total_is_unknown()
    finish()


if __name__ == "__main__":
    main()
