"""Shared test support: assertions, fakes, and a raw-ASGI client.

House style (matches the repo's smoke tests): each ``test_*.py`` file
runs standalone -- ``check()`` prints PASS/FAIL per assertion, and
``finish()`` exits non-zero if anything failed. No pytest dependency.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.application.ports.config_inspector import (
    ConfigDescription,
    ConfigInspector,
    ConfigSummary,
    StartOption,
)
from backend.application.ports.graph_execution_repository import (
    GraphExecutionRepository,
)
from backend.application.ports.graph_library import GraphLibrary
from backend.application.ports.graph_runtime import GraphRuntime
from backend.application.ports.monitor_bus import MonitorBus as MonitorBusPort
from backend.application.ports.run_repository import RunRepository
from backend.application.ports.training_gateway import (
    TrainingGateway,
    TrainingLaunch,
)
from backend.application.dataset_task_sweeper import DatasetTaskSweeper
from backend.application.errors import ConfigNotFoundError
from backend.application.project_paths import ProjectPaths
from backend.application.graph_supervisor import GraphExecutionSupervisor
from backend.application.event_publisher import EventPublisher
from backend.application.lifecycle_writer import (
    ExecutionLifecycleWriter,
    RunLifecycleWriter,
)
from backend.application.services import (
    MonitorServices,
    ApplicationServices,
    AssetServices,
    ConfigServices,
    DatasetServices,
    GraphServices,
    SettingsServices,
)
from backend.application.supervisor import RunSupervisor
from backend.application.use_cases import (
    SubscribeMonitor,
    BrowseAssets,
    BulkUpdateDatasetItems,
    CommitDatasetItems,
    CreateDataset,
    DeleteDataset,
    DeleteGraph,
    DeleteGraphExecutions,
    DeleteRuns,
    DiscardDatasetItems,
    GetActiveRun,
    GetConfig,
    GetConfigOptions,
    GetDataset,
    GetGraph,
    GetGraphExecution,
    GetRun,
    GetRunLog,
    GetSettings,
    GetStartOptions,
    InspectAsset,
    ListAssets,
    ListDatasetItems,
    ListDatasetSets,
    ListDatasetTasks,
    ListDatasets,
    ListGraphExecutions,
    ListGraphs,
    ListNodeCatalog,
    ListRuns,
    MakeAssetFolder,
    NodeDiagnostics,
    ReadConfigRaw,
    ReadDatasetFile,
    ReconcileDatasetTasks,
    ReconcileGraphExecutions,
    ReconcileRuns,
    SaveGraph,
    SetDatasetPreview,
    StartDatasetTask,
    StartGraphExecution,
    StartTraining,
    StopDatasetTask,
    StopGraphExecution,
    StopTraining,
    UpdateConfig,
    UpdateDatasetItem,
    UpdateSettings,
    UploadAsset,
    ValidateGraph,
    WriteConfigRaw,
)
from backend.domain.entities.run import Run
from backend.domain.events import DomainEvent
from backend.domain.exceptions import DomainError
from backend.domain.value_objects import RunId, RunStatus
from backend.application.ports.dataset_task_gateway import (
    DatasetTaskGateway,
    DatasetTaskLaunch,
)
from backend.infrastructure.config_options import PydanticConfigOptions
from backend.infrastructure.core_config_files import CoreConfigFiles
from backend.infrastructure.core_config_inspector import CoreConfigInspector
from backend.infrastructure.dataset_files import FsDatasetFiles
from backend.infrastructure.dataset_library import SqliteDatasetLibrary
from backend.infrastructure.dataset_previews import SqliteDatasetPreviews
from backend.infrastructure.dataset_tasks import SqliteDatasetTasks
from backend.infrastructure.directory_run_artifacts import DirectoryRunArtifacts
from backend.infrastructure.events.callback_event_bus import CallbackEventBus
from backend.infrastructure.file_asset_store import FileSystemAssetStore
from backend.infrastructure.graph.catalog import DiscoveredGraphCatalog
from backend.infrastructure.graph.discovery import NodeRegistry
from backend.infrastructure.graph.runtime import ReflectedGraphRuntime
from backend.infrastructure.jsonl_progress_source import JsonlProgressSource as _Jsonl
from backend.infrastructure.monitor_bus import SharedMonitorBus
from backend.infrastructure.persistence.graph_execution_repository import (
    SqliteGraphExecutionRepository,
)
from backend.infrastructure.persistence.graph_library import SqliteGraphLibrary
from backend.infrastructure.persistence.sqlite import SqliteDatabase
from backend.infrastructure.settings_store import SqliteSettingsStore
from backend.infrastructure.workspace import WorkspaceLayout
from nodes.core import Node, NodePreset, Port

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    print(f"  {'PASS' if condition else 'FAIL'}: {message}")
    if not condition:
        FAILURES.append(message)


def finish() -> None:
    print()
    print("=" * 60)
    if FAILURES:
        print(f"SMOKE TEST: {len(FAILURES)} FAILURE(S)")
        for failure in FAILURES:
            print(f"  - {failure}")
        sys.exit(1)
    print("SMOKE TEST: ALL CHECKS PASSED")


def wait_until(predicate, *, timeout: float = 2.0, interval: float = 0.01) -> bool:
    """Poll ``predicate`` until true or timeout; for supervisor threads."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class FakeClock:
    """Deterministic clock; starts at a fixed instant and only moves
    when a test says so."""

    def __init__(self, start: datetime | None = None) -> None:
        self._now = start or datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    def now(self) -> datetime:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += timedelta(seconds=seconds)


class RecordingEventBus(CallbackEventBus):
    """The real bus plus a record of everything published."""

    def __init__(self) -> None:
        super().__init__()
        self.published: list[DomainEvent] = []

    def publish(self, event: DomainEvent) -> None:
        self.published.append(event)
        super().publish(event)

    def types(self) -> list[str]:
        return [event.event_type for event in self.published]


class InMemoryRunRepository(RunRepository):
    """Same semantics as the SQLite port, no persistence.

    ``_statuses`` mirrors the persisted status separately from the
    entity objects (which callers mutate *before* writing): the CAS
    must see what the "row" said, not what some in-flight entity copy
    now says.
    """

    _UNFINISHED = (RunStatus.CREATED, RunStatus.RUNNING)

    def __init__(self) -> None:
        self._runs: dict[int, Run] = {}
        self._statuses: dict[int, RunStatus] = {}
        self._next_id = 1

    def add(self, run: Run) -> Run:
        if run.id is not None:
            raise DomainError(f"run already has id {run.id}")
        run.assign_id(RunId(self._next_id))
        self._next_id += 1
        self._runs[run.id] = run
        self._statuses[run.id] = run.status
        return run

    def get(self, run_id: RunId) -> Run | None:
        return self._runs.get(run_id)

    def list(self, *, limit: int = 50, status: RunStatus | None = None) -> list[Run]:
        runs = [r for r in self._runs.values() if status is None or r.status is status]
        runs.sort(key=lambda r: r.id, reverse=True)
        return runs[:limit]

    def update(self, run: Run) -> bool:
        if run.id is None:
            raise DomainError("cannot update an unpersisted run (no id yet)")
        if run.id not in self._runs:
            return False
        self._runs[run.id] = run
        self._statuses[run.id] = run.status
        return True

    def update_if_status(self, run: Run, expected: RunStatus) -> bool:
        if run.id is None:
            raise DomainError("cannot update an unpersisted run (no id yet)")
        if run.id not in self._runs:
            return False
        if self._statuses.get(run.id) is not expected:
            return False
        self._runs[run.id] = run
        self._statuses[run.id] = run.status
        return True

    def find_active(self) -> Run | None:
        unfinished = [
            r
            for r in self._runs.values()
            if self._statuses.get(r.id) in self._UNFINISHED
        ]
        return max(unfinished, key=lambda r: r.id) if unfinished else None

    def continue_ids_above(self, run_id: RunId) -> None:
        self._next_id = max(self._next_id, int(run_id) + 1)

    def list_unfinished(self) -> list[Run]:
        unfinished = [
            r
            for r in self._runs.values()
            if self._statuses.get(r.id) in self._UNFINISHED
        ]
        unfinished.sort(key=lambda r: r.id, reverse=True)
        return unfinished

    def delete_all(self) -> int:
        deleted = len(self._runs)
        self._runs.clear()
        self._statuses.clear()
        return deleted


class FakeTrainingGateway(TrainingGateway):
    """Scriptable gateway: tests decide what is alive and how it exits.

    spawn() registers the pid as alive with exit code 0; tests kill it
    by discarding the pid from ``alive`` (setting ``exit_codes[pid]``
    first for a non-zero exit). ``spawn_error`` fails the launch.
    """

    def __init__(self) -> None:
        self.spawned: list[TrainingLaunch] = []
        self.stopped: list[tuple[int, bool]] = []
        self.killed: list[int] = []
        self.alive: set[int] = set()
        self.foreign: set[int] = set()  # pids whose number was recycled
        self.exit_codes: dict[int, int] = {}
        self.spawn_error: Exception | None = None
        self.next_pid = 4242

    def spawn(self, launch: TrainingLaunch) -> int:
        if self.spawn_error is not None:
            raise self.spawn_error
        self.spawned.append(launch)
        pid = self.next_pid
        self.next_pid += 1
        self.alive.add(pid)
        self.exit_codes[pid] = 0
        return pid

    def is_alive(self, pid: int) -> bool:
        return pid in self.alive

    def owns(self, pid: int) -> bool:
        """Identity: every scripted pid is ours unless the test says
        otherwise (``foreign`` marks a recycled pid number)."""
        return pid not in self.foreign

    def wait_exit_code(self, pid: int, timeout: float = 5.0) -> int | None:
        if pid in self.alive:
            return None
        return self.exit_codes.get(pid)

    def stop(self, pid: int, *, force: bool = False) -> bool:
        self.stopped.append((pid, force))
        self.alive.discard(pid)
        return True

    def kill(self, pid: int) -> bool:
        self.killed.append(pid)
        # Same contract as the real adapter: a pid that is not ours is
        # never signalled, whatever kill() is called for.
        if pid in self.foreign or pid not in self.alive:
            return False
        self.alive.discard(pid)
        return True


class FakeDatasetTaskGateway(DatasetTaskGateway):
    """Scriptable fork gateway: spawn registers a fake pid as alive;
    tests kill it by discarding from ``alive``; ``spawn_error`` fails
    the launch (mirrors FakeTrainingGateway's posture)."""

    def __init__(self) -> None:
        self.spawned: list[DatasetTaskLaunch] = []
        self.killed: list[int] = []
        self.alive: set[int] = set()
        self.spawn_error: Exception | None = None
        self.next_pid = 7777

    def spawn(self, launch: DatasetTaskLaunch) -> int:
        if self.spawn_error is not None:
            raise self.spawn_error
        self.spawned.append(launch)
        pid = self.next_pid
        self.next_pid += 1
        self.alive.add(pid)
        return pid

    def kill(self, pid: int) -> None:
        self.killed.append(pid)
        self.alive.discard(pid)

    def is_alive(self, pid: int) -> bool:
        return pid in self.alive


class FakeConfigInspector(ConfigInspector):
    """Existence check is real; summaries/descriptions are scripted.

    ``invalid`` maps a path string to an exception to raise (for
    config_invalid tests); everything else resolves to the defaults.
    """

    def __init__(self) -> None:
        self.default = ConfigSummary(mode="distillation", total_steps=100)
        self.summaries: dict[str, ConfigSummary] = {}
        self.descriptions: dict[str, ConfigDescription] = {}
        self.default_description = ConfigDescription(
            mode="distillation",
            total_steps=100,
            start_from={
                "teacher": StartOption(path="", available=False, label="Base Model"),
                "student": StartOption(path="", available=False, label="Student"),
                "resume": StartOption(path="", available=False, label="Resume"),
            },
        )
        self.invalid: dict[str, Exception] = {}

    def _guard(self, config_path: Path) -> str:
        key = str(config_path)
        if key in self.invalid:
            raise self.invalid[key]
        if not config_path.exists():
            raise ConfigNotFoundError(f"config file not found: {config_path}")
        return key

    def summarize(self, config_path: Path) -> ConfigSummary:
        key = self._guard(config_path)
        return self.summaries.get(key, self.default)

    def describe(self, config_path: Path) -> ConfigDescription:
        key = self._guard(config_path)
        return self.descriptions.get(key, self.default_description)


# --------------------------------------------------------------------------
# Graph fixture nodes (M4)
# --------------------------------------------------------------------------
#
# Defined here rather than under nodes/ because the catalog/runtime/
# execution tests want deterministic shapes -- every validation issue
# code has a class that can trigger it -- without walking all 94 real
# modules. stdlib-only, same as nodes.core itself, so no test in this
# file needs torch.


class SumNode(Node):
    """Add two numbers."""

    INPUTS = {
        "a": Port(name="a", type=float, doc="first addend"),
        "b": Port(name="b", type=float, doc="second addend"),
    }
    OUTPUTS = {"sum": Port(name="sum", type=float, doc="a + b")}

    def build(self, a, b):
        return {"sum": a + b}


class ScaleNode(Node):
    """Multiply or divide (closed ``choices`` set for invalid_choice)."""

    INPUTS = {
        "value": Port(name="value", type=float, doc="input value"),
        "factor": Port(name="factor", type=float, required=False, default=2.0),
        "mode": Port(
            name="mode", type=str, required=False, default="mul",
            choices=("mul", "div"),
        ),
    }
    OUTPUTS = {"scaled": Port(name="scaled", type=float)}

    def build(self, value, factor=2.0, mode="mul"):
        return {"scaled": value * factor if mode == "mul" else value / factor}


class LabelNode(Node):
    """String output -- str -> float wiring is an incompatible_types issue."""

    INPUTS = {"text": Port(name="text", type=str)}
    OUTPUTS = {"label": Port(name="label", type=str)}

    def build(self, text):
        return {"label": text}


class ObjectNode(Node):
    """Non-JSON output: reported as a ``_type``/``_repr`` summary."""

    INPUTS = {"token": Port(name="token", type=str, required=False, default="x")}
    OUTPUTS = {"obj": Port(name="obj", type=Any)}

    def build(self, token="x"):
        return {"obj": object()}


class BoomNode(Node):
    """build() always raises -- a node failure is a normal outcome."""

    INPUTS = {"trigger": Port(name="trigger", type=bool, required=False, default=True)}
    OUTPUTS = {"never": Port(name="never", type=bool)}

    def build(self, trigger=True):
        raise RuntimeError("boom")


class PickyNode(Node):
    """Shape hook that raises -- reported as shape_resolution_failed."""

    INPUTS = {"x": Port(name="x", type=float)}
    OUTPUTS = {"y": Port(name="y", type=float)}

    @classmethod
    def resolve_inputs(cls, params: dict):
        raise ValueError("shape unavailable")

    def build(self, x):
        return {"y": x}


class DiagnosingNode(Node):
    """diagnostics() overridden -- has_diagnostics=True."""

    INPUTS = {"path": Port(name="path", type=str, required=False, default="")}
    OUTPUTS = {"ok": Port(name="ok", type=bool)}

    def build(self, path=""):
        return {"ok": True}

    def diagnostics(self, inputs: dict) -> dict[str, list[str]]:
        return {"path": [f"looked at {inputs.get('path')!r}"]}


class BadDiagnosticsNode(Node):
    """diagnostics() raises -- 400 node_diagnostics_failed, never 500."""

    INPUTS = {"x": Port(name="x", type=float, required=False, default=0.0)}
    OUTPUTS = {"y": Port(name="y", type=float)}

    def build(self, x=0.0):
        return {"y": x}

    def diagnostics(self, inputs: dict) -> dict[str, list[str]]:
        raise ValueError("cannot resolve mid-edit path")


class PresetChoiceNode(Node):
    """Dynamic kind: presets are its palette payload."""

    NODE_KIND = "dynamic"

    INPUTS = {
        "source": Port(name="source", type=str),
        "strength": Port(name="strength", type=float, required=False, default=1.0),
    }
    OUTPUTS = {"result": Port(name="result", type=str)}

    @classmethod
    def list_presets(cls) -> list[NodePreset]:
        return [
            NodePreset(
                name="identity",
                required_inputs={"source": Port(name="source", type=str)},
                required_outputs={"result": Port(name="result", type=str)},
            ),
        ]

    def build(self, source, strength=1.0):
        return {"result": source}


class PathNode(Node):
    """Path-typed input: str accepted (editors send strings), int not."""

    INPUTS = {
        "path": Port(
            name="path", type=Path, required=False, default=None,
            path_kind="checkpoint",
        )
    }
    OUTPUTS = {"exists": Port(name="exists", type=bool)}

    def build(self, path=None):
        return {"exists": True}


class SlowNode(Node):
    """Sleeps briefly -- gives stop/single-active tests a live window."""

    INPUTS = {
        "seconds": Port(name="seconds", type=float, required=False, default=0.3)
    }
    OUTPUTS = {"slept": Port(name="slept", type=float)}

    def build(self, seconds=0.3):
        time.sleep(seconds)
        return {"slept": seconds}


class Handle:
    """Marker base classes: edge compatibility is a real issubclass
    check on real class objects, so SubHandle satisfies a Handle input."""


class SubHandle(Handle):
    pass


class EmitHandleNode(Node):
    """Source node with a class-typed output (Handle)."""

    INPUTS = {}
    OUTPUTS = {"handle": Port(name="handle", type=Handle)}

    def build(self):
        return {"handle": Handle()}


class EmitSubHandleNode(Node):
    """Source node whose output is a Handle *subclass*."""

    INPUTS = {}
    OUTPUTS = {"handle": Port(name="handle", type=SubHandle)}

    def build(self):
        return {"handle": SubHandle()}


class MonitorProbeNode(Node):
    """Touches the execution context's monitor bus (M6 wiring proof).

    Mirrors what MonitorNode does at build start: clear the id, then
    hand a handle-like report through. ``seen`` says whether a bus was
    actually injected (None = the runtime was built without one).
    """

    INPUTS = {
        "monitor_id": Port(name="monitor_id", type=str, required=False, default="mon-test")
    }
    OUTPUTS = {"seen": Port(name="seen", type=bool)}

    def build(self, monitor_id="mon-test"):
        bus = self.context.monitor_bus
        if bus is not None:
            bus.clear(monitor_id)
            bus.report(monitor_id, {"step": 1, "loss": 0.5})
        return {"seen": bus is not None}


class RecordingMonitorBus(MonitorBusPort):
    """Port test double: records thread-side calls; no queues/loops."""

    def __init__(self) -> None:
        self.reports: list[tuple[str, dict]] = []
        self.cleared: list[str] = []

    def report(self, monitor_id: str, data: dict) -> None:
        self.reports.append((monitor_id, dict(data)))

    def clear(self, monitor_id: str) -> None:
        self.cleared.append(monitor_id)

    def subscribe(self, monitor_id: str):  # pragma: no cover - not loop-bound
        raise NotImplementedError("RecordingMonitorBus has no subscriber side")

    def unsubscribe(self, monitor_id: str, queue) -> None:  # pragma: no cover
        raise NotImplementedError("RecordingMonitorBus has no subscriber side")


class TakeHandleNode(Node):
    """Class-typed input (Handle) -- accepts SubHandle outputs."""

    INPUTS = {"handle": Port(name="handle", type=Handle)}
    OUTPUTS = {"ok": Port(name="ok", type=bool)}

    def build(self, handle):
        return {"ok": isinstance(handle, Handle)}


FIXTURE_NODES: dict[str, type] = {
    cls.__name__: cls
    for cls in (
        SumNode,
        ScaleNode,
        LabelNode,
        ObjectNode,
        BoomNode,
        PickyNode,
        DiagnosingNode,
        BadDiagnosticsNode,
        PresetChoiceNode,
        PathNode,
        SlowNode,
        EmitHandleNode,
        EmitSubHandleNode,
        TakeHandleNode,
        MonitorProbeNode,
    )
}


def fixture_graph_registry() -> NodeRegistry:
    """A NodeRegistry over the fixture classes above (no nodes/ walk,
    no import errors, deterministic ordering)."""
    return NodeRegistry(scan=lambda: (dict(FIXTURE_NODES), ()))


def build_services(
    *,
    runs: RunRepository | None = None,
    events: RecordingEventBus | None = None,
    gateway: TrainingGateway | None = None,
    inspector: ConfigInspector | None = None,
    artifacts: DirectoryRunArtifacts | None = None,
    supervisor: RunSupervisor | None = None,
    clock: FakeClock | None = None,
    project_root: Path | None = None,
    runs_dir: Path | None = None,
    poll_interval: float = 0.05,
    settings_store: SqliteSettingsStore | None = None,
    config_files: CoreConfigFiles | None = None,
    config_options: PydanticConfigOptions | None = None,
    assets: FileSystemAssetStore | None = None,
    dataset_library: SqliteDatasetLibrary | None = None,
    dataset_tasks: SqliteDatasetTasks | None = None,
    dataset_previews: SqliteDatasetPreviews | None = None,
    dataset_gateway: DatasetTaskGateway | None = None,
    graph_registry: NodeRegistry | None = None,
    graph_runtime: GraphRuntime | None = None,
    graph_executions: GraphExecutionRepository | None = None,
    graph_library: GraphLibrary | None = None,
    graph_supervisor: GraphExecutionSupervisor | None = None,
    monitor_bus: MonitorBusPort | None = None,
) -> ApplicationServices:
    """Wire the use cases against fakes (the composition root's twin).

    The runs domain uses fakes (scriptable lifecycle); the config /
    settings / assets / datasets domains default to the *real* adapters
    over temp locations -- they are cheap, and exercising the real TOML /
    SQLite / filesystem code paths is the point of these tests. The
    dataset *gateway* is fake by default: spawning a real child that
    imports torch is nobody's unit test. The graph domain defaults to
    the *fixture* node classes (a NodeRegistry that skips the nodes/
    walk) with the real SQLite execution/library tables and a no-op
    memory releaser -- validation and execution run for real, the GPU
    never does.
    """
    runs = runs if runs is not None else InMemoryRunRepository()
    events = events if events is not None else RecordingEventBus()
    publisher = EventPublisher(events=events)
    paths = ProjectPaths(root=project_root)
    gateway = gateway if gateway is not None else FakeTrainingGateway()
    inspector = inspector if inspector is not None else FakeConfigInspector()
    clock = clock if clock is not None else FakeClock()
    project_root = project_root if project_root is not None else Path(
        tempfile.mkdtemp(prefix="backend-project-")
    )
    runs_dir = runs_dir if runs_dir is not None else Path(
        tempfile.mkdtemp(prefix="backend-runs-")
    )
    if settings_store is None:
        database = SqliteDatabase(project_root / "test-settings.db")
        database.initialize()
        settings_store = SqliteSettingsStore(database, project_root)
    layout = WorkspaceLayout(
        project_root, runs_dir=runs_dir, settings_kv=settings_store.get
    )
    if artifacts is None:
        artifacts = DirectoryRunArtifacts(layout)
    if dataset_library is None:
        dataset_library = SqliteDatasetLibrary(layout)
    if dataset_tasks is None:
        tasks_db = SqliteDatabase(project_root / "test-dataset-tasks.db")
        tasks_db.initialize()
        dataset_tasks = SqliteDatasetTasks(tasks_db, clock)
    if dataset_gateway is None:
        dataset_gateway = FakeDatasetTaskGateway()
        task_sweeper = DatasetTaskSweeper(
            tasks=dataset_tasks, gateway=dataset_gateway, clock=clock
        )
    if dataset_previews is None:
        previews_db = SqliteDatabase(project_root / "test-dataset-previews.db")
        previews_db.initialize()
        dataset_previews = SqliteDatasetPreviews(previews_db, dataset_library)
    dataset_files = FsDatasetFiles(layout.datasets_dir)
    if assets is None:
        assets = FileSystemAssetStore(layout, datasets=dataset_library)
    if config_files is None:
        config_files = CoreConfigFiles()
    if config_options is None:
        config_options = PydanticConfigOptions()
    if graph_registry is None:
        graph_registry = fixture_graph_registry()
    graph_catalog = DiscoveredGraphCatalog(graph_registry)
    # Default to the real adapter (replay/clear semantics under test);
    # the API tests exercise it, the runtime test swaps in a recorder.
    monitor_bus = monitor_bus if monitor_bus is not None else SharedMonitorBus()
    if graph_runtime is None:
        graph_runtime = ReflectedGraphRuntime(
            graph_registry, memory_releaser=lambda: None, monitor_bus=monitor_bus
        )
    if graph_executions is None or graph_library is None:
        graphs_db = SqliteDatabase(project_root / "test-graphs.db")
        graphs_db.initialize()
        if graph_executions is None:
            graph_executions = SqliteGraphExecutionRepository(graphs_db)
        if graph_library is None:
            graph_library = SqliteGraphLibrary(graphs_db)
    execution_writer = ExecutionLifecycleWriter(
        repository=graph_executions, events=publisher
    )
    run_writer = RunLifecycleWriter(repository=runs, events=publisher)
    if graph_supervisor is None:
        graph_supervisor = GraphExecutionSupervisor(
            executions=graph_executions,
            writer=execution_writer,
            runtime=graph_runtime,
            events=publisher,
            clock=clock,
        )
    if supervisor is None:
        supervisor = RunSupervisor(
            runs=runs,
            writer=run_writer,
            events=publisher,
            gateway=gateway,
            progress=_Jsonl(),
            artifacts=artifacts,
            clock=clock,
            poll_interval=poll_interval,
        )
    return ApplicationServices(
        list_runs=ListRuns(runs),
        get_run=GetRun(runs),
        delete_runs=DeleteRuns(runs, events=publisher),
        get_active_run=GetActiveRun(runs),
        start_training=StartTraining(
            runs=runs,
            writer=run_writer,
            gateway=gateway,
            inspector=inspector,
            artifacts=artifacts,
            watcher=supervisor,
            clock=clock,
            paths=paths,
        ),
        stop_training=StopTraining(
            runs=runs, writer=run_writer, gateway=gateway, clock=clock
        ),
        get_run_log=GetRunLog(runs=runs, artifacts=artifacts),
        reconcile_runs=ReconcileRuns(
            runs=runs, writer=run_writer, gateway=gateway, clock=clock,
            watcher=supervisor, artifacts=artifacts,
        ),
        config=ConfigServices(
            read=GetConfig(files=config_files, paths=paths),
            update=UpdateConfig(files=config_files, paths=paths),
            read_raw=ReadConfigRaw(files=config_files, paths=paths),
            write_raw=WriteConfigRaw(files=config_files, paths=paths),
            options=GetConfigOptions(options=config_options),
            start_options=GetStartOptions(
                inspector=inspector, runs=runs, paths=paths
            ),
        ),
        settings=SettingsServices(
            read=GetSettings(settings=settings_store),
            update=UpdateSettings(settings=settings_store),
        ),
        assets=AssetServices(
            list=ListAssets(assets=assets),
            browse=BrowseAssets(assets=assets),
            make_folder=MakeAssetFolder(assets=assets),
            upload=UploadAsset(assets=assets),
            inspect=InspectAsset(assets=assets),
        ),
        datasets=DatasetServices(
            list=ListDatasets(library=dataset_library, previews=dataset_previews),
            get=GetDataset(
                library=dataset_library,
                tasks=dataset_tasks,
                previews=dataset_previews,
            ),
            create=CreateDataset(library=dataset_library),
            delete=DeleteDataset(
                library=dataset_library,
                tasks=dataset_tasks,
                previews=dataset_previews,
            ),
            items=ListDatasetItems(library=dataset_library),
            update_item=UpdateDatasetItem(library=dataset_library),
            bulk_update=BulkUpdateDatasetItems(library=dataset_library),
            discard=DiscardDatasetItems(library=dataset_library),
            sets=ListDatasetSets(library=dataset_library),
            commit=CommitDatasetItems(library=dataset_library),
            tasks=ListDatasetTasks(
                library=dataset_library,
                tasks=dataset_tasks,
            ),
            start_task=StartDatasetTask(
                library=dataset_library,
                tasks=dataset_tasks,
                gateway=dataset_gateway,
                checkpoints_dir=layout.checkpoints_dir,
                sweeper=task_sweeper,
            ),
            stop_task=StopDatasetTask(tasks=dataset_tasks, gateway=dataset_gateway),
            reconcile_tasks=ReconcileDatasetTasks(sweeper=task_sweeper),
            read_file=ReadDatasetFile(files=dataset_files),
            set_preview=SetDatasetPreview(
                library=dataset_library, previews=dataset_previews
            ),
        ),
        graphs=GraphServices(
            catalog=ListNodeCatalog(catalog=graph_catalog),
            diagnostics=NodeDiagnostics(catalog=graph_catalog),
            validate=ValidateGraph(runtime=graph_runtime),
            start_execution=StartGraphExecution(
                executions=graph_executions,
                writer=execution_writer,
                runtime=graph_runtime,
                launcher=graph_supervisor,
                clock=clock,
            ),
            list_executions=ListGraphExecutions(executions=graph_executions),
            get_execution=GetGraphExecution(executions=graph_executions),
            stop_execution=StopGraphExecution(
                executions=graph_executions,
                writer=execution_writer,
                launcher=graph_supervisor,
                clock=clock,
            ),
            delete_executions=DeleteGraphExecutions(
                executions=graph_executions, events=publisher
            ),
            reconcile_executions=ReconcileGraphExecutions(
                executions=graph_executions,
                writer=execution_writer,
                clock=clock,
            ),
            save_graph=SaveGraph(library=graph_library),
            get_graph=GetGraph(library=graph_library),
            list_graphs=ListGraphs(library=graph_library),
            delete_graph=DeleteGraph(library=graph_library),
        ),
        events=publisher,
        event_bus=events,
        monitor=MonitorServices(subscribe=SubscribeMonitor(bus=monitor_bus)),
    )


def seed_run(
    repo: RunRepository,
    clock: FakeClock,
    *,
    config_path: str = "configs/test.toml",
    mode: str = "distillation",
    total_steps: int = 100,
    start: bool = False,
    pid: int | None = 111,
) -> Run:
    """Create + persist a run; optionally transition it to ``running``."""
    run = Run.create(
        config_path=config_path,
        mode=mode,
        total_steps=total_steps,
        created_at=clock.now(),
    )
    repo.add(run)
    if start:
        run.mark_started(pid=pid, at=clock.now())
        repo.update(run)
    return run


# --------------------------------------------------------------------------
# Dataset fixtures -- raw format-v2 / v1 dirs written with plain sqlite3
# --------------------------------------------------------------------------

# Mirrors manager/db.init_local_db's v2 schema verbatim (documented
# duplicate: these fixtures must not import manager -- that pulls torch).
_V2_SCHEMA = """
    CREATE TABLE IF NOT EXISTS info (
        name        TEXT PRIMARY KEY,
        description TEXT,
        created_at  REAL NOT NULL
    );
    CREATE TABLE IF NOT EXISTS sources (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        name        TEXT    NOT NULL,
        type        TEXT    NOT NULL,
        model_path  TEXT,
        config      TEXT,
        created_at  REAL    NOT NULL
    );
    CREATE TABLE IF NOT EXISTS shards (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        file_path    TEXT    NOT NULL UNIQUE,
        layout       TEXT    NOT NULL DEFAULT 'single_latent',
        sample_count INTEGER NOT NULL DEFAULT 0,
        size_bytes   INTEGER NOT NULL DEFAULT 0,
        created_at   REAL    NOT NULL
    );
    CREATE TABLE IF NOT EXISTS trajectories (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        source_id    INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
        shard_id     INTEGER NOT NULL REFERENCES shards(id) ON DELETE CASCADE,
        shard_index  INTEGER NOT NULL,
        sample_count INTEGER NOT NULL DEFAULT 0,
        seed         INTEGER,
        prompt       TEXT,
        neg_prompt   TEXT NOT NULL DEFAULT '',
        model_type   TEXT NOT NULL DEFAULT 'eps',
        type         TEXT NOT NULL DEFAULT 'good',
        cfg          REAL,
        source_path  TEXT,
        latent_h     INTEGER NOT NULL DEFAULT 0,
        latent_w     INTEGER NOT NULL DEFAULT 0,
        preview_path TEXT,
        extra        TEXT
    );
    CREATE TABLE IF NOT EXISTS training_sets (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        name        TEXT    NOT NULL UNIQUE,
        description TEXT,
        created_at  REAL    NOT NULL
    );
    CREATE TABLE IF NOT EXISTS set_members (
        set_id        INTEGER NOT NULL REFERENCES training_sets(id) ON DELETE CASCADE,
        trajectory_id INTEGER NOT NULL REFERENCES trajectories(id) ON DELETE CASCADE,
        PRIMARY KEY (set_id, trajectory_id)
    );
"""

_EPOCH = 1767225600.0  # 2026-01-01T00:00:00Z


def make_v2_dataset(
    project_root: Path,
    name: str,
    *,
    items: int = 4,
    shard_file: bool = True,
    previews: bool = True,
) -> Path:
    """A format-v2 dataset dir: 1 source, 1 shard, ``items`` rows.

    Item i (1-based) gets prompt ``"photo i"``; the last row is type
    ``bad``; row 1 optionally carries a real preview file. Returns the
    dataset directory (``datasets/<name>`` under ``project_root``).
    """
    import sqlite3  # local: keeps this module's import surface light

    directory = project_root / "datasets" / name
    (directory / "shards").mkdir(parents=True, exist_ok=True)
    (directory / "previews").mkdir(exist_ok=True)
    shard_rel = "shards/x0_0.safetensors"
    if shard_file:
        (directory / shard_rel).write_bytes(b"\x00" * 16)
    if previews:
        (directory / "previews" / "p1.png").write_bytes(b"\x89PNG")

    conn = sqlite3.connect(str(directory / "metadata.db"))
    try:
        conn.executescript(_V2_SCHEMA)
        conn.execute(
            "INSERT INTO info (name, description, created_at) VALUES (?, ?, ?)",
            (name, "fixture", _EPOCH),
        )
        conn.execute(
            "INSERT INTO sources (name, type, model_path, config, created_at) "
            "VALUES (?, 'real', ?, NULL, ?)",
            (f"{name}_lora", "ckpt/model.safetensors", _EPOCH),
        )
        conn.execute(
            "INSERT INTO shards (file_path, layout, sample_count, size_bytes, "
            "created_at) VALUES (?, 'single_latent', ?, 1024, ?)",
            (shard_rel, items, _EPOCH),
        )
        for i in range(1, items + 1):
            last = i == items
            conn.execute(
                "INSERT INTO trajectories (source_id, shard_id, shard_index, "
                "sample_count, seed, prompt, neg_prompt, model_type, type, cfg, "
                "source_path, latent_h, latent_w, preview_path, extra) "
                "VALUES (1, 1, ?, 1, ?, ?, '', 'eps', ?, 5.0, ?, 64, 64, ?, NULL)",
                (
                    i - 1,
                    100 + i,
                    f"photo {i}",
                    "bad" if last else "good",
                    f"img_{i}.png",
                    "previews/p1.png" if (previews and i == 1) else None,
                ),
            )
        conn.execute(f"PRAGMA user_version = 2")
        conn.commit()
    finally:
        conn.close()
    return directory


def make_v1_dataset(project_root: Path, name: str) -> Path:
    """A legacy (pre-v2) dataset dir: ``info`` table only, version 0 --
    enough for identity reads; every v2-only operation must refuse it
    *before* touching columns this fixture doesn't have."""
    import sqlite3  # local: keeps this module's import surface light

    directory = project_root / "datasets" / name
    directory.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(directory / "metadata.db"))
    try:
        conn.execute(
            "CREATE TABLE info (name TEXT PRIMARY KEY, description TEXT, "
            "created_at REAL NOT NULL)"
        )
        conn.execute(
            "INSERT INTO info (name, description, created_at) VALUES (?, ?, ?)",
            (name, "legacy fixture", _EPOCH),
        )
        conn.commit()  # user_version stays 0 (legacy never set it to 1 either)
    finally:
        conn.close()
    return directory


# --------------------------------------------------------------------------
# Raw-ASGI client (no httpx dependency, like the legacy smoke tests)
# --------------------------------------------------------------------------


async def _asgi_call(
    app,
    path: str,
    *,
    method: str = "GET",
    json_body=None,
    body_bytes: bytes | None = None,
    content_type: str | None = None,
    extra_headers: dict | None = None,
):
    method = method or "GET"
    raw_path, _, query = path.partition("?")
    start: dict = {}
    chunks: list[bytes] = []
    body = b""
    # caller-supplied headers win over the auto ones (e.g. a declared
    # content-length that differs from the body, for cap checks)
    extra = {
        str(k).lower().encode(): str(v).encode()
        for k, v in (extra_headers or {}).items()
    }
    header_map: dict[bytes, bytes] = {b"host": b"localhost"}
    if body_bytes is not None:
        body = body_bytes
        header_map[b"content-type"] = (
            content_type or "application/octet-stream"
        ).encode()
        header_map.setdefault(b"content-length", str(len(body)).encode())
    elif json_body is not None:
        body = json.dumps(json_body).encode("utf-8")
        header_map[b"content-type"] = b"application/json"
        header_map.setdefault(b"content-length", str(len(body)).encode())
    # Extra headers replace the default of the same name -- a test that
    # sends a foreign `host` must end up with exactly one Host header,
    # not two (the first would win on the wire).
    header_map.update(extra)
    headers = list(header_map.items())

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        if message["type"] == "http.response.start":
            start.update(message)
        elif message["type"] == "http.response.body":
            chunks.append(message.get("body", b""))

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        # ASGI spec: scope["path"] is the *decoded* path (what uvicorn
        # hands the app); raw_path keeps the original bytes.
        "path": unquote(raw_path),
        "raw_path": raw_path.encode(),
        "query_string": query.encode(),
        "root_path": "",
        "headers": headers,
        "client": ("1.2.3.4", 1234),
        "server": ("localhost", 8766),
    }
    await app(scope, receive, send)
    resp_headers = {k.decode().lower(): v.decode() for k, v in start.get("headers", [])}
    return start.get("status"), resp_headers, b"".join(chunks)


def asgi_request(
    app,
    path: str,
    *,
    method: str = "GET",
    json_body=None,
    body_bytes: bytes | None = None,
    content_type: str | None = None,
    extra_headers: dict | None = None,
    strict_json: bool = False,
) -> tuple[int, dict, object]:
    """One request through the whole app; returns (status, headers, body).

    ``body`` is parsed JSON when the response is JSON, else the text,
    else ``None``. ``strict_json`` parses with ``parse_constant``
    rejecting NaN/Infinity -- the standard JSON forbids them, so a
    response that carries one comes back as its raw text and fails the
    assertion that expected an object (docs 07 F-03).
    """

    def reject(constant: str):
        raise ValueError(f"invalid JSON constant {constant}")

    status, headers, raw = asyncio.run(
        _asgi_call(
            app,
            path,
            method=method,
            json_body=json_body,
            body_bytes=body_bytes,
            content_type=content_type,
            extra_headers=extra_headers,
        )
    )
    text = raw.decode("utf-8", errors="replace")
    body: object = None
    if text:
        try:
            body = json.loads(
                text, parse_constant=reject if strict_json else None
            )
        except ValueError:  # JSONDecodeError, or a rejected NaN/Infinity
            body = text
    return status or 0, headers, body
