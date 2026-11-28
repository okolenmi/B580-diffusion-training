"""Shared test support: assertions, fakes, and a raw-ASGI client.

House style (matches the repo's smoke tests): each ``test_*.py`` file
runs standalone -- ``check()`` prints PASS/FAIL per assertion, and
``finish()`` exits non-zero if anything failed. No pytest dependency.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import sys
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, UTC
from pathlib import Path
from typing import Any
from urllib.parse import unquote

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.application.ports.environment import DeviceProbe, DeviceReport
from backend.application.ports.graph_execution_repository import (
    GraphExecutionRepository,
)
from backend.application.ports.graph_library import GraphLibrary
from backend.application.ports.host_memory import HostMemory
from backend.application.ports.graph_runtime import GraphRuntime
from backend.application.ports.monitor_bus import MonitorBus as MonitorBusPort
from backend.application.memory_admission import (
    LedgerProvider,
)
from backend.application.memory_ledger import (
    DEFAULT_FOREIGN_RESERVE_MB,
    DEFAULT_PROCESS_OVERHEAD_MB,
)
from backend.application.graph_peak_source import GraphPeakSource
from backend.application.graph_supervisor import GraphExecutionSupervisor
from backend.application.event_publisher import EventPublisher
from backend.application.lifecycle_writer import (
    ExecutionLifecycleWriter,
)
from backend.application.services import (
    ApplicationServices,
    AssetServices,
    GraphServices,
    InstallerServices,
    MonitorServices,
    SettingsServices,
)
from backend.application.ports.tokenizer_files import TokenizerFiles, set_by_key
from backend.application.ports.venv_inspection import VenvDevice, VenvInspector, VenvInventory
from backend.application.ports.package_installer import (
    InstallError,
    PackageInstaller,
)
from backend.application.use_cases.install_packages import (
    GetInstall,
    StartInstall,
)
from backend.application.use_cases import (
    ApplyInstallation,
    BrowseAssets,
    InspectEnvironment,
    TokenizerSetup,
    DeleteGraph,
    DeleteGraphExecutions,
    DescribeRequirements,
    GetGraph,
    GetExecutionLog,
    GetExecutionMetrics,
    GetGraphExecution,
    GetSettings,
    InspectAsset,
    ListAssets,
    ListGraphExecutions,
    ListGraphs,
    MakeAssetFolder,
    GetNodeCatalog,
    NodeDiagnostics,
    ListImageLibraries,
    ListModelFiles,
    ResolveNode,
    PreviewGraphMemory,
    ReconcileGraphExecutions,
    SaveGraph,
    StartGraphExecution,
    RenameGraphExecution,
    SendExecutionAction,
    StopGraphExecution,
    SubscribeMonitor,
    SweepExecutionScratch,
    UpdateSettings,
    UploadAsset,
    ValidateGraph,
)
from backend.domain.events import DomainEvent
from backend.application.journal_listener import JournalListener
from backend.application.services import JournalServices
from backend.application.use_cases.journal import ReadJournal, WriteJournal
from backend.infrastructure.events.callback_event_bus import CallbackEventBus
from backend.infrastructure.file_journal import FileJournal
from backend.infrastructure.file_asset_store import FileSystemAssetStore
from backend.infrastructure.graph.contract_catalog import ContractNodeCatalog
from backend.infrastructure.graph.next_starters import NextStarters
from backend.application.use_cases.starters import BuildStarter, ListStarters
from backend.infrastructure.graph.memory_fields import memory_fields_resolver
from backend.domain.graph import GraphDefinition
from backend.infrastructure.graph.runtime import ContractGraphRuntime
from backend.infrastructure.graph_event_stream import ExecutionEventTail
from backend.infrastructure.execution_logs import FileExecutionLogs
from backend.infrastructure.execution_metrics import FileExecutionMetrics
from backend.infrastructure.graph_task_gateway import InProcessGraphTaskGateway
from backend.infrastructure.memory_peak_store import SqlitePeakStore
from backend.infrastructure.monitor_bus import SharedMonitorBus
from backend.infrastructure.persistence.graph_execution_repository import (
    SqliteGraphExecutionRepository,
)
from backend.infrastructure.persistence.graph_library import SqliteGraphLibrary
from backend.infrastructure.file_artifact_store import FileArtifactStore
from backend.infrastructure.file_command_channel import FileCommandChannel
from backend.infrastructure.persistence.app_library import SqliteAppLibrary
from backend.application.services import AppServices, LibraryServices
from backend.application.use_cases.image_libraries import (
    AddToSet, CreateLibrary, DeleteCache, DeleteLibrary, DeleteSet, EditCaptions, LibraryReport, ListImageIds, ListImages, ListImports, ListLibraries, ReadImage,
    RemoveImages, StartImport,
)
from backend.infrastructure.image_libraries import FsImageLibraries
from backend.application.use_cases.graph_modules import DescribeGraphModules, ResolveBoard
from backend.application.use_cases.run_app import RunApp
from backend.application.use_cases.app_crud import (
    DeleteApp,
    GetApp,
    ListApps,
    SaveApp,
    SaveAppGraph,
    SaveAppValues,
)
from backend.infrastructure.persistence.sqlite import SqliteDatabase
from backend.infrastructure.settings_store import SqliteSettingsStore
from backend.infrastructure.workspace import WorkspaceLayout

FAILURES: list[str] = []
#: Every message passed to `check`, in order. Counted so `finish` can refuse
#: to report success for a file that ran nothing -- see there.
CHECKED: list[str] = []


def check(condition: bool, message: str) -> None:
    print(f"  {'PASS' if condition else 'FAIL'}: {message}")
    CHECKED.append(message)
    if not condition:
        FAILURES.append(message)


def finish() -> None:
    """Report, and refuse to report success for a file that ran nothing.

    Zero checks is a failure, not a pass. This is not hypothetical: two
    test files in this repository once ran none of their tests and the
    suite stayed green, because `finish` only ever looked at FAILURES. A
    restructure that leaves the last call to `finish()` above the code --
    or a `main()` that never calls the functions it defines -- produces
    exactly that, and nothing downstream can tell it from a real pass.

    `python -c "from backend.tests.support import finish; finish()"` used
    to print ALL CHECKS PASSED and exit 0.
    """
    print()
    print("=" * 60)
    if not CHECKED:
        print("SMOKE TEST: NO CHECKS RAN -- treating that as a failure")
        print("  A test file that runs nothing passes vacuously, and this")
        print("  repository has been bitten by that twice.")
        sys.exit(1)
    if FAILURES:
        print(f"SMOKE TEST: {len(FAILURES)} of {len(CHECKED)} CHECK(S) FAILED")
        for failure in FAILURES:
            print(f"  - {failure}")
        sys.exit(1)
    print(f"SMOKE TEST: ALL {len(CHECKED)} CHECKS PASSED")


def wait_until(predicate, *, timeout: float = 2.0, interval: float = 0.01) -> bool:
    """Poll ``predicate`` until true or timeout; for supervisor threads."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


class _ThreadRoutedStdout:
    """``sys.stdout`` that sends each thread's output to its own buffer.

    Not ``contextlib.redirect_stdout``, which swaps ``sys.stdout`` for the
    whole process: several tests entering it concurrently tangle its
    stack, so some threads' output lands in a buffer nobody prints and
    the log silently loses most of the file. Routing on a thread-local
    instead has no shared state to get wrong.

    ``write`` returns the character count because ``print`` inspects it; a
    ``None`` here truncates the output rather than merely reordering it.
    """

    def __init__(self, real) -> None:
        self._real = real
        self._local = threading.local()
        self._saved = None

    def __enter__(self):
        self._saved = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, *_exc) -> None:
        sys.stdout = self._saved

    def bind(self, buffer) -> None:
        self._local.buffer = buffer

    def unbind(self) -> None:
        self._local.buffer = None

    def write(self, text: str) -> int:
        target = getattr(self._local, "buffer", None)
        if target is None:
            return self._real.write(text)
        target.write(text)
        return len(text)

    def flush(self) -> None:
        target = getattr(self._local, "buffer", None)
        (target if target is not None else self._real).flush()


#: The model subdirectories `path_tiers` and `paths` resolve inside a
#: ComfyUI tree. Only `checkpoints` and `loras` are reached by the backend's
#: own resolution; the rest are here so a test that wanders further finds a
#: directory rather than a FileNotFoundError, which is a much worse way to
#: learn that a fixture was incomplete.
_COMFY_MODEL_DIRS = (
    "checkpoints", "loras", "diffusion_models", "vae", "clip", "unet",
    "text_encoders", "upscale_models", "controlnet",
)


class ThisInterpreterLayout(WorkspaceLayout):
    """A layout whose child processes run under the interpreter running the
    tests.

    `WorkspaceLayout.venv_python` is whatever the machine is configured with
    (`VENV_PYTHON`, a sibling ../venv, a setting, then plain "python"). The
    graph-child tests spawn real children through it, so on a machine
    without that configuration -- a fresh clone, CI, a sandbox -- the child
    ran under a different Python than the test and refused to start ("needs
    Python 3.14 or newer; this is 3.13"). The tests then failed on the
    *child's* absence rather than on anything they check, and the whole
    end-to-end child path went unexercised there.

    A subclass, not `VENV_PYTHON` in the environment: the environment is
    process-global and the files here run tests concurrently (see the
    comment in test_graph_task_gateway on the flake that caused)."""

    @property
    def venv_python(self) -> str:
        return sys.executable


def use_temporary_comfy_dir(prefix: str = "backend-comfy-") -> Path:
    """Point this process, and any child it spawns, at a throwaway ComfyUI.

    Round-3 N3-05 found the suite was not hermetic. On a checkout with no
    `COMFY_DIR` -- and therefore no `.env` either, since both are how a
    developer points at their own ComfyUI -- two files died rather than
    failed:

        RuntimeError: Cannot find ComfyUI directory
        -> GraphLaunchError

    Reproduced on a `git archive` of HEAD, which is what a fresh clone looks
    like: `test_graph_task_gateway` and `test_graph_adoption` both exit 1.
    Both spawn a *real* child, and the child is the thing that needs the
    directory. With `COMFY_DIR` set they pass, which is why this went
    unnoticed: the developer's machine is configured.

    (The review also named `test_process_identity`. It does not fail on a
    bare checkout -- it spawns `/bin/sleep` and `/bin/true`, nothing that
    resolves a ComfyUI path. Two of the three reproduce, not three.)

    Both halves of the override are set, deliberately:

    * `paths.set_comfy_dir` sets the explicit override, which
      `get_comfy_dir()` consults *before* the environment, so nothing a
      developer's `.env` says can win;
    * `os.environ["COMFY_DIR"]` is set as well, because that is what a child
      process inherits. `paths._load_dotenv` uses `setdefault`, so an
      already-set variable is not overwritten -- but relying on import order
      would be a race with the child's own startup.

    Returns the directory, so a test that needs to write a fixture *into* it
    (a checkpoint, say) has somewhere to put it that is not the developer's.
    """
    root = Path(tempfile.mkdtemp(prefix=prefix))
    for name in _COMFY_MODEL_DIRS:
        (root / "models" / name).mkdir(parents=True, exist_ok=True)
    for name in ("custom_nodes", "input", "output", "temp"):
        (root / name).mkdir(parents=True, exist_ok=True)

    os.environ["COMFY_DIR"] = str(root)
    try:
        import paths  # noqa: PLC0415 -- explicit repo bridge, as path_tiers does
    except ImportError:  # pragma: no cover -- only if the repo is not importable
        return root
    paths.set_comfy_dir(root)
    return root


#: The hermetic ComfyUI this process is pointed at, or None when the real
#: one is in use. A test that needs the path asks for it; a test that needs
#: the *real* one asserts this is None first, so that asking for the real
#: thing is a visible decision rather than an accident.
TEMP_COMFY_DIR: Path | None = None

if os.environ.get("BACKEND_TESTS_REAL_COMFY") != "1":
    # **At import time, and that is the whole point.** This used to be
    # opt-in: two of thirty-one files called it, and every other file passed
    # only because the developer's machine has a `.env` naming their
    # ComfyUI. Round-4 R4-03 is that a test suite should not depend on the
    # machine it runs on, and the cheapest way to get that is for there to
    # be no way to forget.
    #
    # It is also what `test_config.py`, `test_installer.py` and
    # `test_settings.py` were silently asserting: they passed here and
    # failed on a fresh clone, which is the same class of bug as a test that
    # passes on the author's box.
    #
    # `BACKEND_TESTS_REAL_COMFY=1` opts out, for the times a test genuinely
    # needs the real thing -- and it has to be asked for by name, so it
    # cannot happen by omission.
    TEMP_COMFY_DIR = use_temporary_comfy_dir(prefix="backend-suite-comfy-")


def run_tests_concurrently(tests) -> None:
    """Run ``tests`` concurrently and print each one's output in order.

    For tests that are independent but each cost real wall time -- a
    child process starting, a supervisor thread reaching a state. In
    sequence they add up; together they overlap.

    Output is buffered per test and printed in the original order, so a
    failure still reads as a sequence rather than as whichever line won
    the race. Then ``finish()``, which turns accumulated failures into an
    exit code.
    """
    results: dict = {}
    with _ThreadRoutedStdout(sys.stdout) as router:
        with ThreadPoolExecutor(max_workers=len(tests)) as pool:
            futures = {pool.submit(_run_capturing, router, t): t for t in tests}
            for future in as_completed(futures):
                test = futures[future]
                output, error = future.result()
                if error is not None:
                    # Recorded as a failure, not just printed. A test that
                    # raises used to leave the run green, because the
                    # exception landed in a buffer and no check() ever
                    # ran -- which is the worst possible failure mode for
                    # a test suite: it looks like it passed.
                    FAILURES.append(f"{test.__name__} raised")
                    output += f"\n  !! {test.__name__} raised\n{error}"
                results[test.__name__] = output
    for test in tests:
        print(results[test.__name__], end="")
    finish()


def _run_capturing(router: _ThreadRoutedStdout, test):
    """Run one test, returning its output instead of printing it live."""
    buffer = io.StringIO()
    error = None
    router.bind(buffer)
    try:
        test()
    except Exception as exc:  # noqa: BLE001 -- reported, not swallowed
        error = "".join(traceback.format_exception(exc))
    finally:
        router.unbind()
    return buffer.getvalue(), error


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class FakeClock:
    """Deterministic clock; starts at a fixed instant and only moves
    when a test says so."""

    def __init__(self, start: datetime | None = None) -> None:
        self._now = start or datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)

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


def fixture_tree_registry():
    """The registry of the new node tree over ``fixture_nodes`` (what validation and runs use)."""
    from backend.infrastructure.graph import tree

    return tree.NodeRegistry(packages=("backend.tests.fixture_nodes",))


def uid(name: str) -> str:
    """A valid node uid for a short test name: stable, so a test can name a node by ``"a"`` and read it back."""
    import base64
    import hashlib

    return "n_" + base64.b32encode(hashlib.sha1(name.encode()).digest()[:4]).decode().lower()[:6]


def gnode(name: str, class_name: str, **params) -> dict:
    return {"uid": uid(name), "class_name": class_name, "params": params}


def gedge(src: str, out: str, dst: str, inp: str) -> dict:
    return {"from_node": uid(src), "from_port": out, "to_node": uid(dst), "to_port": inp}


def gdoc(nodes=(), edges=(), settings: dict | None = None, **rest) -> dict:
    """A graph document (docs/contracts/graph-json.md) from compact pieces."""
    return {"format": 3, "settings": dict(settings or {}), "nodes": list(nodes), "edges": list(edges), **rest}


def gdef(nodes=(), edges=(), settings: dict | None = None, **rest) -> GraphDefinition:
    return GraphDefinition.from_dict(gdoc(nodes, edges, settings, **rest))


class RefusingPackageInstaller(PackageInstaller):
    """An installer that refuses, for suites that are not about installing.

    A default that *worked* would mean any test that reached the
    install endpoint for an unrelated reason would run real pip against a
    real interpreter. This one refuses, so a test that means to install has
    to say so by passing its own.

    It records the request, so a test that *wants* to assert what would
    have been installed can still do that without anything running.
    """

    def __init__(self) -> None:
        self.requests: list = []

    def install(self, request, on_line=None):
        self.requests.append(request)
        raise InstallError(
            "no installer is wired in this test; pass a PackageInstaller if "
            "the test is about installing"
        )


class FakeVenvInspector(VenvInspector):
    """Environments held in memory: ``{folder: VenvInventory}``. Anything not listed has no interpreter.

    The real inspector runs the environment's Python, which would make a test a statement about the machine it runs on.
    """

    def __init__(self, environments: dict[str, VenvInventory] | None = None, device: VenvDevice | None = None) -> None:
        self.environments = environments or {}
        self.device_result = device or VenvDevice(True, "Fake Arc", 12288, "2.9.0+xpu")

    def inspect(self, venv_dir: str) -> VenvInventory:
        return self.environments.get(venv_dir, VenvInventory(None, error=f"no Python found in {venv_dir}"))

    def device(self, venv_dir: str) -> VenvDevice:
        return self.device_result


class FakeTokenizerFiles(TokenizerFiles):
    """Tokenizer files held in memory; ``download`` marks a set complete without touching the network."""

    def __init__(self) -> None:
        self.have: set[str] = set()
        self.downloads: list[str] = []

    def present(self, key: str) -> tuple[str, ...]:
        return tuple(set_by_key(key).files) if key in self.have else ()

    def download(self, key: str) -> tuple[str, ...]:
        self.downloads.append(key)
        self.have.add(key)
        return tuple(set_by_key(key).files)


class FakeDeviceProbe(DeviceProbe):
    """A device that is always there, for tests that are not about it.

    The real probe spawns a subprocess that imports torch: correct, and
    about 1.7 s per call on this machine. No unit test should pay that, and
    a test suite that did would be reporting on how loaded the box is. The
    "device present" default matches this project's only supported card
    (ADR 0004), so a test that forgets to stub it is not reporting a
    surprise.
    """

    def __init__(self, report: DeviceReport | None = None) -> None:
        self._report = report or DeviceReport(
            present=True, backend="xpu", name="Intel(R) Arc(TM) B580 Graphics",
            total_memory_mb=12216,
        )
        self.calls = 0

    def report(self) -> DeviceReport:
        self.calls += 1
        return self._report


def build_services(
    *,
    events: RecordingEventBus | None = None,
    clock: FakeClock | None = None,
    project_root: Path | None = None,
    runs_dir: Path | None = None,
    poll_interval: float = 0.05,
    settings_store: SqliteSettingsStore | None = None,
    assets: FileSystemAssetStore | None = None,
    graph_registry: Any | None = None,
    graph_runtime: GraphRuntime | None = None,
    real_runs: bool = False,
    graph_executions: GraphExecutionRepository | None = None,
    graph_library: GraphLibrary | None = None,
    graph_supervisor: GraphExecutionSupervisor | None = None,
    monitor_bus: MonitorBusPort | None = None,
    device_probe: DeviceProbe | None = None,
    venv_inspector: VenvInspector | None = None,
    tokenizer_files: TokenizerFiles | None = None,
    host_memory: HostMemory | None = None,
) -> ApplicationServices:
    """Wire the use cases against fakes (the composition root's twin).

    The config / settings / assets / graph domains default to
    the *real* adapters
    over temp locations -- they are cheap, and exercising the real TOML /
    SQLite / filesystem code paths is the point of these tests. The
    graph domain defaults to
    the *fixture* node classes (a NodeRegistry that skips the nodes/
    walk) with the real SQLite execution/library tables and a no-op
    memory releaser -- validation and execution run for real, the GPU
    never does.
    """
    events = events if events is not None else RecordingEventBus()
    publisher = EventPublisher(events=events)
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
    image_libraries = FsImageLibraries(layout)
    if assets is None:
        assets = FileSystemAssetStore(layout)
    if graph_registry is None:
        graph_registry = fixture_tree_registry()
    contract_catalog = ContractNodeCatalog(graph_registry)
    # Default to the real adapter (replay/clear semantics under test);
    # the API tests exercise it, the runtime test swaps in a recorder.
    monitor_bus = monitor_bus if monitor_bus is not None else SharedMonitorBus()
    # ``real_runs``: runs get the runtime a child process builds (frames reach the event file, commands reach the nodes) instead of
    # a silent one; a runtime the caller passes is used as given.
    own_runtime = graph_runtime is not None or not real_runs
    if graph_runtime is None:
        graph_runtime = ContractGraphRuntime(graph_registry, releaser=lambda: None)
    # One directory for the supervisor and the sweep, so a test
    # can assert that clearing the history cleared the disk too.
    # Unconditional: a caller that passes only ``graph_executions``
    # still needs the path defined (the old nested assignment left it
    # unbound and the first use raised NameError).
    journal = FileJournal(project_root / "test-journal.jsonl")
    events.subscribe(JournalListener(journal))
    graph_scratch = project_root / "test-graph-scratch"
    execution_metrics = FileExecutionMetrics(graph_scratch)
    # The test twin of bootstrap's peak store (MEM-04 #2): one store per
    # composition, next to the databases this build already owns, shared
    # by admission (reads) and the supervisor (files what a run reports).
    peak_store = SqlitePeakStore(project_root / "test-memory-peaks.db")
    # Fingerprint + remembered peak, built once and shared by the panel
    # preview and the run admission below (see bootstrap: three
    # constructions is three chances to disagree on which configuration
    # a peak belongs to). Fixture nodes declare no memory_fields and
    # fixture graphs name no dataset, so every fingerprint here is
    # unknown -- exactly what these tests claimed before the read half
    # existed, never a zero peak.
    fixture_peak_source = GraphPeakSource(
        peaks=peak_store,
        resolve_memory_fields=memory_fields_resolver(graph_registry),
    )
    if graph_executions is None or graph_library is None:
        graphs_db = SqliteDatabase(project_root / "test-graphs.db")
        graphs_db.initialize()
        if graph_executions is None:
            graph_executions = SqliteGraphExecutionRepository(graphs_db)
        if graph_library is None:
            graph_library = SqliteGraphLibrary(graphs_db)
    shared_probe = device_probe or FakeDeviceProbe()
    # The test twin of bootstrap's ledger wiring (MEM-03): the fake
    # probe always reports 12,216 MB, so with the ADR defaults capacity
    # is 12,216 - 1,024 foreign = 11,192 MB. Built on first use, same
    # as production; first use in a test is normally the first start,
    # which also rebuilds it from unfinished rows.
    memory_ledger = LedgerProvider(
        probe=shared_probe,
        graph_executions=graph_executions,
        foreign_reserve_mb=DEFAULT_FOREIGN_RESERVE_MB,
        process_overhead_mb=DEFAULT_PROCESS_OVERHEAD_MB,
    )
    execution_writer = ExecutionLifecycleWriter(
        clock=clock,
        repository=graph_executions, events=publisher
    )
    if graph_supervisor is None:
        # The in-process gateway, so a test that supplies its own runtime
        # (to count memory releases, or to swap the memory releaser) has
        # that runtime be the one that runs. Same producer as the child
        # gateway -- only where the interpreter state lives differs.
        graph_supervisor = GraphExecutionSupervisor(
            executions=graph_executions,
            writer=execution_writer,
            gateway=InProcessGraphTaskGateway(
                graph_registry,
                runtime_factory=(lambda _writer: graph_runtime) if own_runtime else None,
            ),
            events=publisher,
            clock=clock,
            memory_ledger=memory_ledger,
            peak_store=peak_store,
            monitor_bus=monitor_bus,
            journal=journal,
            scratch_dir=graph_scratch,
            make_tail=ExecutionEventTail,
        )
    # One job dict, so a job created by a test can be polled through the
    # same services object the route reads.
    install_jobs: dict = {}
    refusing_installer = RefusingPackageInstaller()
    inspect_environment = InspectEnvironment(venv_inspector or FakeVenvInspector(), project_root)
    app_db = SqliteDatabase(project_root / "test-apps.db")
    app_db.initialize()
    app_library = SqliteAppLibrary(app_db)
    # One instance for the editor's Run and for running an app, as in
    # bootstrap: the single-active check lives under its lock.
    start_execution = StartGraphExecution(
        executions=graph_executions,
        writer=execution_writer,
        runtime=graph_runtime,
        launcher=graph_supervisor,
        clock=clock,
        memory_ledger=memory_ledger,
        # Fixture nodes declare no memory_fields and fixture graphs name
        # no dataset, so every fingerprint here is unknown -- exactly what
        # these tests claimed before the read half existed, never a zero peak.
        peak_source=fixture_peak_source.observed,
        host_memory=host_memory,
    )
    describe_modules = DescribeGraphModules(catalog=contract_catalog)
    resolve_board = ResolveBoard(describe=describe_modules, runtime=graph_runtime)
    return ApplicationServices(
        installer=InstallerServices(
            device_probe=shared_probe,
            apply=ApplyInstallation(settings=settings_store, project_root=project_root),
            manifest=DescribeRequirements(),
            inspect=inspect_environment,
            # Refusing by default: a default that ran real pip would let any test that reaches the endpoint modify a real environment.
            install=StartInstall(
                installer=refusing_installer,
                inspect=inspect_environment,
                base_python=sys.executable,
                jobs=install_jobs,
            ),
            install_status=GetInstall(jobs=install_jobs),
            tokenizers=TokenizerSetup(tokenizer_files or FakeTokenizerFiles()),
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
        artifacts=FileArtifactStore(project_root / "artifacts"),
        journal=JournalServices(read=ReadJournal(journal), write=WriteJournal(journal)),
        apps=AppServices(
            describe=describe_modules,
            resolve=resolve_board,
            save=SaveApp(library=app_library),
            get=GetApp(library=app_library),
            list=ListApps(library=app_library),
            delete=DeleteApp(library=app_library),
            save_graph=SaveAppGraph(library=app_library, resolve=resolve_board),
            save_values=SaveAppValues(library=app_library),
            run=RunApp(library=app_library, start=start_execution),
        ),
        libraries=LibraryServices(
            list=ListLibraries(image_libraries), create=CreateLibrary(image_libraries), delete=DeleteLibrary(image_libraries),
            images=ListImages(image_libraries), image_ids=ListImageIds(image_libraries), report=LibraryReport(image_libraries), read_image=ReadImage(image_libraries), edit_captions=EditCaptions(image_libraries),
            remove_images=RemoveImages(image_libraries), add_to_set=AddToSet(image_libraries), delete_set=DeleteSet(image_libraries),
            delete_cache=DeleteCache(image_libraries), start_import=StartImport(image_libraries), imports=ListImports(image_libraries),
        ),
        graphs=GraphServices(
            node_catalog=GetNodeCatalog(catalog=contract_catalog),
            resolve_node=ResolveNode(catalog=contract_catalog),
            model_files=ListModelFiles(catalog=contract_catalog),
            image_libraries=ListImageLibraries(catalog=contract_catalog),
            list_starters=ListStarters(starters=NextStarters()),
            build_starter=BuildStarter(starters=NextStarters()),
            preview_memory=PreviewGraphMemory(
                memory_ledger=memory_ledger,
                peak_source=fixture_peak_source.observed,
            ),
            diagnostics=NodeDiagnostics(catalog=contract_catalog),
            validate=ValidateGraph(runtime=graph_runtime),
            start_execution=start_execution,
            list_executions=ListGraphExecutions(executions=graph_executions),
            get_execution=GetGraphExecution(executions=graph_executions),
            execution_metrics=GetExecutionMetrics(executions=graph_executions, metrics=execution_metrics),
            execution_log=GetExecutionLog(executions=graph_executions, logs=FileExecutionLogs(graph_scratch)),
            rename_execution=RenameGraphExecution(executions=graph_executions),
            send_action=SendExecutionAction(
                executions=graph_executions, catalog=contract_catalog, channel=FileCommandChannel(graph_scratch)),
            stop_execution=StopGraphExecution(
                executions=graph_executions,
                writer=execution_writer,
                launcher=graph_supervisor,
                clock=clock,
            ),
            delete_executions=DeleteGraphExecutions(
                executions=graph_executions, events=publisher,
                scratch=SweepExecutionScratch(
                    executions=graph_executions, scratch_dir=graph_scratch, metrics=execution_metrics,
                ),
                memory_ledger=memory_ledger,
            ),
            reconcile_executions=ReconcileGraphExecutions(
                executions=graph_executions,
                writer=execution_writer,
                launcher=graph_supervisor,
                clock=clock,
                memory_ledger=memory_ledger,
            ),
            save_graph=SaveGraph(library=graph_library),
            get_graph=GetGraph(library=graph_library),
            list_graphs=ListGraphs(library=graph_library),
            delete_graph=DeleteGraph(library=graph_library),
        ),
        events=publisher,
        event_bus=events,
        memory_ledger=memory_ledger,
        monitor=MonitorServices(subscribe=SubscribeMonitor(bus=monitor_bus)),
    )



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

    sent_request = False

    async def receive():
        # After the body, a real server blocks here until the client goes away; a response that listens for the disconnect while it
        # streams (newer Starlette) would otherwise spin on an endless supply of requests.
        nonlocal sent_request
        if not sent_request:
            sent_request = True
            return {"type": "http.request", "body": body, "more_body": False}
        await asyncio.Event().wait()

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
