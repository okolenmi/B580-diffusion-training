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

from backend.application.ports.config_inspector import (
    ConfigDescription,
    ConfigInspector,
    ConfigSummary,
    StartOption,
)
from backend.application.ports.environment import (
    DeviceProbe,
    DeviceReport,
    MetadataPackageInventory,
)
from backend.application.ports.graph_execution_repository import (
    GraphExecutionRepository,
)
from backend.application.ports.graph_library import GraphLibrary
from backend.application.ports.graph_runtime import GraphRuntime
from backend.application.ports.monitor_bus import MonitorBus as MonitorBusPort
from backend.application.dataset_task_sweeper import DatasetTaskSweeper
from backend.application.errors import ConfigNotFoundError
from backend.application.memory_admission import (
    LedgerProvider,
    release,
    task_owner,
)
from backend.application.memory_ledger import (
    DEFAULT_FOREIGN_RESERVE_MB,
    DEFAULT_PROCESS_OVERHEAD_MB,
)
from backend.application.project_paths import ProjectPaths
from backend.application.graph_peak_source import GraphPeakSource
from backend.application.graph_supervisor import GraphExecutionSupervisor
from backend.application.event_publisher import EventPublisher
from backend.application.lifecycle_writer import (
    ExecutionLifecycleWriter,
)
from backend.application.services import (
    ApplicationServices,
    AssetServices,
    ConfigServices,
    DatasetServices,
    GraphServices,
    InstallerServices,
    MonitorServices,
    SettingsServices,
)
from backend.application.ports.comfy_environment import (
    ComfyEnvironment,
    ComfyEnvironmentInfo,
)
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
    BulkUpdateDatasetItems,
    CheckComfyConflicts,
    CheckRequirements,
    CommitDatasetItems,
    CreateDataset,
    DeleteDataset,
    DeleteGraph,
    DeleteGraphExecutions,
    DescribeRequirements,
    DiscardDatasetItems,
    GetConfig,
    GetConfigOptions,
    GetDataset,
    GetGraph,
    GetGraphExecution,
    GetSettings,
    InspectAsset,
    ListAssets,
    ListDatasetItems,
    ListDatasetSets,
    ListDatasetTasks,
    ListDatasets,
    ListGraphExecutions,
    ListGraphs,
    ListNodeCatalog,
    MakeAssetFolder,
    NodeDiagnostics,
    ReadConfigRaw,
    ReadDatasetFile,
    ReconcileDatasetTasks,
    ReconcileGraphExecutions,
    SaveGraph,
    SetDatasetPreview,
    StartDatasetTask,
    StartGraphExecution,
    StopDatasetTask,
    StopGraphExecution,
    SubscribeMonitor,
    SweepExecutionScratch,
    UpdateConfig,
    UpdateDatasetItem,
    UpdateSettings,
    UploadAsset,
    ValidateGraph,
    WriteConfigRaw,
)
from backend.domain.events import DomainEvent
from backend.application.ports.dataset_task_gateway import (
    DatasetTaskGateway,
    DatasetTaskLaunch,
)
from backend.infrastructure.config_options import PydanticConfigOptions
from backend.infrastructure.core_config_files import CoreConfigFiles
from backend.infrastructure.dataset_files import FsDatasetFiles
from backend.infrastructure.dataset_library import SqliteDatasetLibrary
from backend.infrastructure.dataset_previews import SqliteDatasetPreviews
from backend.infrastructure.dataset_tasks import SqliteDatasetTasks
from backend.infrastructure.events.callback_event_bus import CallbackEventBus
from backend.infrastructure.file_asset_store import FileSystemAssetStore
from backend.infrastructure.graph.catalog import DiscoveredGraphCatalog
from backend.infrastructure.graph.discovery import NodeRegistry, memory_fields_resolver
from backend.infrastructure.graph.runtime import ReflectedGraphRuntime
from backend.infrastructure.graph_event_stream import ExecutionEventTail
from backend.infrastructure.graph_task_gateway import InProcessGraphTaskGateway
from backend.infrastructure.memory_peak_store import SqlitePeakStore
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
#: Every message passed to `check`, in order. Counted so `finish` can refuse
#: to report success for a file that ran nothing -- see there.
CHECKED: list[str] = []


def check(condition: bool, message: str) -> None:
    print(f"  {'PASS' if condition else 'FAIL'}: {message}")
    CHECKED.append(message)
    if not condition:
        FAILURES.append(message)


def concrete_node_classes() -> set[str]:
    """Names of the node classes discovery *should* expose, derived from
    ``nodes/``'s own source rather than hardcoded.

    Discovery itself can't answer this -- it needs an expected value to
    be checked against, and a hardcoded number is a tripwire that goes
    off on every legitimate node addition or retirement (it did, twice
    over the 2026-10 optimizer retirements) while catching nothing that
    the structural checks around it don't already catch better.

    The rule: a class under ``nodes/`` belongs in the palette if it is a
    ``Node`` subclass and no *other* class under ``nodes/`` inherits from
    it. Inheriting-from is how the abstract bases are spelled in this
    codebase -- ``OptimizerNode``, ``TrainerNode``, ``DataSourceNode``,
    ``TextEncoderNode``, ``MonitorNode``, ``LRScheduleNode``,
    ``CheckpointSaverNode``, ``LoRAInjectorNode``, ``ModelProviderNode``,
    ``LossWeightingNode`` and ``Node`` itself are all bases, never
    palette entries. Filtering by that (rather than by name suffix or by
    re-implementing ``inspect.isabstract``) keeps the two in step when a
    new base or a new leaf is added.

    Uses AST only to learn *inheritance*, and imports nothing it wouldn't
    otherwise import, so it cannot mask a module that fails to import.
    """
    import ast
    import importlib
    import pkgutil
    from pathlib import Path

    import nodes
    from nodes.core import Node

    root = Path(nodes.__path__[0])
    inherited_from: set[str] = set()
    for path in sorted(root.rglob("*.py")):
        if "smoke_tests" in path.parts:
            continue
        for stmt in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(stmt, ast.ClassDef):
                for base in stmt.bases:
                    inherited_from.add(ast.unparse(base).split("(")[0].split(".")[-1])

    discovered: set[str] = set()
    for module_info in pkgutil.walk_packages(nodes.__path__, prefix="nodes."):
        if "smoke_tests" in module_info.name:
            continue
        module = importlib.import_module(module_info.name)
        for attribute in dir(module):
            candidate = getattr(module, attribute, None)
            if (
                isinstance(candidate, type)
                and issubclass(candidate, Node)
                and candidate.__module__.startswith("nodes.")
            ):
                discovered.add(candidate.__name__)
    return {name for name in discovered if name not in inherited_from}


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


class FakeDatasetTaskGateway(DatasetTaskGateway):
    """Scriptable fork gateway: spawn registers a fake pid as alive;
    tests kill it by discarding from ``alive``; ``spawn_error`` fails
    the launch (mirrors FakeTrainingGateway's posture).

    ``on_exit`` is the same hook the real gateway's reaper thread calls
    when the child is gone (it hands the admission claim back). Two
    ways a fake child dies, both faithful to the real gateway:

    * ``kill()`` -- the real gateway SIGKILLs the whole group, so the
      child is gone when kill returns and the waiter fires then;
    * ``child_exited(task_id)`` -- a *natural* end (the real child
      finalises its own row and the reaper notices the exit); a fake
      child has no thread, so a test that ends a task this way fires
      it explicitly.

    Never fired automatically on row finalisation: "the child is still
    running" is exactly what most task tests are asserting.
    """

    def __init__(self, on_exit=None) -> None:
        self.spawned: list[DatasetTaskLaunch] = []
        self.killed: list[int] = []
        self.alive: set[int] = set()
        self.spawn_error: Exception | None = None
        self.next_pid = 7777
        self._on_exit = on_exit
        self._pid_to_task: dict[int, int] = {}

    def spawn(self, launch: DatasetTaskLaunch) -> int:
        if self.spawn_error is not None:
            raise self.spawn_error
        self.spawned.append(launch)
        pid = self.next_pid
        self.next_pid += 1
        self.alive.add(pid)
        self._pid_to_task[pid] = launch.task_id
        return pid

    def kill(self, pid: int) -> None:
        self.killed.append(pid)
        self.alive.discard(pid)
        # SIGKILL is immediate: the real reaper fires when the child is
        # gone, and here the child is gone now. A kill that skipped the
        # waiter would hold a dead task's claim and refuse the next
        # start against a card that is actually free.
        task_id = self._pid_to_task.pop(pid, None)
        if task_id is not None and self._on_exit is not None:
            self._on_exit(task_id)

    def is_alive(self, pid: int) -> bool:
        return pid in self.alive

    def child_exited(self, task_id: int) -> None:
        """Simulate the child leaving: fire the exit hook if wired."""
        if self._on_exit is not None:
            self._on_exit(task_id)
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


class MemoryProbeNode(Node):
    """Touches the execution context's memory object (MEM-05 #1 proof).

    Reports the injected GraphMemory's grant, or ``None`` when the
    runtime was built without one (direct calls, spawns that carried no
    memory numbers) -- the same shape MonitorProbeNode gives the bus.
    """

    INPUTS = {}
    OUTPUTS = {"grant_mb": Port(name="grant_mb", type=float, required=False)}

    def build(self):
        memory = self.context.memory
        return {"grant_mb": None if memory is None else memory.grant_mb}


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
        MemoryProbeNode,
    )
}


def fixture_graph_registry() -> NodeRegistry:
    """A NodeRegistry over the fixture classes above (no nodes/ walk,
    no import errors, deterministic ordering)."""
    return NodeRegistry(scan=lambda: (dict(FIXTURE_NODES), ()))


class RefusingPackageInstaller(PackageInstaller):
    """An installer that refuses, for suites that are not about installing.

    Same reasoning as `UnreadableComfyEnvironment`, and the stakes are
    higher: a default that *worked* would mean any test that reached the
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


class UnreadableComfyEnvironment(ComfyEnvironment):
    """A ComfyUI environment that cannot be read, which refuses.

    The default for every suite that calls `build_services` without caring
    about the conflict check, and it is deliberately *not* a real read.

    The alternative -- wiring `LocalComfyEnvironment` -- would make any test
    that touched the endpoint a statement about the developer's own
    ComfyUI: green on this machine, red or (worse) a different answer on
    anyone else's. The same trap `path_tiers` has, for the same reason.

    It refuses rather than reporting an empty environment, because "safe
    with nothing pinned" is a *pass*. A default that can pass is a default
    that lets a test assert the check works without having checked
    anything.
    """

    def read(self, comfy_dir: str, venv_python: str | None = None) -> ComfyEnvironmentInfo:
        return ComfyEnvironmentInfo(
            comfy_dir=comfy_dir,
            requirements_error=(
                "no ComfyUI environment is wired in this test; pass a "
                "stub port if the test is about the conflict check"
            ),
        )


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
    device_probe: DeviceProbe | None = None,
) -> ApplicationServices:
    """Wire the use cases against fakes (the composition root's twin).

    The config / settings / assets / datasets / graph domains default to
    the *real* adapters
    over temp locations -- they are cheap, and exercising the real TOML /
    SQLite / filesystem code paths is the point of these tests. The
    dataset *gateway* is fake by default: spawning a real child that
    imports torch is nobody's unit test. The graph domain defaults to
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
    # After the defaults above, not before: this used to be built from
    # `project_root` while that was still None for every caller that did not
    # pass one, so ProjectPaths.root was None and any relative path through
    # it raised TypeError. Latent until ProjectPaths.config() started
    # resolving its root, which turned the TypeError into an AttributeError
    # -- but the wiring was wrong either way.
    paths = ProjectPaths(root=project_root)
    if settings_store is None:
        database = SqliteDatabase(project_root / "test-settings.db")
        database.initialize()
        settings_store = SqliteSettingsStore(database, project_root)
    layout = WorkspaceLayout(
        project_root, runs_dir=runs_dir, settings_kv=settings_store.get
    )
    if dataset_library is None:
        dataset_library = SqliteDatasetLibrary(layout)
    if dataset_tasks is None:
        tasks_db = SqliteDatabase(project_root / "test-dataset-tasks.db")
        tasks_db.initialize()
        dataset_tasks = SqliteDatasetTasks(tasks_db, clock)
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
    # One directory for the supervisor and the sweep, so a test
    # can assert that clearing the history cleared the disk too.
    # Unconditional: a caller that passes only ``graph_executions``
    # still needs the path defined (the old nested assignment left it
    # unbound and the first use raised NameError).
    graph_scratch = project_root / "test-graph-scratch"
    # The test twin of bootstrap's peak store (MEM-04 #2): one store per
    # composition, next to the databases this build already owns, shared
    # by admission (reads) and the supervisor (files what a run reports).
    peak_store = SqlitePeakStore(project_root / "test-memory-peaks.db")
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
        dataset_tasks=dataset_tasks,
        foreign_reserve_mb=DEFAULT_FOREIGN_RESERVE_MB,
        process_overhead_mb=DEFAULT_PROCESS_OVERHEAD_MB,
    )
    # Unconditional, like ``graph_scratch``: a caller that passes only
    # ``dataset_gateway`` still needs ``task_sweeper`` bound (the old
    # nested assignment left it unbound and its first use raised
    # NameError). A fake child has no waiter thread of its own, so
    # tests end tasks by calling ``child_exited`` -- the same thing the
    # real gateway's reaper does on child exit.
    if dataset_gateway is None:
        dataset_gateway = FakeDatasetTaskGateway(
            on_exit=lambda task_id: release(memory_ledger, task_owner(task_id)),
        )
    task_sweeper = DatasetTaskSweeper(
        tasks=dataset_tasks,
        gateway=dataset_gateway,
        clock=clock,
        memory_ledger=memory_ledger,
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
                runtime_factory=lambda _writer: graph_runtime,
            ),
            events=publisher,
            clock=clock,
            memory_ledger=memory_ledger,
            peak_store=peak_store,
            monitor_bus=monitor_bus,
            scratch_dir=graph_scratch,
            make_tail=ExecutionEventTail,
        )
    # One job dict, so a job created by a test can be polled through the
    # same services object the route reads.
    install_jobs: dict = {}
    refusing_installer = RefusingPackageInstaller()
    return ApplicationServices(
        config=ConfigServices(
            read=GetConfig(files=config_files, paths=paths),
            update=UpdateConfig(files=config_files, paths=paths),
            read_raw=ReadConfigRaw(files=config_files, paths=paths),
            write_raw=WriteConfigRaw(files=config_files, paths=paths),
            options=GetConfigOptions(options=config_options),
        ),
        installer=InstallerServices(
            check=CheckRequirements(
                inventory=MetadataPackageInventory(),
                device=shared_probe,
                memory_ledger=memory_ledger,
            ),
            # The same object `check` uses, so a test cannot be handed two
            # views of the machine. FakeDeviceProbe falls back to the single
            # current device and reports enumerate_all False, which is the
            # honest answer for a stub.
            device_probe=shared_probe,
            apply=ApplyInstallation(settings=settings_store),
            manifest=DescribeRequirements(),
            # No interpreter: these tests never ask for a conflict check, and
            # wiring the real one would put a subprocess in every suite that
            # happens to call build_services. The field is required rather
            # than defaulted so that a new installer capability cannot be
            # added without deciding what the tests should see of it -- which
            # is how the four suites that construct InstallerServices by hand
            # found out.
            conflicts=CheckComfyConflicts(environment=UnreadableComfyEnvironment()),
            # Both share `install_jobs`, so polling works. The installer
            # refuses by default: a default that ran real pip would make any
            # test that reached this endpoint for another reason capable of
            # modifying a real environment.
            install=StartInstall(
                installer=refusing_installer,
                project_root=project_root,
                base_python=sys.executable,
                jobs=install_jobs,
            ),
            install_status=GetInstall(jobs=install_jobs),
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
                checkpoints_dir=lambda: layout.checkpoints_dir,
                memory_ledger=memory_ledger,
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
                memory_ledger=memory_ledger,
                # Fixture nodes declare no memory_fields and fixture
                # graphs name no dataset, so every fingerprint here is
                # unknown -- exactly what these tests claimed before the
                # read half existed, never a zero peak.
                peak_source=GraphPeakSource(
                    datasets=dataset_library,
                    peaks=peak_store,
                    resolve_memory_fields=memory_fields_resolver(graph_registry),
                ).observed,
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
                executions=graph_executions, events=publisher,
                scratch=SweepExecutionScratch(
                    executions=graph_executions, scratch_dir=graph_scratch,
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
        conn.execute("PRAGMA user_version = 2")
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
