"""Everything that runs and records graphs: catalog, runtime, ledger, supervisor and the single start-execution use case."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..application.graph_peak_source import GraphPeakSource
from ..application.graph_supervisor import GraphExecutionSupervisor
from ..application.lifecycle_writer import ExecutionLifecycleWriter
from ..application.memory_admission import LedgerProvider
from ..application.use_cases.graph_modules import DescribeGraphModules, ResolveBoard
from ..application.use_cases import (
    SweepExecutionScratch,
    StartGraphExecution,
)
from ..config import GRAPH_EXECUTION_CHILD
from ..infrastructure.host_memory import ProcHostMemory
from ..infrastructure.graph.contract_catalog import ContractNodeCatalog
from ..infrastructure.graph.memory_fields import memory_fields_resolver
from ..infrastructure.graph.runtime import ContractGraphRuntime
from ..infrastructure.execution_metrics import FileExecutionMetrics
from ..infrastructure.graph_event_stream import ExecutionEventTail
from ..infrastructure.graph_task_gateway import (
    InProcessGraphTaskGateway,
    SubprocessGraphTaskGateway,
)
from ..application.ports.graph_task_gateway import GraphTaskGateway
from ..infrastructure.memory_peak_store import SqlitePeakStore
from ..infrastructure.persistence.graph_library import SqliteGraphLibrary
from ..infrastructure.persistence.app_library import SqliteAppLibrary

from .infrastructure import Infrastructure, node_environment


@dataclass(frozen=True, slots=True)
class GraphKit:
    catalog: ContractNodeCatalog
    runtime: ContractGraphRuntime
    graph_library: SqliteGraphLibrary
    app_library: SqliteAppLibrary
    memory_ledger: LedgerProvider
    writer: ExecutionLifecycleWriter
    scratch_dir: Path
    metrics: FileExecutionMetrics
    sweep_scratch: SweepExecutionScratch
    supervisor: GraphExecutionSupervisor
    peak_source: GraphPeakSource
    start_execution: StartGraphExecution
    describe_modules: DescribeGraphModules
    resolve_board: ResolveBoard

    @classmethod
    def build(cls, infra: Infrastructure) -> "GraphKit":
        settings, layout, executions = infra.settings, infra.layout, infra.executions

        # One NodeRegistry feeds both the palette and the executor -- a single discovery cache, swapped atomically
        # on refresh. Execution rows and the saved library are server state in backend.db; the runtime keeps the
        # default memory releaser (gc, then the XPU caching allocator) so freed VRAM goes back to the driver after
        # every run.
        catalog = ContractNodeCatalog(environment=lambda: node_environment(layout))
        registry = catalog.registry
        runtime = ContractGraphRuntime(registry, env=lambda: node_environment(layout))

        # One ledger per container, built on first use (MEM-03): the total comes from the cached probe, which may
        # only be able to answer after the installer has run -- probing here would tax every start-up and, worse,
        # freeze "unknown" forever on a fresh machine. Construction rebuilds it from the unfinished rows that carry
        # a claim, so a restart reproduces the same held total; the startup reconcile then releases the rows that
        # turn out to be debris.
        memory_ledger = LedgerProvider(
            probe=infra.device_probe,
            graph_executions=executions,
            foreign_reserve_mb=settings.memory_foreign_reserve_mb,
            process_overhead_mb=settings.memory_process_overhead_mb,
        )
        writer = ExecutionLifecycleWriter(clock=infra.clock, repository=executions, events=infra.publisher)

        # The run happens somewhere other than here. Both gateways run the *same* producer
        # (graph_task_worker.run_execution) and both report through the same event file, so choosing between them is
        # a choice of where the interpreter state lives and nothing else. The child gets no registry -- it discovers
        # its own, which costs a few seconds of node discovery once per run and buys an isolation boundary the
        # server's already-warm registry cannot cross.
        if settings.graph_execution_mode == GRAPH_EXECUTION_CHILD:
            gateway: GraphTaskGateway = SubprocessGraphTaskGateway(layout)
        else:
            gateway = InProcessGraphTaskGateway(registry, environment=lambda: node_environment(layout))

        # Supervision scratch, next to the database rather than in the runs dir: these are per-execution working
        # files (the graph handed to the child, its event stream, its log), not run artifacts anyone collects.
        scratch_dir = infra.database.path.parent / "graph_executions"
        scratch_dir.mkdir(parents=True, exist_ok=True)
        # Nothing else ever removed these files, so the directory grew by one run's worth forever while the UI
        # reported the history as cleared. What a run reported to its monitors outlives that scratch: the sweep
        # files it first.
        metrics = FileExecutionMetrics(scratch_dir)
        sweep_scratch = SweepExecutionScratch(executions=executions, scratch_dir=scratch_dir, metrics=metrics)

        # Remembered peaks per configuration (MEM-04), in their own file next to the database: one writer for
        # peaks, shared by admission (reads through it) and the watcher (files what a child reports). Self-creating
        # table, so no migration owns it.
        peak_store = SqlitePeakStore(infra.database.path.parent / "memory_peaks.db")
        supervisor = GraphExecutionSupervisor(
            executions=executions,
            writer=writer,
            gateway=gateway,
            events=infra.publisher,
            clock=infra.clock,
            memory_ledger=memory_ledger,
            peak_store=peak_store,
            monitor_bus=infra.monitor_bus,
            scratch_dir=scratch_dir,
            make_tail=ExecutionEventTail,
        )

        # Fingerprint (MEM-01) + remembered peak (MEM-04), built once here and shared by everything that has to
        # agree on which configuration a peak belongs to: the panel's preview, the run admission below, and the
        # watcher above that files this run's own reported peaks. Three constructions of the same thing is three
        # chances not to.
        peak_source = GraphPeakSource(peaks=peak_store, resolve_memory_fields=memory_fields_resolver(registry))

        # One instance, shared by the editor's Run and by running an app: the single-active check lives under this
        # object's lock, so a second instance would be a second lock and a window in which two graphs could both
        # pass it.
        start_execution = StartGraphExecution(
            executions=executions,
            writer=writer,
            runtime=runtime,
            launcher=supervisor,
            clock=infra.clock,
            memory_ledger=memory_ledger,
            # The same instance the panel previews through: the fingerprint it computes is what the watcher files
            # this run's own reported peaks under, and a preview keyed differently would quote a peak belonging to
            # some other configuration.
            peak_source=peak_source.observed,
            host_memory=ProcHostMemory(),
        )
        describe_modules = DescribeGraphModules(catalog=catalog)
        return cls(
            catalog=catalog,
            runtime=runtime,
            graph_library=SqliteGraphLibrary(infra.database),
            app_library=SqliteAppLibrary(infra.database),
            memory_ledger=memory_ledger,
            writer=writer,
            scratch_dir=scratch_dir,
            metrics=metrics,
            sweep_scratch=sweep_scratch,
            supervisor=supervisor,
            peak_source=peak_source,
            start_execution=start_execution,
            describe_modules=describe_modules,
            resolve_board=ResolveBoard(describe=describe_modules, runtime=runtime),
        )
