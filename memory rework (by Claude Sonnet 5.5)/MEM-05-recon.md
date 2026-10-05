# MEM-05 recon: the child-side memory terrain

Working notes for implementing MEM-05 of `TASK-memory-rework.md`. Verified
against the tree at commit `b74a393` (2026-10-05); line numbers drift —
re-verify before relying on them. Deliberately *not* in `docs/`: it is
point-in-time terrain, and `docs/design/backend/README.md`'s own rule is that
inventories restating the code cannot help but drift.

## What does not exist yet (greenfield, per spec)

- **`FakeDevice`** — the spec's test double (capacity, allocation, overhead, a
  foreign user that can appear) has no namesake. Closest existing fakes:
  `_FakeDeviceContext` (scripted `reserved_mb` sequence, synchronize counter)
  in `nodes/smoke_tests/smoke_test_resource_control_strict.py`, and
  `_FixedDeviceContext` in `smoke_test_resource_profile.py`. Server-side
  `FakeDeviceProbe` is a different thing entirely (probe of record, not the
  device itself).
- **Device *free* memory** — `DeviceContext` exposes `memory_stats()`,
  `total_memory_mb()`, `synchronize()`, `empty_cache()` but no free/total
  query. The in-repo precedent for `mem_get_info()` is
  `nodes/smoke_tests/fast_construction.py::_device_memory_mb()`
  (`torch.xpu.mem_get_info()` / `torch.cuda`) plus `archive/core/comfy_setup.py`.
- **`set_per_process_memory_fraction`** — zero references anywhere in the repo
  (only in the task/plan prose). The allocator backstop is new code.
- **The layering test** (nodes must not import backend): the *rule* exists in
  prose (ADR 0005 "Layering", task rule 9) but nothing enforces it. The
  natural home is `backend/tests/test_declared_dependencies.py` — it already
  AST-walks imports with a `LOCAL` frozenset containing both `"nodes"` and
  `"backend"`, but checks no direction between local packages.
- **The child's periodic `memory` producer** — `EventKind.MEMORY` and
  `ExecutionEventWriter.memory(reserved_mb, allocated_mb, peak_mb, budget_mb)`
  exist, but **no production code calls them** (tests only). The writer's own
  header says "the fixed-interval producer is MEM-05 #4", and
  `test_memory_wiring.py::_MemoryReportChild`'s docstring pins the same
  promise. There is no `threading.Timer`/child-side thread anywhere today.
- **Lifting node values into `MemorySettings` on load** (the
  `VRAMBudgetControllerNode` shim's other half): no such code exists;
  `vram_budget_mb` appears nowhere under `backend/`. Candidate hooks are
  `GraphDefinition.from_dict` (format-1 graphs) and `save_graph`.

## The seams (injection points)

- **Spawn args.** `GraphTaskLaunch` is `frozen/slotted` with
  `execution_id, graph_path, event_path, log_path`; filled by
  `GraphExecutionSupervisor.launch`, command built in
  `SubprocessGraphTaskGateway._build_command`, parsed by
  `graph_task_worker._build_parser` (all three flags `required=True`).
  Adding two flags touches: port dataclass, supervisor, command builder,
  parser — and the in-process gateway, which has no argv at all and must get
  the same numbers through `run_execution`/`build_runtime` or the two paths
  diverge (the worker module's own docstring demands they stay identical).
- **Where the numbers live today.** Only on the *row*: `GraphExecution.reserved_mb`
  (the grant) and `row.memory → EffectiveMemory` (the budget source).
  `ExecutionLauncher.launch(execution_id, graph)` receives neither, and the
  graph JSON carries only the graph's own `MemorySettings` — never the
  effective admission result (overrides, peak-derived demand). So the launch
  port or the supervisor must grow a way to pass them.
- **Units.** budget = allocator MB; grant = device MB (ADR 0005 Units;
  stated/observed grants are `demand + process_overhead_mb`).
- **`ExecutionContext`** is a plain two-field bag
  (`monitor_bus`, `cancel_event`) whose docstring explicitly sanctions new
  fields; the runtime constructs it inside `ReflectedGraphRuntime.execute`
  and hands it to `cls(context).build(...)`. That construction is the
  exposure point for `memory`.
- **`graph_task_worker.main()`** already has argv + writer + graph before
  `run_execution` — "build `GraphMemory` first, before loading anything"
  maps there (and to the in-process gateway's `body()` equivalently).
- **Duck-typing precedent for reaching the writer from nodes/ code** without
  breaking layering: `_EventMonitorBus` (report/clear only) — the same
  technique fits telemetry callbacks.

## Must-pass-unchanged (the adapter's real contract)

- **`nodes/smoke_tests/smoke_test_resource_control_strict.py`** is *the* file:
  8 checks pinning `_make_room()` order, never-touch, strict raise/pass,
  `ensure_loaded` reload+sync, the omitted-`device_ctx` default, and
  `release()` semantics — and it reaches into privates
  (`_coordinator`, `_offloaded`, `_device_ctx`). "Run its whole existing test
  file against the adapter" means this one.
- **`smoke_test_vram_budget_controller.py`** — node still returns a real
  handle unconditionally, keeps its over-budget warning (both numbers named),
  and survives `total_memory_mb() is None`; it monkeypatches
  `DeviceContext.for_device`.
- **Fakes with fixed signatures**: `_RecordingResourceControl` in
  `smoke_test_text_encoder_cache.py` has `register()` **without** a
  `sacrificable` kwarg — the adapter must not start calling `register` with
  new arguments. Also `_FakeResourceControl` in `smoke_test_managed_trainer.py`
  and `smoke_test_trainer_profiling.py`, `_RC` in `smoke_test_t_probe.py`.
- **Real-handle end-to-end**: `smoke_test_managed_trainer.py` builds
  `BudgetedResourceControlHandle(ResourceBudget(1e9, 0.0), device="cpu")` at
  two sites and runs full trainers through it.
- **ABC surface to preserve exactly**: `register(name, resident,
  offloadable=False, sacrificable=False)`, `before_step(step)`,
  `ensure_loaded(name)`, `release(name)`, `usable_budget_mb() -> Optional[float]`.

## Traps and open decisions

1. **Strict default mismatch**: `ResourceBudget.strict=False` vs
   `MemorySettings.strict=True`. Any lift of node values into settings must
   decide this mapping explicitly.
2. **Budget may be unknown** (`vram_max_mb == "auto"` with no peak): decide
   what `--memory-budget-mb` carries then (absent ⇒ no fraction backstop is
   the honest reading; the warning+continue case still gets recorded).
3. **Physical check exit**: the spec says "write an `outcome` with the
   numbers and exit cleanly" — outcome semantics are what the supervisor
   treats as completion, so this must be a recorded, explained exit, not an
   error crash.
4. **Eviction telemetry crosses the layer line**: evictions happen in
   `ResourceControlHandle` (nodes/), records are written by
   `ExecutionEventWriter` (backend/) — needs a duck-typed callback on
   `GraphMemory`, mirroring `_EventMonitorBus`.
5. **Argv/adoption safety is already fine**: `find_by_argv` matches the
   module marker plus *adjacent* flag/value pairs, so extra pairs don't
   disturb adoption; no test asserts the exact command list. But every
   real-child test (`test_graph_task_gateway.py`, `test_graph_adoption.py`)
   dies loudly if the parser rejects the new flags — parser and builder
   must land together.
6. **`VRAMBudgetControllerNode` warning path** (`budget > total` warning via
   `DeviceContext.for_device(device).total_memory_mb()`) must survive the
   shim transformation.
7. **Telemetry payload**: `writer.memory` takes exactly the four numbers;
   "record which backstop case happened" (fraction set vs unavailable) has
   no field for it today — the record shape may need one more key, which
   ripples into the reader, schema and contract test.

## Related tests that will notice

- `backend/tests/test_graph_task_gateway.py` — 7 real-child spawn tests.
- `backend/tests/test_graph_adoption.py` — real spawns + argv discovery.
- `backend/tests/test_graph_event_stream.py` — memory-record round trip.
- `backend/tests/test_memory_wiring.py` — `_MemoryReportChild` scripted
  frames; its docstring (and `graph_event_stream.py`'s header) are the
  placeholders to update once #4 produces real frames.
