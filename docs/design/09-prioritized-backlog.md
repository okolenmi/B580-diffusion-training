*[← docs/design index](README.md)*

# 10. Prioritized backlog

**New: first-run installer and dependency gate** --
[`11-first-run-and-installer.md`](11-first-run-and-installer.md). A
separate item rather than one on this list because it has a different
shape: the work is a sequence that is only worth doing as a sequence
(check -> gate -> download -> first-run state -> wizard -> multi-root
model search), and the second half of it is a *design* item rather than a
build item. The part that is pure friction-removal and can be taken
independently is Phase B: make `run_server.sh` check its dependencies and
say what is missing, instead of starting and letting the failure surface
as a `ModuleNotFoundError` traceback. That is a small change with most of
the user-visible benefit.

Two things in that plan are worth noticing before anything is built. We
would be installing into a venv the user did not create -- made safe by
splitting the requirements into "everything" and "additions only" and
installing the latter under a constraints file of what that venv already
holds, so nothing already installed can move. And model lookup would
become main-then-reserve rather than the single-root-with-override it is
today, which changes what existing code has to assume in five places.

1. **Verify `NF4WeightStore`'s quality against a real training run**
   (3.3). The diffusion-specific quality question -- does NF4's real ~9%
   relative RMSE (see that module's own docstring) actually produce
   usable LoRA training results on this project's real UNet -- still
   needs checking directly, not assumed from QLoRA's own LLM benchmarks.
   A `MemoryManager`-backed scratch buffer for the dequantized tensor
   (real, separate optimization, not blocking correctness) is the other
   remaining piece -- see `nf4_lora_layer.py`'s own docstring.
   `DoRAAdapter` doesn't honor `NF4WeightStore` yet (`QDoRA`) -- real,
   separate follow-up, not done here.

**A real gap surfaced while landing `ResourceProfile`, not yet its own
backlog item because nothing above needs it yet:** there is no single
shared `MemoryManager` instance reachable from the trainer's `build()` --
each optimizer execution strategy
that uses one (`ChunkedScratchBufferStrategy`) constructs its own private
instance when none is injected, and no `Composed*OptimizerNode` exposes
a port to inject a shared one. Harmless today (each strategy's own
private manager is internally consistent), but it means
`ResourceProfile.memory_manager_stats` is `None` in every real run, and
it'll become a real problem the moment two things need to share one VRAM
budget on purpose (e.g. a future `AutoResourcePolicy`, or several
strategies deliberately pooling scratch space) -- worth threading a
shared `MemoryManager` through optimizer construction as part of
whichever future item first has a concrete reason to, rather than
speculatively now.

**Run the test suites in parallel. Done** (`277b16f`, and the `nodes/`
runner alongside it).

The backend suite was **21.5s serial** with the top five files 53% of it
and the CPU 58% idle; `backend/tests/run_all.py` now runs the files in a
`ProcessPoolExecutor`, **22.69s -> 6.32s on 6 jobs**. Each file keeps its
own `TMPDIR`, removed afterwards -- which both contains the litter every
test file leaves behind (see `docs/known-issues/open.md`) and is what makes
the files independent enough to run at once. Output order is preserved:
`pool.map` yields in submission order, because a log interleaving 26 files
makes a failure much harder to find.

`nodes/smoke_tests/run_all.py` was **119.9s serial** over 66 files, with
the cost spread evenly (ten slowest = 26%, longest single file 4.4s), so
it parallelises well -- but **not all of it**. An explicit serial list runs
first and alone for the tests that assert on *device state*
(`device_context_equivalence` asserts the `xpu_empty_cache` call order,
`memory_manager` asserts allocator and residency, `sdxl_text_encoder_offload`
asserts memory was actually released). These do not strain VRAM -- that was
measured, not assumed -- but two of them at once are measuring each other's
allocations, so a failure would mean nothing. **113.5s -> 46.5s**, and the
serial list is now 43% of what remains: the next lever is trimming that
list, which trades risk-reduction for speed and should be a deliberate
call rather than a default.

Multiprocessing there uses the **spawn** context, not the default `fork`:
a forked child inheriting a torch/XPU-initialised parent crashes the
pool, and it does so as `ConnectionResetError` from the forkserver rather
than anything legible.

**Validation work, not construction -- real code exists for both, what's
missing is a real run:**

- **`RescaledZeroTerminalSNRSchedule` + `VPredParameterization`, end to
  end** (1.4). What's missing is a real training run and qualitative
  image-quality evaluation -- does it actually fix medium-brightness
  clustering on this project's own data. Unlike everything numbered
  above, there's no old code path to equivalence-test against; this needs
  real training runs to trust, not a unit test.
- **`LoRAPlusGroups`, actually run** (3.4). What's missing is running it
  on a real LoRA training job and comparing against a `UniformGroups`
  baseline, at whatever `ratio` (the `16.0` default is an unverified
  starting point) turns out to matter for this project's own data.
- **`GreedyRatioPlacement`, wired and run** (2.3). Nothing has actually
  run the instrumentation against this project's own UNet yet, and
  `ComfyUNetLoRANode` has no port to select `ProfilingCheckpointing` or
  `GreedyRatioPlacement` in the first place -- both real, separate,
  smaller follow-ups once a first profiled run's numbers exist to wire
  a real placement decision against.
- **`DoRAAdapter`, a real training run** (3.1). Not yet run on this
  project's own data to confirm the quality improvement DoRA reports in
  its own published benchmarks (LLaMA/LLaVA/VL-BART, not diffusion
  UNets) actually shows up here too. Checkpoint save/load (direction +
  `.dora_scale` magnitude + alpha) is real now for the common, unsplit
  case -- see 9.1/9.2 -- so this item is validation-only, same as the
  others in this list.
- ~~**A tiny-parameter (`< 10,000` element) `ExecutionStrategy` for
  Adafactor's cross-parameter batching case** (11.1)~~ -- **closed
  2026-10-02 by retiring the wrapper instead of building it.**
  `nodes/optimizer/adafactor.py` is deleted; `ComposedAdafactorOptimizerNode`
  is the only Adafactor node. `ChunkedXPUAdafactor` tied every tiny
  parameter in the optimizer together into one shared clip/EMA state --
  a batching-strategy concern no `Algorithm` can see (by design, see
  `algorithms/base.py`) -- and the proposed fix was a
  `ShapeGroupedBatchStrategy` variant grouping "under a size threshold,
  any shape". Part C of `smoke_test_adafactor_tiny_parameter_gap.py` had
  already measured that the shared state *contaminates*: a parameter's
  update depends on unrelated parameters' gradients sharing its batch. So
  the strategy would have existed to reintroduce coupling the canonical
  math deliberately lacks. The formula gap that was real is closed
  (`AdafactorAlgorithm.tiny_parameter_threshold`, Fused's per-parameter
  mechanism). What remains is one unmeasured performance delta -- see
  `docs/known-issues/open.md`, which also carries the B580 measurement
  that should precede writing any such strategy.
- **`core`/`manager` coupling in the model and dataset domains** (new
  section, not yet numbered above -- see `docs/architecture.md`). The
  `optimizer/` domain's `Algorithm`/`ExecutionStrategy` split proved a
  domain can be fully separated from `core/`'s legacy implementation,
  verified equivalent, and the old wrapper retired; text encoding then
  followed on 2026-10-02 (`SDXLClipEncoder` -> `nodes/model/clip_encoder.py`,
  a relocation, `core/clip_encode.py` kept as a shim for `core/`'s and
  `manager/`'s own use). Three sub-pieces remain, and they are not
  equally hard:

  - ~~**`nodes/model/` LoRA/UNet injection** (`core.lora`,
    `core.unet_wrapper`)~~ -- **done 2026-10-02.** This was the one
    flagged as "not a thin import": `core.unet_wrapper` is the *model*
    every LoRA path is built on, and `core.lora._inject_lora` resolved
    `LoRALinear`/`LoRAConv2d` as module-level names at call time, which
    is what `adapter_injection.py` patched in place to substitute DoRA
    and NF4 layers. Owning the file made that patch unnecessary --
    `_inject_lora` now takes the classes as an argument -- which removed
    a documented concurrent-build race, deleted `lora_class_cache.py`
    entirely, and fixed four `isinstance` gates that had been silently
    skipping every DoRA and NF4 layer. `nodes/` now imports nothing from
    `core/`; `core/{lora,unet_wrapper,clip_encode,seed}.py` are re-export
    shims for `core/`'s own use.
  - **`nodes/dataset/managed.py`** (`manager.loader`, and through it
    `manager/t_sampling.py` -> `core.noise_schedule.sample_timestep` and
    `core.model_io.make_init_noise`, `core.seed.derive_seed`). Note the
    shape here is *inverted* relative to the other two: `nodes/` doesn't
    depend on `manager/`, `manager/` is the implementation and `nodes/`
    calls into it. Worth deciding deliberately which side should own it.
  - **`manager/builder.py`'s own use of `core/`** (9 modules), which is
    the backend's dataset ingestion path rather than anything `nodes/`
    reaches.

  So dataset ingestion is now the only domain left to unwire. It is
  worth doing deliberately rather than by following the pattern above:
  the other two were relocations of self-contained modules, whereas here
  the dependency runs the opposite direction -- `manager/` is the
  implementation and `nodes/dataset/managed.py` calls into it. Deciding
  which side owns `sample_timestep`, `make_init_noise` and the dataset
  loading is the actual content of that piece, and the move is only the
  easy half.

**Not recommended as near-term work, with reasoning kept where it's
argued in full:** `ComponentRegistry`/`TrainingRecipe`/`PipelineFactory`
(5.3, 5.4 -- nothing built through `nodes/components/` so far is
graph-editor-selectable, so the side-by-side-registration problem these
solve hasn't materialized). **Deliberately deferred or rejected, for the
reasons section 7 gives in full:** `AutoResourcePolicy`, automatic
eviction inside `MemoryManager`, layer-wise base offload, flow matching,
GaLore, 8-bit optimizer moments.
