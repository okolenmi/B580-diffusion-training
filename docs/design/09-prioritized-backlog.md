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

**Run the backend suite in parallel (cheap, and it gets worse every time a
test is added).** `backend/tests/run_all.py` runs 26 files in sequence, one
interpreter each. Measured on this machine:

| | |
|---|---|
| serial suite | **21.5s** |
| slowest file | `test_api_graphs.py`, 2.77s |
| top 5 files | 11.5s — **53% of the total** |
| cores available | 6 |

So the floor for a parallel run is roughly the slowest file (2.8s) plus
pool overhead, and a `ProcessPoolExecutor` over the files should land
around 5-6s: a **~4x** cut on every gate run. The files are already
independent — each builds its own temporaries, and `run_all.py` already
runs them in separate processes with a per-file `TMPDIR`, which is exactly
the isolation a pool needs. The change is small.

Worth doing for a reason beyond the seconds: **the cost is per-file and the
number of files only goes up.** Every test added makes every gate run
slower, in a repo whose whole quality argument rests on running the gate.
This is the change that stops that from compounding. `scripts/coverage_report.py`
and `scripts/mutation_report.py` both already do the parallel version for
their own sub-processes, so the pattern exists in the repo — this would
apply it to the suite itself.

Two things to preserve when doing it: the per-file `TMPDIR` (it is what
makes the files independent, and `run_all.py` also relies on it to clean
up), and the exit code, since `run_all.py` is what the gate calls.

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
