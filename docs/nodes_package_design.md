# `nodes/` -- design, current state, priorities

Read this first. It's meant to stay short -- if it starts growing past a
few hundred lines again, that's a signal to cut, not to keep appending.
Detailed investigation narratives (bug hunts, equivalence-test writeups)
belong in commit messages when the work happens, not here -- `git log`
has the full history for anything condensed to a one-liner below.

## What this is, and what it isn't (read this before assuming anything)

Two independent pipelines exist in this codebase today:

1. **Production**: `core/` + `manager/`, config/TOML-driven, launched as a
   subprocess (`server/routes_training.py` -> `core/cli.py` ->
   `core.trainer.Trainer`). This is what real training runs use. Untouched
   by any of the work below.
2. **`nodes/`**: an in-process graph of typed `Node.build()` calls, run via
   `server/graph_executor.py` (`/nodegraph` tab). **A real, working, if
   minimal, dataset -> LoRA training graph already runs through this
   today** (`nodes/train/supervised.py`'s `SupervisedLoRATrainerNode`,
   wired to dataset/model/optimizer nodes below). This is the one being
   actively developed.

They share no code path except that `nodes/` still imports `core/` and
`manager/` directly for the technical pieces it hasn't rewritten yet
(dataset loading, UNet/LoRA construction, noise-schedule math, model I/O
parameterization -- see "Current state by subpackage" below). **`nodes/`
is not yet the production path, and most of its technical internals
still are old code, not new code that happens to wrap old code for
compatibility.**

## Rules

- **Strict OOP for all new `nodes/` code.** Real ABCs (`Node`/`Port`,
  `Algorithm`/`ExecutionStrategy`, `OptimizerHandle`, ...), no ad-hoc
  dict-passing or flag-threading.
- **`core/`/`manager/` are reference material, not a source to copy.**
  Read them to understand current behavior; design the `nodes/` version as
  if starting fresh. Wrapping via composition is only justified where the
  old logic is genuinely correct and not worth re-deriving (e.g. actual
  UNet/LoRA-layer math) -- and even then, equivalence-test it, don't
  assume it. Where a technical piece gets rewritten instead of wrapped, it
  belongs in `nodes/components/` (see below), not scattered elsewhere.
- **Comment budget**: a 1-4 line module docstring saying what a file is
  and what it depends on is enough. Long "why this is safe" writeups
  belong in a commit message, not the `.py` file.
- **Any reusable device buffer goes through `nodes/memory/manager.py`'s
  `MemoryManager`**, not a hand-rolled `self._scratch` attribute. Today
  only `ChunkedScratchBufferStrategy` actually does this -- extending that
  to the rest of `nodes/` (dataset batching, model activations, anything
  else that allocates a reusable device buffer) is open work, not done.
- **New non-legacy implementations of a `core/`/`manager/` dependency go
  in `nodes/components/`.** Nothing has moved there yet -- it's a
  destination for future work, not a completed migration. Same discipline
  as the optimizer rewrite: equivalence-test against the thing being
  replaced before anything switches over to it.

## Current state by subpackage

| Subpackage | State |
|---|---|
| `nodes/core.py` | `Node`/`Port` base. Stable, no dependencies. |
| `nodes/memory/` | `MemoryManager` -- centralized device-buffer lifecycle (get/release/free, optional `torch.xpu.MemPool`). Used today only by `ChunkedScratchBufferStrategy`. Should be the standard path project-wide; isn't yet. |
| `nodes/optimizer/` | Real rewrite, not a wrapper, for the math itself: `Algorithm` (CAME/Adafactor/AdamW) x `ExecutionStrategy` (simple/chunked/foreach) x `ComposedFusedOptimizerHandle` (backward-hook execution), all equivalence-tested against the legacy classes they correspond to. Legacy-wrapping nodes (`adamw.py`, `foreach_adafactor.py`, `foreach_came.py`, `fused_adafactor.py`) still exist, unretired, as a fallback until the composed nodes are validated on real XPU hardware. See "Optimizer subpackage" below for detail. |
| `nodes/model/` | Mostly adapter-only. `lora_injector.py` delegates real UNet construction and LoRA-layer injection to `core.unet_wrapper`/`core.lora` entirely. `gradient_checkpointing.py` patches ComfyUI's own `CheckpointFunction` directly (third-party, not this project's old code, but still not owned/rewritten). `checkpoint_loader.py`, `lora_saver.py`, `lora_checkpoint_loader.py`, `parameters.py`, `text_encoder*.py`, `lora_phases.py` are real, self-contained `nodes/` logic (caching, phase-split bookkeeping, port contracts) -- not legacy wraps, but they still call into old code for the underlying model/tensor operations they sit on top of. |
| `nodes/dataset/` | Adapter-only. `managed.py` wraps `manager.loader.ManagedDatasetLoader` completely -- no dataset logic reimplemented in `nodes/` yet. `renoise.py` is real `nodes/` logic (decorates any `TrainingBatchSource`). |
| `nodes/train/` | `SupervisedLoRATrainerNode` -- real, working v1 step loop (see its own module docstring for explicit scope limits: no CFG dual-pass, no grad accumulation, no resume cadence). Calls `core.model_io`/`core.noise_schedule`/`core.comfy_setup` directly for per-step math -- not yet reimplemented, a `nodes/components/` candidate. `loss.py`/`schedule.py` (loss weighting, LR schedules) are self-contained, no legacy dependency. |
| `nodes/monitor/` | Self-contained (SSE-fed live step/loss/lr dashboard). No legacy dependency. |
| `nodes/primitive/` | Trivial constant-value nodes. No legacy dependency. |
| `nodes/smoke_tests/` | One real-torch smoke test per component that has one. `run_all.py` discovers and runs everything matching `smoke_test_*.py`; run a file directly for detail on one thing. 26 files as of this writing, all passing (including on real user hardware, not just this sandbox). |

## Planned structural changes (not started)

- **`nodes/components/`**: destination for rewritten, non-legacy versions
  of the `core/`/`manager/` dependencies listed above (dataset loading,
  UNet/LoRA construction, noise-schedule math, model I/O
  parameterization, ...). Created this session, empty except a README --
  see `nodes/components/README.md`. Move things here one at a time,
  equivalence-tested, same discipline `nodes/optimizer/` already used.
- **`MemoryManager` everywhere.** Extend beyond `ChunkedScratchBufferStrategy`
  to anything else in `nodes/` that allocates a reusable device buffer.
- **A theoretically ideal training-pipeline design**, to use as a
  comparison target for what `nodes/` currently is vs. what it should
  become. Done -- see `docs/theoretical_pipeline_design.md` for the
  from-scratch design and the final gap analysis/prioritized backlog
  against everything below (that section incorporates all three passes,
  documented separately in `docs/theoretical_pipeline_design_iteration2.md`
  and `docs/theoretical_pipeline_design_iteration3.md`). Planning only;
  nothing in that backlog has been implemented yet.

## Optimizer subpackage (most mature part of `nodes/`)

`Algorithm` (pure per-parameter math: `CAMEAlgorithm`, `AdafactorAlgorithm`,
`AdamWAlgorithm`) x `ExecutionStrategy` (how updates get applied:
`SimpleLoopStrategy`, `ChunkedScratchBufferStrategy`, `ForeachApplyStrategy`),
composed by `ComposedOptimizerHandle`. Fused (backward-hook) execution
doesn't fit that same `step()`-driven contract, so it's a separate
`ComposedFusedOptimizerHandle` instead, reusing the same lifecycle methods
via subclassing. All three algorithms work with all three strategies and
with the fused handle -- proven, not just designed to.

**Verified, all real torch. Note which checks are deliberately CPU-only
by design (bit-exact/tolerance numerical checks don't need a specific
device to be meaningful) vs. which auto-detect and actually exercise
whatever hardware they're run on:**

| Component | How | Real XPU hardware |
|---|---|---|
| CAME/Adafactor `Algorithm`s, `SimpleLoopStrategy` + `ChunkedScratchBufferStrategy` | Equivalence-tested vs. `ChunkedXPUCAME`/`ChunkedXPUAdafactor` directly | **yes** -- user-confirmed on an Intel Arc B580, both strategies |
| `MemoryManager` | `smoke_test_memory_manager.py` | **yes** -- user-confirmed, including cross-step buffer reuse |
| AdamW `Algorithm`, all 3 strategies | Equivalence-tested vs. `CPUAdamW` directly (deliberately CPU-only -- see below) | **yes** for `ComposedAdamWOptimizerNode`'s own lifecycle test (`smoke_test_composed_adamw.py`, auto-detects device); the dedicated formula-equivalence test is CPU-only by design |
| `ForeachApplyStrategy` | Bit-exact vs. `SimpleLoopStrategy`, all 3 algorithms | Equivalence test is CPU-only by design (bit-exact comparison doesn't need a specific device); exercised via CAME/Adafactor's own composed lifecycle tests too |
| `ComposedFusedOptimizerHandle` + Adafactor | Equivalence-tested vs. `FusedXPUAdafactor` directly, real backward()/hooks, single- and multi-pass | Equivalence test is CPU-only by design; not yet separately run with device="xpu" |
| `ComposedFusedOptimizerHandle` + CAME/AdamW | No legacy reference exists; internal lifecycle + generalization checked | Lifecycle test is CPU-only by design; no legacy reference to check against on any device |
| axis-2 in-place scratch paths (CAME/Adafactor) | Bit-exact vs. out-of-place | not yet |

**Why several of these are "CPU-only by design", not an oversight:**
equivalence/bit-exactness is a property of the math, not the device --
`torch.equal()`/tolerance checks on CPU are the complete verification for
"does this formula match," and hardcoding CPU there keeps the tests fast
and fully reproducible without depending on whatever's plugged in.
Device-specific behavior (real VRAM lifecycle, `MemPool`, actual XPU
kernels) is what genuinely needs real hardware, and is exactly what
`smoke_test_composed_adamw.py`'s (and `smoke_test_composed_came.py`'s /
`smoke_test_composed_adafactor.py`'s) lifecycle tests auto-detect and
exercise for.

**Known, deliberate divergences from legacy** (not bugs -- documented
choices, made once and not re-litigated per file):
- `ForeachApplyStrategy` batches only the shared apply step (`param.data
  *= decay; param.data -= delta`), not each algorithm's own math -- see
  its module docstring for why legacy's own "foreach" classes don't fully
  batch either.
- `AdafactorAlgorithm` doesn't implement `FusedXPUAdafactor`'s/
  `ForeachXPUAdafactor`'s `TINY_NUMEL` (<10,000-element) special case --
  a real formula difference for small parameters (most individual LoRA
  matrices), not yet ported. Real algorithm-engineering work, not done.
- `ComposedFusedOptimizerHandle` does **not** replicate a real,
  pre-existing bug found in `FusedXPUAdafactor`'s momentum path: a
  float32-specific buffer-aliasing corruption (`g = self.exp_avg[i]`
  aliased, then `.to(dtype=p.dtype).mul_()` mutates it in place when
  `p.dtype` is already float32). Confirmed directly, not replicated on
  purpose -- see `smoke_test_fused_adafactor_equivalence.py`. A twin of
  this same bug exists in `ChunkedXPUAdafactor` too (see
  `docs/suspicious_findings.md`'s "Deferred" section) -- looks like a
  copy-pasted pattern across the legacy classes, not two independent bugs.
- `ComposedCAMEOptimizerNode`/`ComposedAdafactorOptimizerNode` default to
  `scale_parameter=False`, `weight_decay=0` -- safety-first defaults, not
  matching legacy's own defaults (`True`/`1.0`).
- Foreach/chunked/fused variants of one algorithm don't share a legacy
  reference across all three families evenly: CAME/AdamW have no legacy
  fused class to check against at all (only Adafactor does).

File index: `algorithms/{came,adafactor,adamw}.py`, `strategies/{simple,
chunked,foreach}.py`, `composed.py` (the `step()`-driven handle),
`composed_fused.py` (the hook-driven handle), `composed_{came,adafactor,
adamw}.py` and `composed_fused_{came,adafactor,adamw}.py` (the Node
wrappers), `handle.py` (`OptimizerHandle`/`FusedOptimizerHandle` ABCs).
Legacy-wrapping Nodes sit alongside these, unretired: `adamw.py`,
`adafactor.py`, `came.py`, `foreach_adafactor.py`, `foreach_came.py`,
`fused_adafactor.py`.
