# Optimizer execution redesign — plan, 2026-08-09

Written after a real profiling report (`optimizer_step=967ms` vs.
`forward=250ms`+`backward=308ms`, CAME) plus follow-up from the person
that changes the diagnosis: `ForeachCAMEOptimizerNode` and
`ComposedCAMEOptimizerNode` show "almost the same performance" as each
other. That rules out the host-device-sync explanation (`ForeachXPUCAME`
in `core/optimizers.py`, read directly, has none) as the dominant cost,
and points at something structural shared by every current CAME path.
Separately: the VRAM reserved/allocated gap (9802MB vs 5539MB) was
measured on a **uniform 512x512 dataset** — no shape variation at all —
which rules out the mixed-image-dimension hypothesis as the explanation
for *that specific* gap (still worth testing on its own merits, just not
provenly the cause here).

Person's instruction for this round: most effective fix, not cheapest.
Long refactor chains are fine. This doc is the plan; no code changes yet
— implementation happens in follow-up patches once this is reviewed.

## Confirmed by reading the code directly (not assumed)

1. **No CAME implementation in this codebase does real multi-tensor
   batching of its actual math**, for any of the four variants
   (`ChunkedXPUCAME`, `ForeachXPUCAME` — both `core/optimizers.py` — and
   `ComposedCAMEOptimizerNode`'s `simple`/`chunked`/`foreach` strategies).
   `CAMEAlgorithm.compute_update()` (`nodes/optimizer/algorithms/came.py`)
   runs strictly one parameter at a time, by its own docstring.
   `ForeachApplyStrategy` (`nodes/optimizer/strategies/foreach.py`) only
   batches the *final* apply (`param -= delta`, one or two
   `torch._foreach_*` calls per (device, dtype) group) — everything
   before that (row/col normalize, clip, momentum, confidence residual,
   ~15-20 elementwise ops) still runs per-parameter, in a plain Python
   loop, in every strategy including "foreach".

2. **`ForeachXPUAdafactor`'s factored path (`_step_factored`,
   `core/optimizers.py`) is *also* a plain per-parameter loop** — zero
   `torch._foreach_*` calls. Only its unfactored (1D) path gets real
   batching. Every LoRA adapter weight is 2D, so **nothing in this
   codebase currently gets true multi-tensor batching for the shape LoRA
   training actually uses.** This was already known, precisely, in
   `nodes/optimizer/strategies/foreach.py`'s own docstring: *"the legacy
   ForeachXPUAdafactor's own 'factored' path already falls back to a
   plain per-parameter loop for exactly this reason."*

3. **This gap was already named as a real limitation, not discovered
   fresh here.** `algorithms/base.py`'s module docstring states it
   directly: `core/optimizers.py`'s four hand-written classes are "2
   algorithms x up to 3 memory strategies, hand-crossed, with CAME only
   getting 1 of the 3 possible strategies because writing each
   combination by hand is expensive" — and that the
   Algorithm/ExecutionStrategy split exists specifically to stop needing
   to hand-write that grid. The batched strategy this plan proposes is
   the follow-up work that same docstring already flagged as not yet
   done ("the memory-optimized strategies are real, valuable, separate
   follow-up work once the split itself is validated").

4. **`ForeachXPUCAME`'s per-step memory churn is already fairly
   controlled** — state buffers (`vr`/`vc`/`vs`/`exp_avg`/`res_r`/`res_c`)
   are allocated once and reused in-place across steps; the working
   tensor `g` is mutated in-place for normalize/clip/momentum/residual.
   Real per-step allocations are small: one `.float()` cast (only if the
   gradient isn't already fp32) and one `g2` temp, per parameter, per
   step. By contrast, `ComposedCAMEOptimizerNode`'s default `strategy`
   is `"simple"` (`SimpleLoopStrategy`), which never passes `scratch`,
   so `CAMEAlgorithm._compute_update_safe()` runs — genuinely
   out-of-place, allocating **four-plus fresh full-parameter-sized
   tensors per parameter, every single step, forever.** If VRAM was
   profiled while wired to `ComposedCAMEOptimizerNode` without
   explicitly setting `strategy="chunked"`, this is a real, distinct
   contributor, independent of dataset shape.

## Open unknowns that gate parts of this plan

- Which exact optimizer class produced the `967ms`/VRAM numbers —
  `ForeachCAMEOptimizerNode`, or `ComposedCAMEOptimizerNode` at which
  `strategy`? The print-on-construction lines
  (`core/optimizers.py`) or an explicit log of `strategy=...` answer
  this directly; not yet confirmed which.
- Whether the VRAM reserved/allocated gap is *growing* over a run or
  stable-but-high. Every number seen so far is a single snapshot.
- Whether periodic preview generation runs during the profiled window
  (a plausible independent VRAM contributor — different resolution/batch
  shape than training, unrelated to dataset uniformity).
- Whether `PYTORCH_ALLOC_CONF=expandable_segments:True` (or
  `max_split_size_mb`) does anything measurable on this XPU build.

None of these block starting Phase 0/1 (below) — they're exactly what
Phase 0/1 is for.

## Phase 0 — instrumentation (do first, cheap, unblocks everything else)

This project has now hit "only one VRAM data point, can't tell if it's
growing" twice in a row across sessions. Fix the tool, not just the one
measurement:

- **`SupervisedLoRATrainerNode`'s `profile` output**: add
  reserved/allocated *deltas* — vs. step 1's numbers, and vs. the value
  at the last preview generation — alongside the existing absolute
  numbers. Turns every future run into a trend automatically instead of
  requiring a person to manually diff snapshots across messages.
- **Shape histogram diagnostic**: a small standalone utility (or a debug
  branch under `--profile`) that, given a built LoRA graph, prints
  `{shape: count}` across all trainable parameters before training
  starts. Directly sizes the expected win from Phase 2's grouping before
  investing in it — if a real SDXL LoRA config turns out to have mostly
  unique shapes, the whole batching plan below needs rethinking; if (as
  expected, given how UNet attention blocks repeat the same
  in/out-feature widths) it produces a modest number of shapes each
  shared by many parameters, that's the confirmation to proceed.
- **Log which optimizer class + strategy is active** somewhere more
  durable than a construction-time print — e.g. included in
  `monitor.report()` alongside the existing per-phase timings, so it's
  in the same place as the numbers it explains, not just the console
  scrollback.

## Phase 1 — diagnostics (cheap, real hardware, before committing to Phase 2's design)

Run with Phase 0's tooling in place:

1. `ForeachCAMEOptimizerNode` with `verbose_profile=True` for one run —
   real phase breakdown (cast/normalize/clip/momentum/update) with
   syncs between each. Confirms whether cost is spread evenly (supports
   "no batching, many small launches, structural") or concentrated
   (would point somewhere more specific).
2. Same config, `ComposedAdamWOptimizerNode` vs `ComposedCAMEOptimizerNode`
   — isolates "CAME's math is heavier" from "everything here is
   loop-bound regardless of algorithm."
3. Multi-point VRAM logging across one run (start/50/100/200 steps),
   with and without periodic preview generation, to separate "steady
   high baseline" from "actually climbing" and to test the preview
   hypothesis directly instead of by inference.
4. `ComposedCAMEOptimizerNode` with `strategy="chunked"` explicitly vs.
   default `"simple"` — tests finding #4 above directly: does forcing
   the in-place scratch path reduce the reserved/allocated gap even
   without any of Phase 2's work?
5. Try `PYTORCH_ALLOC_CONF=expandable_segments:True`, then
   `max_split_size_mb:512` if the first does nothing, both against a
   fixed repro (uniform 512x512, several hundred steps).

Phase 2's design (below) doesn't depend on these results — it's
justified either way — but Phase 4 (VRAM fixes) does, so run this in
parallel with Phase 2, not strictly before it.

## Phase 2 — the core redesign: shape-grouped batched execution

The actual "most effective, not cheapest" fix for speed. Goal: turn
"100+ individual per-parameter update chains, ~15-20 kernel launches
each" into "one batched chain per *distinct parameter shape*, a handful
of kernel launches each" — without weakening the existing bit-exact
verification standard this codebase already holds itself to.

**Design**, concretely enough to implement without re-deriving it:

- At optimizer construction, group all trainable parameters by
  `(shape, dtype, device)`. Stable for the whole run — LoRA parameter
  shapes don't change after injection.
- For each group of size k>1: stack that group's per-parameter state
  buffers along a new leading group axis at init (e.g. CAME's `r`
  becomes shape `(k, rows)` instead of k separate `(rows,)` tensors,
  `ea` becomes `(k, rows, cols)`). One tensor per buffer type per group,
  not one per parameter.
- Each step: stack the group's k gradients into one `(k, rows, cols)`
  tensor (a real copy — grads live in separate autograd-owned storages,
  can't be aliased away; this is one allocation *per group*, not per
  parameter, still a large reduction from today). Run the *entire* CAME
  update — row/col means via `.mean(dim=2)`/`.mean(dim=1)`, sqrt,
  normalize, clip via `.norm(dim=(1,2))`, momentum, confidence residual,
  second normalize — as batched ops across the leading axis. A handful
  of kernel launches for the **whole group**, not per member.
- Apply step: split back to per-parameter tensors for the actual
  `param.data -= update` (autograd parameters can't be restrided into
  shared storage), via `torch._foreach_sub_` across the group — same
  pattern `ForeachApplyStrategy` already uses correctly.
- Groups of size 1 (no shape siblings) fall back to the existing
  per-parameter path unchanged — no regression for the ungrouped tail,
  and the design degrades gracefully if Phase 0's shape histogram turns
  out less favorable than expected.
- Deliberately **exact-shape grouping only, no padding of near-matching
  shapes.** Padding-based batching is how some other frameworks do this
  more aggressively, but it's a real extra source of correctness risk
  (masking, gradient leakage through pad regions) for a win this design
  already gets most of — real UNet target modules repeat exact
  in/out-feature widths constantly (every q/k/v/out projection at a
  given attention width, across many blocks), so exact-shape grouping
  should already collapse "100+ parameters" down to "a modest number of
  groups," per Phase 0's histogram once it's run for real.

**New interface needed**: `Algorithm` gets a batched counterpart to
`compute_update()` — something like
`compute_update_batched(self, grad_stack, state_stack, lr_stack,
scratch=None)` — operating on the `(k, ...)` stacked tensors. This is a
real rewrite of `CAMEAlgorithm`'s math with an extra leading batch
dimension threaded through every reduction, not a strategy-only change
— algebraically direct (CAME's row/col reductions are already
per-parameter-independent, this just adds one axis) but every line
needs the same "bit-exact via `torch.equal()`, not tolerance" check this
codebase already applies to every past restructuring
(`smoke_test_composed_came.py`, `smoke_test_foreach_strategy_equivalence.py`
are the direct precedents — same pattern, one more axis).

**New `ExecutionStrategy`**: call it `ShapeGroupedBatchStrategy`
(`nodes/optimizer/strategies/`), built once, generic — same "one
strategy works with any Algorithm" contract every existing strategy
already follows. Benefits CAME first (built and validated here since
it's the heaviest math and the current worst performer), then Adafactor
and AdamW for free once they implement the batched interface too
(Phase 5).

**Status labeling once built** — matching this project's own existing
convention exactly (see `composed_came.py`'s docstring): start
`"shape_grouped"` as *equivalence-verified only*, promote to *real-
hardware validated* only after an actual run on the person's B580, same
gate every other strategy here already goes through. Not shipped as the
default until that gate passes.

## Phase 3 — rollout for CAME

1. Add `strategy="shape_grouped"` as a new option on
   `ComposedCAMEOptimizerNode`, default stays `"simple"` until validated.
2. Bit-exact equivalence tests: `ShapeGroupedBatchStrategy` vs.
   `SimpleLoopStrategy`, group sizes 1/2/many, float32 and bf16 — same
   rigor and same file-naming convention as the existing
   `smoke_test_*_equivalence.py` files.
3. Real-hardware run: same profiling setup as the original report
   (`profile=True`, same rank/resolution/batch), compare
   `optimizer_step` and reserved/allocated directly against the
   `967ms`/`9802MB` baseline. This is the number that actually answers
   whether the plan worked — everything before this step is reasoned,
   not measured.
4. If validated: flip `ComposedCAMEOptimizerNode`'s default to
   `"shape_grouped"`, update `docs/suspicious_findings.md`'s optimizer
   entries accordingly.

## Phase 4 — VRAM fixes (parallel to Phase 2/3, informed by Phase 1)

Concrete fix depends on which Phase 1 result actually lands:

- If it's optimizer per-step churn (finding #4, `SimpleLoopStrategy`'s
  out-of-place path): Phase 2's batched strategy already fixes this as
  a side effect (persistent group-level buffers, not fresh
  per-parameter allocations every step) — no separate fix needed beyond
  landing Phase 2. In the meantime, forcing `strategy="chunked"` is a
  same-day mitigation that doesn't wait on the bigger redesign.
- If it's preview generation: give preview's VAE decode its own scoped
  memory handling — bracket it with `torch.xpu.empty_cache()`
  before/after, or route it through its own `torch.xpu.MemPool()` the
  way `ChunkedXPUCAME`'s scratch buffer already does for a different
  purpose, so it stops competing with training's own allocator segments
  for space.
- If `PYTORCH_ALLOC_CONF` measurably helps: make it a documented default
  in the launch scripts, not just a one-off manual export.
- Regardless of root cause: wire `empty_cache_every_n_steps` (already
  built, `nodes/train/supervised.py`) to a sensible non-zero default
  once Phase 1's data shows what "sensible" means for this workload —
  it's a real, cheap safety margin against the near-ceiling crash risk
  even if it's not the primary fix for whatever's causing the gap.

## Phase 5 — extend batched execution to Adafactor and AdamW

Once `ShapeGroupedBatchStrategy` is validated on CAME (the heaviest
math, so the hardest case), give `AdafactorAlgorithm` and
`AdamWAlgorithm` the same batched interface. Smaller expected win per
algorithm (their per-parameter math is already shorter than CAME's
double row/col normalization), but removes the *same* structural
ceiling from the two other real optimizer paths, not just CAME's.

## Phase 6 — retire redundant legacy paths

Once the new strategy is real-hardware-validated and promoted to
default for all three algorithms: retire `ChunkedXPUCAME`,
`ForeachXPUCAME`, `ChunkedXPUAdafactor`, `ForeachXPUAdafactor`
(`core/optimizers.py`), and `SimpleLoopStrategy`/`ChunkedScratchBufferStrategy`/
`ForeachApplyStrategy` as anything other than an explicit, documented
fallback. This directly follows through on what `composed_came.py`'s
own docstring already said was the plan ("once ... real-hardware
validated ... the legacy path can eventually be retired -- a
deliberate, separate later step, not this one" — this is that step).
Real payoff: right now "CAME optimizer" doesn't map to one obvious node
in this codebase — it's four separate implementations spread across two
files with different validation status each. Collapsing that to one
validated path per algorithm is a real reduction in the kind of
confusion this exact conversation started from ("It was Foreach or
Composed" — two different answers to what should be one question).

## Phase 7 — docs cleanup

Update `docs/nodes_package_design.md`'s Algorithm/ExecutionStrategy
section and `docs/suspicious_findings.md`'s optimizer-speed and
VRAM-near-ceiling entries to reflect whatever Phase 3/4 actually
measured — real numbers, not the hypotheses in this doc, once they
exist.

## Sequencing summary

```
Phase 0 (instrumentation)  ─┬─> Phase 1 (diagnostics) ─┬─> Phase 4 (VRAM fixes)
                             │                          │
                             └─> Phase 2 (batched design, CAME) ─> Phase 3 (rollout, CAME)
                                                                        │
                                                                        v
                                                          Phase 5 (extend: Adafactor, AdamW)
                                                                        │
                                                                        v
                                                          Phase 6 (retire legacy paths)
                                                                        │
                                                                        v
                                                              Phase 7 (docs)
```

Phase 0 blocks everything (it's what makes every later measurement
trustworthy). Phase 1 and Phase 2 can run in parallel — Phase 2's design
is justified independent of Phase 1's specific findings. Phase 4 is
gated on Phase 1's actual result, not on Phase 2 landing, except for the
one case where Phase 2 already fixes it as a side effect.
