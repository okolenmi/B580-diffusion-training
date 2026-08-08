# Session handoff (2026-08)

Written by Claude, for Claude -- there's no memory between conversations,
and the person working on this project has been through several sessions
now, each starting from zero. This is the note that session would have
wanted. Read this before `docs/training_pipeline_design.md` (the
architectural design doc, mostly a historical record now, see below) and
before `docs/suspicious_findings.md` (the live problems list, format
already established, just added four entries this session).

## Where the code actually is

The person applies patches manually (`git am`) to their own local clone
and pushes to GitHub (`okolenmi/B580-diffusion-training`) themselves --
GitHub is the real source of truth, not any sandbox state. A fresh
session has an empty sandbox; clone from GitHub first, don't assume
anything persists. Patches produced in a conversation live in
`/mnt/user-data/outputs/` *within that conversation only* -- they don't
carry over either.

Git identity isn't configured in a fresh sandbox clone --
`git config user.name "Claude"` / `git config user.email
"claude@anthropic.com"` first, or the first commit attempt errors.
Matches the identity every commit in this project's history already
uses.

## What's been done: all 12 numbered backlog items, complete

`docs/training_pipeline_design.md` section 10 laid out a 12-item
prioritized backlog for turning the `nodes/` package's design into real,
OOP, equivalence-tested code. As of this session, **all 12 are done and
applied** -- 13 patches total (12 feature items + one real bugfix, see
below), each its own reviewed commit:

1. `DiffusionProcess`/`DeviceContext` (`nodes/components/`)
2. `DeviceResident` ABC, `OptimizerHandle` retrofit (`nodes/memory/handle.py`)
3. `ParameterGroupPolicy` bugfix in `ComposedOptimizerHandle`
4. `LoRAScalingPolicy` (`nodes/model/lora_injector.py`)
5. Min-SNR v-prediction completed + `P2LossWeighting` (`nodes/train/loss.py`)
6. `ActivationCheckpointingStrategy` (`nodes/model/gradient_checkpointing.py`)
7. `ProjectLayout`, bridging `paths.py` (`nodes/components/layout.py`)
8. `FrozenWeightStore`/`AdapterStrategy` seam (`nodes/model/`)
9. `TrainingStepPipeline`/`StepPhase` refactor of `SupervisedLoRATrainerNode`
   -- the biggest single change, `nodes/train/step_pipeline.py`
10. `PrefetchingBatchSource` (`nodes/dataset/prefetch.py`)
11. `ResourceCoordinator`/`OffloadOrchestrator`, `TextEncoder`
    `DeviceResident` conformance (`nodes/memory/coordinator.py`)
12. (bugfix, not a backlog item) `AdamWOptimizerNode`'s class declaration
    had been silently swallowed by an earlier edit -- see "hard-won
    lessons" below.

The design doc's own "Further out" section (DoRAAdapter, NF4WeightStore,
`RescaledZeroTerminalSNRSchedule`+real v-prediction training,
`CheckpointPlacementPolicy`) is explicitly *not* part of this backlog --
each needs real profiling data, a larger validation effort, or a genuine
behavioral change to be worth building, not more refactoring. Don't
start any of these without the person asking for them specifically, and
expect them to need real training runs to validate, not sandbox testing.

## What's real vs. what's actually been validated

Everything above was built and tested in a CPU-only sandbox: equivalence
tests against the legacy code being replaced, hand-computed independent
references for the trickiest math (the step-pipeline refactor's
parameter-update check matched a hand-derived reference exactly), fake
hardware stand-ins for concurrency (the prefetch source's deadlock/
exception-propagation tests). That gives real confidence in the *logic*.
None of it had run on real XPU hardware or a real SDXL UNet until the
person's own test this session (see "Open problems" below) -- that's
the current frontier, not more refactoring.

## Open problems, from the person's own real training run

Four real issues reported after actually training with this. Full
writeups with dates are in `docs/suspicious_findings.md`'s "Open"
section -- this is the reasoning behind each, kept here rather than
bloating that file's normal format.

**1. Missing nodes in the palette (fully diagnosed, ready to fix).** Walk
every concrete `Node` subclass under `nodes/`, diff against
`server.nodegraph_registry.get_registry()`. Eight are missing:
`P2LossWeightingNode`, `PrefetchingBatchSourceNode` (both from this
session), and six `Composed*OptimizerNode` classes that predate it. Fix
is mechanical: add each to `server/nodegraph_registry.py`'s import list
and `classes` list. This should be close to the first thing done next
session -- small, zero risk, and directly blocks trying this session's
own `PrefetchingBatchSourceNode` at all.

**2. Training instability -- deformation and useful change appear at the
same LoRA power.** Settings: rank 48, alpha 1, dropout 0, weight_decay 0,
t range [150, 999], LR 1e-5, clip_threshold 1, 18000 steps. Candidates,
cheapest to try first, *none confirmed*:
- alpha=1/rank=48 gives `ClassicLoRAScaling`'s default ~0.021 scaling;
  rank 48 is high for LoRA, and alpha/rank scaling is known to suppress
  the learned update more as rank grows (Kalajdzievski, arXiv:2312.03732
  -- the exact motivation for `RankStabilizedScaling`, built this
  session, not the default, and not wireable from the graph editor UI
  without hand-writing graph JSON since nothing exposes a
  `scaling_policy` Port picker yet). Try a lower rank first (16-32) as
  the cheapest experiment, or try `RankStabilizedScaling`.
- dropout=0 + weight_decay=0 is zero regularization, combined with a
  fairly high rank and a lot of steps -- classic overfitting recipe,
  independent of any code question. Try weight_decay in 0.01-0.1.
- Loss weighting is presumably `UniformLossWeighting` (not mentioned).
  `MinSNRLossWeighting`/`P2LossWeighting` down-weight high-noise
  timesteps, which can reduce the influence of large noisy gradients.
- Double-check `nodes/dataset/renoise.py`/`managed.py`'s actual t-range
  semantics match what the UI label implies (inclusive/exclusive bounds,
  actual sampling distribution) rather than assuming.

Treat this as real experimentation once hardware time is available, not
a single fix.

**3. VRAM grows with mixed image dimensions, eventually crashes.**
Leading hypothesis: caching-allocator fragmentation from constantly
varying tensor shapes (not a genuine leak) -- the one real log line
available (`vram_allocated=5539MB` vs `vram_reserved=9802MB`, a ~4.3GB
gap) is mildly consistent with this but isn't proof by itself (would
need several consecutive steps, early vs. late in a run, to see whether
`reserved` keeps climbing). Try, cheapest first: `empty_cache_every_n_steps`
(already exists on `SupervisedLoRATrainerNode`, untried as far as this
session knows); check whether `manager/loader.py` does any aspect-ratio/
resolution bucketing already or whether batches are free to mix
arbitrary dimensions; check for an IPEX XPU allocator config analogous
to CUDA's `expandable_segments`.

Separately surfaced while thinking about this, worth its own look: every
`DeviceResident.footprint_bytes()` implementation from this session
(`nodes/optimizer/*.py`, `nodes/model/frozen_weight_store.py`,
`nodes/model/text_encoder.py`) sums `numel()*element_size()` without
checking which device the tensor is actually on. Nothing in
`SupervisedLoRATrainerNode` offloads mid-run today, so this shouldn't
matter *yet* -- but the one real log line shows `tracked_footprint_mb`
(6737MB) *higher* than `vram_allocated_mb` (5539MB), which is backwards
from what static-weight-only accounting vs. real allocated-including-
activations should show, and is worth actually understanding before
trusting `tracked_footprint_mb` for anything.

**4. Training is much slower than the pre-nodes/ implementation --
`optimizer_step` is ~66% of step time.** One real line: `forward=245ms
backward=287ms optimizer_step=1041ms total=1575ms`. Fairly confident
hypothesis given `clip_threshold=1` (an Adafactor/CAME-only parameter)
in the settings: a "Chunked" optimizer node
(`AdafactorOptimizerNode`/`CAMEOptimizerNode` -- real per-parameter
device syncs, built for VRAM-bounded *large*-parameter-count scenarios)
is in use instead of the matching "Foreach" variant
(`ForeachAdafactorOptimizerNode`/`ForeachCAMEOptimizerNode`, vectorized
via `torch._foreach_*`, and `nodes/optimizer/foreach_came.py`'s own
docstring says outright it's "the right default for LoRA's small, fixed
parameter set"). Check which node is actually wired first -- this is the
single most likely fix in this whole document. If a Foreach node is
already in use: check whether `core/trainer.py`'s pre-nodes/ path used a
different optimizer by default (a default mismatch, not a real
regression); get several consecutive steps' `profile=True` output, not
one line, to see if `optimizer_step` is consistent or spiking.

## Hard-won lessons from this session

**A class declaration can be silently swallowed by `str_replace`, and
`ast.parse()` won't catch it.** If `old_str` ends right before a `class
X:` line and `new_str` doesn't re-include it, that line vanishes and
everything meant to belong to the new class becomes extra members of the
*previous* class instead. This is syntactically valid Python -- a
docstring turns into a dangling expression statement, `INPUTS`/`build()`
become extra class attributes -- so a syntax check genuinely cannot
distinguish it from correct code. It happened three times in one commit
this session (`nodes/optimizer/came.py`, `foreach_adafactor.py`, caught
immediately by viewing the file right after each edit; `adamw.py`, not
caught the same way, shipped across 12 patches until it caused a real
500 on `/api/nodegraph/registry` and got traced back and fixed as its
own patch). **After any edit near a class boundary, `grep -n "^class "`
the file and actually look at the output** -- don't trust a clean
`ast.parse()` alone.

**Nothing in `nodes/smoke_tests/` exercises
`server/nodegraph_registry.py`'s actual class-loading path.** That's
exactly why the `AdamWOptimizerNode` bug shipped invisibly, and it's
exactly why the 8-missing-nodes finding above exists at all (found by
manually walking and diffing, not by a test). A single smoke test that
imports every registered class by name -- or better, does what this
session's manual check did (`pkgutil.walk_packages`, filter to concrete
`Node` subclasses, diff against `get_registry()`) -- would catch both
bug classes permanently and is small, targeted, cheap. Worth writing
early next session; explicitly *not* generic "add more tests," a
specific fix for a specific demonstrated blind spot.

**On testing discipline generally:** early in this session, testing
tended toward reflexive over-verification -- running the full 28+ file
smoke suite after small, clearly-scoped edits, or re-running several
files after a docstring-only change. The person pushed back on this
directly and repeatedly, and it was a fair correction: for a codebase
this size, changes this incremental, and a person who runs the real
suite on real hardware after every applied patch anyway, that reflex was
waste, not rigor. What actually held up as worth keeping: a real,
hand-computed independent-reference check for the step-pipeline
refactor's core math (this caught a real bug -- the `_ms` suffix
silently missing from the monitor report); real concurrency tests for
the prefetch source (deadlock-on-early-abandonment, exception
propagation -- also caught a real bug, silent exception loss in the
worker thread). The pattern that earned its keep: test the *highest-risk,
newest, least-obviously-correct* logic thoroughly, with real independent
verification, not toy "doesn't crash" checks -- and leave everything
else to the person's own full-suite runs on real hardware, which is
faster and more authoritative than anything achievable in this sandbox
anyway.

**`footprint_bytes()` doesn't check device placement anywhere in this
codebase.** Noted under problem 3 above -- flagging again here since
it's a design gap, not just a bug in one call site, and it'll recur
anywhere a `DeviceResident` might genuinely be offloaded mid-run (which
`OffloadOrchestrator`, once something actually publishes events to it,
will make a real, common case rather than a hypothetical one).

## About working with this person

Direct, no-fluff, technically sharp, reads code closely and calls out
sloppiness immediately and specifically (rightly). Wants real
correctness reasoning, not reassurance -- "don't try to look
professional instead of doing work" was said explicitly, early, and
meant. Fine with being told a limit was hit, a mistake was made, or
something is genuinely uncertain; unimpressed by hedging, apologizing,
or padding. Runs real training on real XPU hardware and treats that as
authoritative over anything a sandbox can show -- proposals to "just run
one more test" past the point of diminishing returns get pushed back on,
correctly. Wants patches delivered as real, applicable `git am` patches,
one coherent commit per logical change, with commit messages that
explain *why*, not just *what* -- matches this project's own existing
git history style, worth continuing exactly as-is.
