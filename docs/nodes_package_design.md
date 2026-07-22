# The `nodes/` package: design

**Read this section first if you're a fresh session picking this up with
no memory of how it got here.** Everything below it is the design
reasoning and decision history, kept because it explains *why* things are
shaped this way -- worth reading if you need that context for a specific
decision, but not required just to keep going.

## Start here: current state

**What this is:** a from-scratch, node-graph-shaped rewrite of the
optimizer subsystem, developed *alongside* the existing, working
`core/optimizers.py` -- not a replacement yet. `core/optimizers.py` has
not been modified by any of this and remains exactly as it was
(confirmed byte-identical to the pre-`nodes/` state at every step of this
work -- see "Course correction" below for why that rule exists).

**Real file list** (regenerate with `find nodes -name "*.py" -not -path
"*__pycache__*" | sort` rather than trusting a hand-maintained list here,
which will drift):
```
nodes/core.py                              Port, Node (ABC) -- domain-independent
nodes/memory/manager.py                    MemoryManager -- domain-independent, centralized
                                            device-buffer acquire/release/free (see "Centralized
                                            memory management" section below)
nodes/optimizer/handle.py                  OptimizerHandle, FusedOptimizerHandle (ABCs)
nodes/optimizer/node.py                    OptimizerNode (ABC) -- intermediate layer
nodes/optimizer/{adafactor,came,foreach_adafactor,fused_adafactor,adamw}.py
                                            5 Nodes wrapping core.optimizers.* (Adapter
                                            pattern -- no math reimplemented)
nodes/optimizer/algorithms/base.py         Algorithm (ABC) -- pure per-parameter math
nodes/optimizer/algorithms/came.py         CAMEAlgorithm -- FRESH reimplementation,
                                            re-verified against the reference (not a
                                            wrapper)
nodes/optimizer/strategies/base.py         ExecutionStrategy (ABC)
nodes/optimizer/strategies/simple.py       SimpleLoopStrategy -- real-hardware validated
nodes/optimizer/strategies/chunked.py      ChunkedScratchBufferStrategy -- real-hardware
                                            validated, now MemoryManager-backed for real
                                            cross-step buffer caching (see "two-axis
                                            distinction" below)
nodes/optimizer/composed.py                ComposedOptimizerHandle -- generic glue for
                                            any Algorithm + any ExecutionStrategy
nodes/optimizer/composed_came.py           ComposedCAMEOptimizerNode -- CAME algorithm,
                                            selectable strategy ("simple" or "chunked")
nodes/smoke_tests/smoke_test_composed_came.py   Real-hardware test, run with:
                                            `python nodes/smoke_tests/smoke_test_composed_came.py`
nodes/smoke_tests/smoke_test_memory_manager.py  Real-torch (CPU) test for MemoryManager, run with:
                                            `python nodes/smoke_tests/smoke_test_memory_manager.py`
```
Also: `server/nodegraph_introspect.py` + `server/routes_nodegraph.py` +
`server/static/nodegraph.html` (the `/nodegraph` dev tab, reads real
declared contracts off these classes -- outside `nodes/` itself but part
of the same effort).

**Verification status, precisely (don't take "it's built" to mean "it's
proven" -- check which level a given piece has actually reached):**
| Piece | Numpy-mock verified | Real XPU hardware verified |
|---|---|---|
| 5 legacy-wrapping wrapper Nodes (adafactor/came/foreach/fused/adamw) | via mock legacy classes | not yet |
| `CAMEAlgorithm` formulas | yes (bounded ~1e-8, see below) | via the composed test, yes |
| `SimpleLoopStrategy` | yes | **yes** -- user-run, passed (97.7% loss reduction, all lifecycle methods, offload/reload round trip) |
| `ChunkedScratchBufferStrategy` | yes, bit-exact vs. `SimpleLoopStrategy` | **yes, but predates the MemoryManager refactor below** -- the user's real-XPU run validated the earlier fresh-`torch.empty()`-per-step version; the MemoryManager-backed version has only been run on CPU so far (behaviorally identical `step()` math, same bit-exact-vs-`simple` guarantee, but the caching/offload logic itself needs its own real-hardware confirmation, not just an inherited pass) |
| `MemoryManager` | n/a -- pure allocator logic, no device-specific code path exists | real torch (CPU) -- `smoke_test_memory_manager.py`, all checks pass; real XPU run not yet done, but nothing in the class is XPU-specific |

**The one open architectural question, not yet answered:** does the
Algorithm/ExecutionStrategy split (see below) generalize cleanly to a
second algorithm (Adafactor) and a third strategy (fused-backward-hook)?
Only ever built/tested with one algorithm (CAME) and two of the
simpler strategies so far -- the design is reasoned to generalize (see
"Fused optimizer family" section) but that's not the same as having
built the second data point yet.

**Concrete next step, in order of what unblocks the most:**
1. Build `AdafactorAlgorithm` (a second `Algorithm` -- tests whether the
   split genuinely generalizes, or whether something CAME-specific
   leaked into the abstractions).
2. Re-run `smoke_test_composed_came.py --strategy chunked` on real XPU
   hardware to confirm the `MemoryManager`-backed version of
   `ChunkedScratchBufferStrategy` (see "Centralized memory management"
   below) behaves the same as the pre-refactor version that was
   originally validated there -- the CPU run already confirms the logic
   is correct, but not the actual VRAM behavior on real hardware.
3. *Then* decide whether `ComposedCAMEOptimizerNode` should start
   replacing the legacy-wrapping `CAMEOptimizerNode` for real use, or
   whether to keep building breadth (more algorithms/strategies) first --
   this is a judgment call worth surfacing to the user rather than
   assuming, since it trades "prove the design generalizes" against
   "get something production-usable sooner."
4. Longer-term, deferred, real but not urgent: `torch.xpu.MemPool`
   integration (now a single, well-defined seam inside
   `MemoryManager.get_buffer()` -- see below), `Algorithm.compute_update`
   actually using its `scratch` hint (axis 2 of the two-axis distinction
   below), a `FusedBackwardHookStrategy`, and eventually wiring `nodes/`
   into the actual training pipeline (`core/trainer.py` currently knows
   nothing about any of this).

**Other docs, and how they relate:**
- `docs/node_architecture_refactor_plan.md` -- the original, higher-level
  plan (why a node-graph architecture at all, the `/nodegraph` playground's
  UI-performance design constraints). Read if the *why build this at all*
  question comes up; this document (`nodes_package_design.md`) is the
  detailed follow-through on its "Phase 1: optimizer nodes" section.
- `docs/suspicious_findings.md` -- unrelated to `nodes/` entirely; tracks
  real bugs/investigations elsewhere in the training pipeline (composition
  destruction in LoRA training, a VRAM leak around preview generation).
  Worth knowing it exists, not required reading for `nodes/` work.

---

## Course correction that produced this document

An earlier attempt this session tried to retrofit a shared base class
directly into `core/optimizers.py` (extracting genuinely duplicated
lifecycle code -- `offload_states_to_cpu`/`reload_states_to_device`/
`decay_states`/`reset_states`/`free_states` -- into one base class the 5
existing optimizer classes would inherit from). That work was reverted,
confirmed byte-identical to the known-good remote state, before this
document was written. Two reasons, both worth stating plainly so they don't
get relitigated later:

1. **`core/optimizers.py` is live, already-verified production code.**
   `ChunkedXPUCAME`'s math was numerically checked against the official
   reference implementation earlier this session; the opt-step counting
   and SNR-weighting fixes elsewhere in the codebase were similarly
   hard-won. Restructuring inheritance in that file, however carefully,
   means every one of those verified properties needs re-verifying against
   the new shape -- real risk for a benefit (cleaner internal duplication)
   that doesn't require touching that file at all to achieve.
2. **The old code was never designed with a node-graph interface in mind.**
   Retrofitting one onto it means fighting its existing shape at every
   step (as the reset-vs-free asymmetry discovered mid-refactor
   demonstrated -- `_scratch`/`_pool` cleared in `free_states()` but not
   `reset_states()` in some classes, `remove_hooks()` similarly asymmetric
   in `FusedXPUAdafactor`; each one a real, easy-to-miss trap when forcing
   old, organically-grown code into a new shared shape it wasn't written
   for). A clean-room design in a new location doesn't have to fight any
   of that.

**Resulting rule for this package: `nodes/` never edits anything under
`core/`, `manager/`, or `server/` (except read-only imports). It wraps
existing, verified code via composition (the Adapter pattern), and is
free to define its own clean interfaces without being constrained by
what the old code happens to already look like.**

## The abstraction hierarchy

Four distinct concepts, deliberately kept separate rather than collapsed
into one "Node does everything" class -- collapsing them is exactly the
kind of shortcut that produces code average training data leans toward and
that this design is explicitly working against.

### 1. `Port` -- declarative port metadata

```python
@dataclass(frozen=True)
class Port:
    name: str
    type: type
    required: bool = True
    default: Any = None
    doc: str = ""
```

Pure data. No behavior. Describes one named input or output slot.

### 2. `Node` (ABC) -- the universal node contract

```python
class Node(ABC):
    INPUTS:  ClassVar[dict[str, Port]] = {}
    OUTPUTS: ClassVar[dict[str, Port]] = {}

    @abstractmethod
    def build(self, **inputs) -> dict[str, Any]:
        """Given values for INPUTS, produce a dict covering OUTPUTS."""
```

This is the only thing every node in the system has in common: a declared,
inspectable set of typed input/output ports, and a way to turn input values
into output values. Deliberately *not* specified here: whether building is
cheap/pure or expensive/stateful, what "running" a node beyond `build()`
means for something long-lived like a training loop -- those are concerns
for domain-specific layers below, added only when a concrete domain
actually needs them, not speculatively baked into the universal base.

`INPUTS`/`OUTPUTS` are **declared, not derived**. This is a deliberate
departure from the earlier read-only playground (which used
`inspect.signature()` to *guess* a class's ports from its constructor).
Guessing is fine for displaying already-existing, non-node-aware code
read-only, which is what that playground was for. It is not fine as the
actual interface contract for newly-designed node code -- a declared
contract can be validated (a node's `build()` can check its own inputs and
outputs against what it declared, and `__init_subclass__` can refuse to
define a concrete node that forgot to declare `INPUTS`/`OUTPUTS` at all),
where a guessed one can only ever be descriptive, never enforced.

### 3. Domain-family ABCs -- the "intermediate node" layer

This is the layer that directly answers "these classes have a big similar
part that can be intermediate node." One example, worked through fully
below: `OptimizerNode`. The general shape, for any future domain
(`TeacherSourceNode`, `DataSourceNode`, ...): a domain ABC extends `Node`,
fixes the `OUTPUTS` shape shared by the whole family (every optimizer node
produces exactly one `optimizer` port, typed as the family's own runtime
contract -- see `OptimizerHandle` below), and may declare a `COMMON_INPUTS`
dict that concrete members merge into their own `INPUTS` rather than
re-declaring shared ports by hand in every subclass.

### 4. Runtime "Handle" ABCs -- separate from the Node that builds them

A `Node` represents a *construction* step (config in, object out) in the
graph. The object it constructs is a different concern with its own
interface, used at a different time (during actual training, not graph
setup) -- keeping these separate is a real, load-bearing distinction, not
just a stylistic split. Concretely: `OptimizerNode.build()` returns an
`OptimizerHandle` -- a formal ABC declaring exactly the methods this
codebase's optimizers are actually called through elsewhere
(`step`, `zero_grad`, `offload_states_to_cpu`, `reload_states_to_device`,
`decay_states`, `reset_states`, `free_states`, `.lr`). This mirrors the
Builder/Product (or Factory/Product) pattern: the Node is the builder, the
Handle is the product, and neither needs to know the other's internal
shape beyond this contract.

## Worked example: the optimizer domain

Chosen first because it's the domain this session already has the deepest,
most rigorously-verified understanding of.

This was the *original* file layout, from when only the legacy-wrapping
layer existed -- kept here as-is since it matches the prose right below it
(which is specifically about that layer). `algorithms/`, `strategies/`,
`composed.py`, `composed_came.py`, and `smoke_tests/` came later -- see
"Start here" at the top of this document for the complete, current list.

```
nodes/
  __init__.py
  core.py                 # Port, Node (ABC) -- domain-independent
  optimizer/
    __init__.py
    handle.py              # OptimizerHandle (ABC) -- the runtime contract
    node.py                 # OptimizerNode (ABC) -- the intermediate layer
    adafactor.py            # AdafactorOptimizerNode + its Handle adapter
    came.py                  # CAMEOptimizerNode + its Handle adapter
    foreach_adafactor.py      # ForeachAdafactorOptimizerNode + its Handle adapter
    fused_adafactor.py         # FusedAdafactorOptimizerNode + its Handle adapter
    adamw.py                    # AdamWOptimizerNode + its Handle adapter
```

Each concrete `*OptimizerNode.build()` constructs a small `*Handle` adapter
class that **wraps** (holds a reference to, delegates every call to) the
corresponding already-verified `core.optimizers.*` instance -- e.g.
`CAMEOptimizerHandle` wraps a real `core.optimizers.ChunkedXPUCAME`. No
optimizer math is reimplemented anywhere in `nodes/`; every adapter is a
thin pass-through. This is the concrete mechanism that satisfies "no risk
of exploding the working solution": the verified numerical behavior lives
in exactly one place (`core/optimizers.py`, untouched), and the new code's
only job is presenting it through a clean, declared, checkable interface.

**A genuine, honest bonus, not the point of the exercise but worth noting
since it fell directly out of doing this properly:** `CPUAdamW` (the legacy
class backing `AdamWOptimizerNode`) doesn't actually implement
`decay_states`/`reset_states` at all -- confirmed earlier this session,
and a real latent bug (`trainer.py` calls `optimizer.decay_states(...)`
unconditionally in cyclic-tuning mode; combining `optimizer="adamw"` with
cyclic tuning would crash with `AttributeError`, apparently never
exercised together so far). Since `OptimizerHandle` declares
`decay_states`/`reset_states` as required abstract methods, Python's `abc`
machinery *refuses to let `AdamWOptimizerHandle` be instantiated at all*
unless it implements them -- so the adapter is forced to provide a real,
new, correct implementation. That implementation is new code, written and
verified fresh (not inherited from or copy-pasted out of the old class),
living entirely in the adapter -- `core/optimizers.py`'s `CPUAdamW` itself
is never touched and remains exactly as it was for any other current
caller. This is what "clean base, no risk to the old solution" concretely
buys: a real bug gets a real fix, without the fix's correctness depending
on successfully modifying code that's already relied upon elsewhere.

## Implemented this round: 4 of 5 optimizers, and why not the 5th

`AdafactorOptimizerNode`, `CAMEOptimizerNode`, `ForeachAdafactorOptimizerNode`,
`AdamWOptimizerNode` are implemented and verified (see below).
`FusedXPUAdafactor` is deliberately **not** wrapped this round.

Reason, found while writing the adapters rather than assumed going in:
`core/train_step.py` drives `FusedXPUAdafactor` through a genuinely
different execution protocol from the other four -- `optimizer.
begin_step(sub_steps=n_passes)` and `optimizer.prepare_next_pass()` get
called at specific points in the per-micro-step loop, detected via a literal
`isinstance(optimizer, FusedXPUAdafactor)` check, because its actual
parameter updates happen inside backward hooks rather than in a single
explicit `step()` call the way the other four work. Wrapping it under the
same plain `OptimizerHandle` used by the other four and quietly dropping
`begin_step`/`prepare_next_pass` would produce a Handle that *looks*
interchangeable but silently doesn't function -- exactly the kind of
false-equivalence a real interface is supposed to prevent, not produce.
The honest options are (a) extend the contract -- a
`FusedOptimizerHandle(OptimizerHandle)` adding `begin_step`/
`prepare_next_pass` as further required methods, with the training loop
(once it's actually wired to use `nodes/`, a later phase) checking
`isinstance(handle, FusedOptimizerHandle)` instead of the current
concrete-class check -- or (b) treat fused/backward-hook-based optimizers
as a genuinely different node family entirely. Left as an open design
question rather than resolved by force-fitting it into today's contract.

## Verification performed (no torch available in this environment)

Every adapter's `build()` and its Handle's delegation were tested against
hand-written mock legacy classes replicating the real ones' exact
structural behavior (constructor signature, which attributes exist,
whether `reset_states`/`decay_states` exist as real methods) -- not just
compiled, actually exercised:

- `CAMEOptimizerHandle`: constructed via `build()` with only the required
  `params` input, confirmed every optional input correctly falls back to
  its declared default (including the CAME-specific `lr=1e-4` override on
  top of `OptimizerNode.COMMON_INPUTS`'s generic `lr=1e-5` default,
  confirming the merge-and-override pattern works) and confirmed
  `step`/`zero_grad`/`decay_states`/`update_lr` correctly delegate to the
  wrapped instance.
- `AdafactorOptimizerHandle`: same delegation check across all seven
  `OptimizerHandle` methods in one pass.
- `ForeachAdafactorOptimizerHandle`: specifically verified `reset_states()`
  -- which doesn't exist as a method on the real legacy class -- correctly
  routes through `decay_states(0.0)` instead (confirmed by reading the real
  `ForeachXPUAdafactor.decay_states`'s body: its `factor<=0` branch already
  does the identical null-out-all-state operation inline).
- `AdamWOptimizerHandle` -- the one adapter containing genuinely new logic,
  not just delegation, so given the most scrutiny: the mock's `step()`
  method replicates the real `CPUAdamW.step()`'s important property
  exactly -- it calls `self.m[i].mul_(...)` with **no None-guard**, unlike
  the GPU optimizers' lazily-populated state. Confirmed the critical
  correctness property this adapter depends on: after `reset_states()`,
  `step()` still runs without crashing, because the adapter zeroes the
  tensors in place (`m[i].zero_()`) rather than nulling them
  (`m[i] = None`) -- the latter would raise `AttributeError` on the very
  next `step()` call, exactly reproducing the real bug this adapter exists
  to fix, if gotten wrong. Also verified `decay_states(0.0)` correctly
  routes to the same zero-in-place reset path.

All of the above ran directly, with real assertions checked and printed --
not reasoned about in the abstract.

## The fused optimizer family, and the honest answer about "Fused CAME"

`FusedXPUAdafactor` genuinely doesn't fit the plain `OptimizerHandle`
contract -- confirmed by reading its real implementation, not assumed:
`step()` and `zero_grad()` are literal no-ops (`pass` bodies); every actual
parameter update happens inside `_update_param`, a per-parameter method
triggered by a `register_post_accumulate_grad_hook` registered once (via
`register_hooks()`, called exactly once in the old code, from
`core/trainer.py`, right after construction) for every trainable
parameter. The real per-micro-step lifecycle a caller drives is
`begin_step(sub_steps=)` before the backward pass(es) that make up one
logical update, and `prepare_next_pass()` between accumulated backward()
calls when `sub_steps > 1`.

**`FusedOptimizerHandle(OptimizerHandle)`** formalizes this: a real
subtype (verified: `isinstance(fused_handle, OptimizerHandle)` is `True`,
so anything written against the generic contract still works) adding
`begin_step`/`prepare_next_pass` as required methods. `
FusedAdafactorOptimizerNode.build()` calls `register_hooks()` as its last
step automatically -- in the old code this is a required, easy-to-forget
manual call site gated behind an `isinstance` check in `trainer.py`; here
the produced Handle is simply always ready to use the moment `build()`
returns, matching every other adapter in this package.

**Does this mean "Fused CAME" (or any other future fused algorithm) is now
possible?** Honest answer, since overselling this would defeat the purpose
of writing it down: **the interface doesn't rule it out, but this work
doesn't unlock it by itself.** `_update_param` is not a generic
"hook-dispatch + pluggable per-algorithm math" architecture -- it's a
single monolithic method with Adafactor's entire algorithm (row/col
second moments, the small-parameter path, gradient clipping, all of it)
hard-coded inline, using instance state (`self.vr`, `self.vc`,
`self._tiny_vs_map`, ...) specific to that algorithm. A "Fused CAME" would
mean writing a genuinely new `core/optimizers.py` class -- CAME's math
(already verified in `ChunkedXPUCAME`) restructured to run per-single-
-parameter inside a backward hook, with its own accumulation/sub-step
bookkeeping mirroring `_in_backward`/`_current_sub_step`/
`sub_steps_required` -- a real, substantial algorithm-engineering task, not
adapter/plumbing work. What *is* true, and worth having done this for:
once such a class existed, it could plug into the exact same
`FusedOptimizerHandle` contract cleanly (the contract only describes the
fused *execution protocol*, nothing Adafactor-specific about its method
signatures), and a `FusedCAMEOptimizerNode` adapter wrapping it would be a
close copy of `fused_adafactor.py` -- the interface groundwork for that
future is real, even though the algorithm itself isn't written.

## Algorithm/ExecutionStrategy: a genuinely reconsidered design, not a wrapper

Raised directly after the `nodes/optimizer/` wrapper work above: wrapping
`core/optimizers.py`'s 5 legacy classes was the right *first* move (zero
risk to already-verified code, fast, and it found real bugs -- see above),
but the `OptimizerHandle`/`OptimizerNode` contracts it produced were shaped
by *reverse-engineering what the 5 existing classes happen to expose*, not
by first-principles design. Left as the permanent architecture, that would
mean the actual duplication and inconsistency living inside
`core/optimizers.py` never gets fixed, just hidden behind a clean facade --
worth naming plainly rather than treating the wrapper as a finished state.

**The reconsidered design, arrived at by asking what these 5 classes are
actually doing, not how they're currently organized:** they're 2 algorithms
(Adafactor, CAME) crossed by hand with up to 3 execution/memory strategies
(chunked-scratch-buffer, foreach-vectorized, fused-backward-hook) -- 4
classes exist because CAME only got 1 of the 3 possible strategies, each
combination being expensive to hand-write. Splitting **Algorithm** (pure
per-parameter math: grad + state -> update; zero knowledge of GPU memory,
batching, or hooks) from **ExecutionStrategy** (how/when that math actually
runs, how memory is managed; zero knowledge of *which* algorithm's math
it's running) turns an M-times-N hand-written grid into M-plus-N composable
pieces. Checked this holds up, not just asserted:

- **Is the "tiny parameter batching" trick (`_tiny_vs`/`_tiny_vs_map` in
  the legacy classes) actually an execution-strategy concern, not an
  algorithm one?** Yes -- confirmed by reading it: it changes how many
  actual tensors get allocated to track many small parameters' state
  efficiently, never what update formula gets computed. Fits cleanly as
  strategy-layer machinery (not yet built in this slice's
  `SimpleLoopStrategy`, but doesn't conflict with the split).
- **Does this actually make "Fused CAME" cheap, or just theoretically
  possible?** Once a `FusedBackwardHookStrategy` exists (driving *any*
  Algorithm's `compute_update()` from within a per-parameter backward
  hook, exactly mirroring `FusedXPUAdafactor`'s real hook-registration
  mechanism but generically), `Optimizer(CAMEAlgorithm(), 
  FusedBackwardHookStrategy())` is a composition of two independently-
  already-existing, independently-testable pieces -- not a new monolithic
  class. Not built yet in this slice (a real, separate, buildable next
  step), but the path there is now concrete instead of hopeful.

### First vertical slice: built and verified end-to-end

`nodes/optimizer/algorithms/base.py` (`Algorithm` ABC) +
`algorithms/came.py` (`CAMEAlgorithm`, a **fresh, from-scratch
reimplementation** -- not a wrapper around `ChunkedXPUCAME`) +
`strategies/base.py` (`ExecutionStrategy` ABC) + `strategies/simple.py`
(`SimpleLoopStrategy` -- deliberately the least sophisticated strategy
possible: a plain Python loop, no scratch-buffer/MemPool/vectorization/
hook-fusion yet, chosen specifically to prove composition works before
adding memory-optimization complexity) + `composed.py`
(`ComposedOptimizerHandle` -- the actual payoff: `offload_states_to_cpu`/
`reload_states_to_device`/`decay_states`/`reset_states`/`free_states`
written **exactly once**, generically over any Algorithm+Strategy pair,
instead of hand-duplicated 5 times with real, found-by-testing
inconsistencies) + `composed_came.py` (`ComposedCAMEOptimizerNode`, kept as
a separate class alongside the already-shipped `CAMEOptimizerNode` rather
than replacing it -- see below for why).

**Verification, in increasing order of rigor:**
1. `CAMEAlgorithm`'s formulas re-verified numerically against the official
   reference (github.com/yangluo7/CAME) in their new pure-function shape --
   found and precisely characterized a real, bounded, harmless discrepancy
   (~1e-8 absolute / ~4e-10 relative, from an intentional eps-after-sqrt
   denominator-safety term the bare reference formula doesn't have; jumps
   once then stays flat across further steps, not compounding -- both
   orders of magnitude below fp32 precision, which is what this actually
   runs at). Documented precisely in `algorithms/came.py` rather than
   claimed as a perfect match it isn't.
2. **End-to-end, through the real unmocked code**: built a minimal
   numpy-backed fake tensor supporting exactly the operations this code
   calls, and ran a genuine gradient-descent toy linear-regression fit
   through `CAMEAlgorithm` + `SimpleLoopStrategy` + `ComposedOptimizerHandle`
   exactly as shipped. Loss dropped 99.8% over 200 steps (2.495 -> 0.0039)
   -- this is the strongest verification possible without real hardware:
   not "the formulas match in isolation" but "the composed optimizer
   actually trains something." All lifecycle methods (`decay_states`,
   `reset_states`, `offload_states_to_cpu`, `reload_states_to_device`,
   `update_lr`, `free_states`) also verified to run correctly against the
   real composed object in the same test.

**Honest status, not oversold:** this proves the Algorithm/ExecutionStrategy
split is sound and genuinely composable -- it does not yet match
`core/optimizers.py`'s real classes on VRAM efficiency (`SimpleLoopStrategy`
has none of the memory optimizations those classes need to actually fit
this project's hardware budget) or run on real Intel Arc hardware (only
verified via the numpy-backed fake tensor above). `ComposedCAMEOptimizerNode`
is deliberately a separate class from the shipped `CAMEOptimizerNode`, not
a replacement -- switching over needs a memory-optimized strategy built
first, plus real-hardware validation, both genuine next steps rather than
this session's claim.

### Second strategy: ChunkedScratchBufferStrategy, and the two-axis distinction

Before building "the memory-optimized strategy," worth naming a real
design tension found while actually designing it rather than assumed
going in: buffer reuse has two genuinely separate axes.

1. **Across parameters** -- one shared buffer instead of N per-parameter
   allocations. Purely a strategy-layer concern; doesn't need the
   Algorithm's cooperation.
2. **Within one parameter's computation** -- this session's earlier,
   carefully-verified `core/optimizers.py` fix (`res` and `update` reusing
   the same buffer instead of allocating fresh tensors) requires the
   *algorithm's own* internal math to be restructured with in-place ops.
   Necessarily algorithm-specific engineering, not something a generic
   strategy wrapper can bolt on from outside.

Built axis 1 (`ChunkedScratchBufferStrategy`) this round; extended
`Algorithm.compute_update()` with an optional, backward-compatible
`scratch` parameter as the seam for axis 2 later, but `CAMEAlgorithm`
explicitly does not use it yet -- documented plainly in the code rather
than silently ignored, since getting in-place buffer reuse subtly wrong
(aliasing a value still needed) was a real bug this session already had
to catch and fix once.

**Verified**: bit-exact match (`0.000e+00` max diff) between
`SimpleLoopStrategy` and `ChunkedScratchBufferStrategy` across 60 real
training steps on two differently-shaped parameters (exercising the
scratch buffer's per-parameter reshape/slice logic) -- confirms this is
purely a memory optimization, zero behavior change. `ComposedCAMEOptimizerNode`
now takes a `strategy` input (`"simple"` or `"chunked"`) making the
Algorithm/Strategy swap real, usable value rather than a design claim --
switching it changes nothing about `CAMEAlgorithm` at all.

**What this strategy does NOT do yet** (see `strategies/chunked.py`'s
module docstring for the precise list): no `torch.xpu.MemPool`
integration yet, and since `CAMEAlgorithm` doesn't use the `scratch` hint
yet, the *only* real memory saving right now is the gradient-cast reuse --
real, but partial, stated precisely rather than implied to be the full
legacy-equivalent optimization. The buffer *is* now cached across
`step()` calls rather than allocated fresh each time -- see "Centralized
memory management" below for how and why.

**Smoke test extended** (`nodes/smoke_tests/smoke_test_composed_came.py`)
to run all real-hardware checks against every registered strategy, not
just `"simple"` -- `--strategy chunked` to check just one, `--strategy all`
(default) to check both in one run. `"simple"` already confirmed passing
on real XPU hardware by the user (toy regression converged 97.7%,
offload/reload round trip and all lifecycle methods correct);
`"chunked"` verified equivalent via the numpy-backed check above but not
yet run on real hardware -- next concrete step is getting that
confirmation.

### Centralized memory management: MemoryManager

`ChunkedScratchBufferStrategy`'s own module docstring already named two
real gaps in its first version: no cross-`step()` buffer caching (fresh
`torch.empty()` every call), and no `torch.xpu.MemPool` integration. The
straightforward fix for the first gap -- give the strategy its own
`self._scratch` attribute, cache the tensor there, remember to clear it
in `offload_extra()`/`free_extra()` -- is exactly the shape of the bug
this package's design already had to correct once: the "Course
correction" section above documents a real reset-vs-free asymmetry
(`_scratch`/`_pool` cleared in `free_states()` but not `reset_states()`
in some of `core/optimizers.py`'s legacy classes) that came from several
classes each hand-managing their own scratch-buffer attributes across
multiple lifecycle hooks. That bug class isn't about the old code's
specific shape -- it's what happens whenever a cached buffer's cleanup
is spread across N call sites instead of living in one place.

So instead of a strategy-local attribute, this round adds
`nodes/memory/manager.py`'s `MemoryManager`: a small, domain-independent
class (lives next to `nodes/core.py`, not under `nodes/optimizer/` --
nothing about it is optimizer-specific) that owns named ("tagged")
device buffers and exposes exactly three operations, kept deliberately
separate rather than collapsed into one:

- `get_buffer(tag, numel, dtype, device)` -- acquire-or-reuse, growing
  (never shrinking) if the tag's existing buffer is too small. Raises
  `RuntimeError` if the tag is already marked in-use, rather than
  silently handing out aliased storage -- this is a real, if narrow,
  version of the exact aliasing-bug class that `algorithms/came.py`'s
  and `algorithms/base.py`'s docstrings already flag as a caught-and-
  fixed real bug from `core/optimizers.py`'s `ChunkedXPUCAME` work; the
  guard turns a future instance of that mistake into a loud error at the
  call site instead of silently-wrong numbers downstream.
- `release(tag)` -- mark a buffer no longer needed *this call*, but keep
  the allocation alive for next time. The common, cheap path, called
  from `ChunkedScratchBufferStrategy.step()`'s `finally` block.
- `free(tag)` / `free_all()` -- actually drop the reference so device
  memory can be reclaimed. `ChunkedScratchBufferStrategy.offload_extra()`
  and `free_extra()` both call `free_all()` -- explicitly, in one place,
  rather than each lifecycle hook needing its own reminder to clean up.

Deliberately **not** a global singleton or module-level registry --
each `ChunkedScratchBufferStrategy` constructs and owns its own
`MemoryManager` instance by default (matching `ComposedOptimizerHandle`'s
existing pattern of each handle owning its own strategy instance), but
the constructor accepts an injected instance too, for a future caller
that legitimately wants several strategies or handles sharing one memory
budget. No implicit global state either way -- consistent with this
package's broader avoidance of implicit, sniff-the-object patterns (see
`handle.py`'s `update_lr()` docstring for the canonical example of the
pattern this avoids).

This is also, deliberately, a general-purpose piece rather than an
optimizer-only one. `docs/suspicious_findings.md` separately tracks a
still-unresolved ~500MB VRAM growth after preview generation, in
completely unrelated code (`preview_sampler.py`/`trainer.py`, nothing to
do with `nodes/` or optimizers) -- not touched by this work, and not
claimed to be fixed by it. But it's the kind of problem a shared,
reusable buffer-lifecycle class is aimed at in general: if `nodes/` ever
grows a non-optimizer domain that needs scratch device buffers, it has
somewhere real to plug into instead of reinventing ad hoc caching again.

**What this does NOT do** (real, separate, honestly-scoped follow-up):
no `torch.xpu.MemPool` integration yet -- `get_buffer()`'s single
`torch.empty()` call is exactly the seam that would wrap later, without
any caller needing to change. No automatic eviction under memory
pressure -- a manager holds whatever it's told to hold until explicitly
released/freed, so behavior stays predictable rather than depending on
runtime memory conditions. Doesn't touch axis 2 of the two-axis
distinction above (`Algorithm`-internal scratch reuse) -- `CAMEAlgorithm`
still doesn't use its `scratch` parameter, unchanged by this work.

**Verified**: `nodes/smoke_tests/smoke_test_memory_manager.py` (real
torch tensors, CPU -- see the file for why CPU is sufficient: no
device-specific code path exists in `manager.py` at all, so this isn't a
"hasn't been hardware-tested" gap the way `ChunkedScratchBufferStrategy`
itself has) checks reuse/growth via real `.data_ptr()` identity,
double-acquire raising, `free()`/`free_all()` actually dropping entries,
and `stats()`'s byte accounting -- all pass. `smoke_test_composed_came.py`
gained a new check `[4]` (chunked strategy only) confirming, through the
real composed handle rather than the manager in isolation: the scratch
buffer's `.data_ptr()` is identical across two separate `step()` calls
(genuine caching, not just "no crash"), and `stats()['total_bytes']`
drops to exactly `0` after `offload_states_to_cpu()` (the asymmetry this
was built to prevent, checked directly rather than assumed fixed) --
also passes, on CPU. Note this is real torch, not the numpy-backed fake
tensor earlier pieces of this package were verified with -- torch became
installable in this session's environment, which is itself a small,
genuine upgrade in verification strength over the mock-based approach
used for the algorithm/strategy work above.

## What changes for the playground UI

**Done.** The `/nodegraph` page's introspection moved from *guessing* ports
via `inspect.signature()` on old `core.optimizers` classes to *reading* the
real, declared `INPUTS`/`OUTPUTS` off the new `nodes/optimizer/` classes
directly (`nodegraph_introspect.py`'s new `introspect_node_class()` /
`introspect_optimizer_nodes()`, replacing the old `introspect_optimizers()`
-- the guess-based `introspect_legacy_class()`, renamed from
`introspect_class()`, stays available for any future domain not yet
migrated into `nodes/`). Concrete, verified improvements this produced,
not just a refactor for its own sake:

- Every node's card now shows its real inheritance chain (`extends:
  OptimizerNode → Node`), read from `cls.__mro__` -- an actual fact about
  the class, not inferred or hand-labeled.
- `FusedAdafactorOptimizerNode` correctly displays a `FusedOptimizerHandle`
  output type, distinct from the other four's plain `OptimizerHandle` --
  the graph now visibly reflects a real subtype relationship in the type
  system, which the old guess-based introspection had no way to represent
  at all (it could only ever report "produces an instance of this same
  class").
- The endpoint no longer needs torch importable at all (a genuine, if
  incidental, improvement): `nodes/optimizer/*.py` defer their
  `core.optimizers` imports to inside `build()`, and introspection only
  ever reads class-level `INPUTS`/`OUTPUTS`, never calls `build()`.

