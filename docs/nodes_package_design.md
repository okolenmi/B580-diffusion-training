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
                                            device-buffer acquire/release/free, opt-in
                                            torch.xpu.MemPool integration (see "Centralized
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
nodes/optimizer/algorithms/adafactor.py    AdafactorAlgorithm -- FRESH reimplementation,
                                            re-verified against core.optimizers.ChunkedXPUAdafactor
                                            directly; scale_parameter=False, weight_decay=0
                                            only -- see "Third data point" section below
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
nodes/optimizer/composed_adafactor.py      ComposedAdafactorOptimizerNode -- Adafactor
                                            algorithm, selectable strategy, scale_parameter
                                            and weight_decay both fully supported (see
                                            "Universal Algorithm contract" section below),
                                            default off for safety, not legacy parity
nodes/smoke_tests/smoke_test_composed_came.py   Real-hardware test, run with:
                                            `python nodes/smoke_tests/smoke_test_composed_came.py`
nodes/smoke_tests/smoke_test_memory_manager.py  Real-torch (CPU) test for MemoryManager, run with:
                                            `python nodes/smoke_tests/smoke_test_memory_manager.py`
nodes/smoke_tests/smoke_test_adafactor_equivalence.py  Real-torch (CPU) numerical check
                                            against core.optimizers.ChunkedXPUAdafactor,
                                            full config surface (scale_parameter, weight_decay,
                                            momentum, both strategies, float32 and bf16)
nodes/smoke_tests/smoke_test_came_equivalence.py       Real-torch (CPU) numerical check
                                            against core.optimizers.ChunkedXPUCAME directly
                                            (first time -- the original CAME formula check
                                            used an external-reference numpy mock, not this)
nodes/smoke_tests/smoke_test_composed_adafactor.py     Real-hardware test, mirrors
                                            smoke_test_composed_came.py
nodes/smoke_tests/xpu_mempool_hardware_check.py        Real-XPU-only, heavier, run manually
                                            (excluded from run_all.py's glob on purpose):
                                            `python nodes/smoke_tests/xpu_mempool_hardware_check.py`
```
Also: `server/nodegraph_introspect.py` + `server/routes_nodegraph.py` +
`server/static/nodegraph.html` (the `/nodegraph` dev tab, reads real
declared contracts off these classes -- outside `nodes/` itself but part
of the same effort).

**Verification status, precisely (don't take "it's built" to mean "it's
proven" -- check which level a given piece has actually reached):**
| Piece | Numerical verification | Real XPU hardware verified |
|---|---|---|
| 5 legacy-wrapping wrapper Nodes (adafactor/came/foreach/fused/adamw) | via mock legacy classes | not yet |
| `CAMEAlgorithm` formulas | yes, numpy-mock vs. external reference (bounded ~1e-8, see below) **and** real torch vs. `core.optimizers.ChunkedXPUCAME` directly (`smoke_test_came_equivalence.py`, ~4e-6 float32) | via the composed test, yes |
| `SimpleLoopStrategy` | yes | **yes** -- user-run, passed (97.7% loss reduction, all lifecycle methods, offload/reload round trip) |
| `ChunkedScratchBufferStrategy` | yes, bit-exact vs. `SimpleLoopStrategy` | **yes** -- user-run on real XPU hardware, including the `MemoryManager`-backed version (97.7% loss reduction, all lifecycle methods, offload/reload round trip, and the new caching/cleanup check all passed -- see "Centralized memory management" below) |
| `MemoryManager` | n/a -- pure allocator logic, no device-specific code path exists | **yes** -- real torch (CPU) via `smoke_test_memory_manager.py`, all checks pass; also exercised indirectly on real XPU through `ChunkedScratchBufferStrategy`'s check `[4]` above |
| `AdafactorAlgorithm` formulas, full config surface (`scale_parameter`, `weight_decay`, momentum) | yes, real torch (CPU) directly against `core.optimizers.ChunkedXPUAdafactor` -- see `smoke_test_adafactor_equivalence.py` and "Universal Algorithm contract" below | **yes** -- user-run, `smoke_test_adafactor_equivalence.py` and `smoke_test_composed_adafactor.py` both pass on real XPU hardware, both strategies |
| `ComposedAdafactorOptimizerNode` (both strategies) | yes, real torch (CPU) toy regression + full lifecycle, `smoke_test_composed_adafactor.py` | **yes** -- user-run on real XPU hardware, both strategies |

**The one open architectural question -- now answered, precisely, not
just "yes it generalizes":** does the Algorithm/ExecutionStrategy split
generalize cleanly to a second, differently-shaped algorithm?
**Yes, once the contract was extended for real.** `AdafactorAlgorithm`
(see below) reuses `SimpleLoopStrategy` and `ChunkedScratchBufferStrategy`
completely unmodified, and its core math -- factored/non-factored second
moments, RMS clipping, momentum -- fit the original
`compute_update(grad, state, scratch)` contract exactly, needing only one
small, honest addition: `Algorithm.begin_step(n_steps)`, a default-no-op
lifecycle hook for algorithms with once-per-real-step (not
once-per-parameter) bookkeeping, which Adafactor's `rho_t` schedule
genuinely needs and CAME's fixed EMA betas don't (see
`algorithms/base.py`). A real boundary was found, not papered over --
Adafactor's `scale_parameter` mode and its weight-decay coupling both
needed `compute_update()` to know `lr` and the live parameter's own
current magnitude, which the original contract didn't provide -- and
resolved, not permanently deferred: `compute_update()` was extended to
receive both, returning `(delta, decay)` instead of a bare update. See
"Universal Algorithm contract" below for the extension and its
verification. The third strategy (`FusedBackwardHookStrategy`) and a
third algorithm are still unbuilt, so "generalizes cleanly" is proven for
one real second-data-point plus one real contract revision, not asserted
in general.

**Concrete next step, in order of what unblocks the most:**
1. [Done] `AdafactorAlgorithm`/`ComposedAdafactorOptimizerNode` run on
   real XPU hardware -- user-confirmed passing, both strategies, full
   `scale_parameter`/`weight_decay` surface. Both algorithms now have
   the same real-hardware verification level as CAME did alone before.
2. [Done] `torch.xpu.MemPool` integration -- opt-in
   (`MemoryManager(use_mempool=True)`), default off, real documented
   tradeoffs (see "Centralized memory management" below). Not yet
   confirmed on real XPU hardware by anyone.
3. **User's explicit direction: depth over breadth** -- fill out the
   full story for CAME and Adafactor (the two deferred items below)
   before starting a third algorithm or replacing either legacy wrapper
   for real use. Both of those remain real, open questions for later,
   not abandoned -- just deliberately not next.
4. `Algorithm.compute_update` actually using its `scratch` hint (axis 2
   of the two-axis distinction below) -- the higher-risk one, given the
   aliasing bug this project already caught once in
   `core/optimizers.py`'s `ChunkedXPUCAME`. Next up.
5. A `FusedBackwardHookStrategy` -- needs reading `FusedOptimizerHandle`
   and the legacy `FusedXPUAdafactor` first; not yet scoped in detail.
6. Longer-term, still not started: wiring `nodes/` into the actual
   training pipeline (`core/trainer.py` currently knows nothing about
   any of this).

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
module docstring for the precise list): `torch.xpu.MemPool` integration
now exists (opt-in, see "Universal memory pooling" note below in
"Centralized memory management"), but since `CAMEAlgorithm` still
doesn't use the `scratch` hint, the *only* real memory saving from this
strategy's own logic (independent of whether MemPool is turned on) is
still the gradient-cast reuse -- real, but partial, stated precisely
rather than implied to be the full legacy-equivalent optimization. The
buffer *is* now cached across `step()` calls rather than allocated fresh
each time -- see "Centralized memory management" below for how and why.

**Smoke test extended** (`nodes/smoke_tests/smoke_test_composed_came.py`)
to run all real-hardware checks against every registered strategy, not
just `"simple"` -- `--strategy chunked` to check just one, `--strategy all`
(default) to check both in one run. Both strategies now confirmed
passing on real XPU hardware by the user (toy regression converged
97.7% for both -- same seeded toy regression, so identical convergence
is expected -- offload/reload round trip and all lifecycle methods
correct for both; `"chunked"`'s run also included the `MemoryManager`
caching/cleanup check added below).

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

**What this still does NOT do** (real, separate, honestly-scoped
follow-up): no automatic eviction under memory pressure -- a manager
holds whatever it's told to hold until explicitly released/freed, so
behavior stays predictable rather than depending on runtime memory
conditions. Doesn't touch axis 2 of the two-axis distinction above
(`Algorithm`-internal scratch reuse) -- `CAMEAlgorithm` still doesn't use
its `scratch` parameter. This is the next item being worked on -- see
"Concrete next step" above.

**`torch.xpu.MemPool` integration -- added this round, opt-in.**
`get_buffer()`'s single `torch.empty()` call was always documented as
the seam this would wrap through later; `MemoryManager(use_mempool=
True)` now does that, routing allocations for XPU-device tags through a
per-device `torch.xpu.MemPool()` via `torch.xpu.use_mem_pool()`.
Confirmed against PyTorch's actual source
(`torch/xpu/memory.py`, `v2.11.0`) rather than assumed --
`torch.xpu.MemPool(allocator=None, use_on_oom=False)` and
`torch.xpu.use_mem_pool(pool, device=None)` are the real, current
signatures, not guessed at.

**Correction, made after the fact -- the user caught a real
mischaracterization here, re-checked directly, and fixed rather than
left standing.** This section originally cited two "documented
tradeoffs" from PyTorch's issue tracker: `pytorch/pytorch#161193`
(nesting `use_mem_pool` contexts) as a currently-open bug, and
`pytorch/pytorch#159674` (OOM-retry skipped inside `use_mem_pool`) as a
general MemPool tradeoff. Re-checked both directly after the correction:
`#161193` is actually **closed**, fixed in `CUDACachingAllocator.cpp` --
CUDA-specific code, not shared with XPU's own allocator implementation.
`#159674` is genuinely still open, but is tagged `module: cuda` and its
reproduction is `cudaMalloc`-specific -- real for CUDA, not confirmed
one way or the other for XPU (checked Intel's own `torch-xpu-ops` issue
tracker too; found nothing there either way). Both are now stated in
`nodes/memory/manager.py`'s docstring precisely: one closed and
CUDA-specific, the other open but CUDA-specific and unconfirmed for XPU
-- not asserted as XPU risks the way the first version of this section
implied. Worth naming plainly: two `web_fetch` calls to
docs.pytorch.org failed to return usable content earlier in the same
research pass that found these issues via `web_search` -- plausibly why
less scrutiny went into confirming these two citations than into the
API signatures above (checked against actual source). The fix is to
check harder next time, not to stop citing external sources.

Default `use_mempool=False` -- confirmed this changes nothing about
already-verified behavior (the full suite, including both composed
handle tests, still passes identically with the change in place).
`ChunkedScratchBufferStrategy.__init__` gained a matching `use_mempool`
pass-through, also defaulted off.

**A real gap in my own first validation attempt, caught before
shipping:** initially gated `use_mempool=True` behind
`hasattr(torch.xpu, "MemPool")`, reasoning that would catch "this torch
build doesn't support XPU." Wrong -- checked directly in this session's
own (CUDA-only) sandbox build: `hasattr` returns `True` (the class
exists at the Python level) but `torch.xpu.MemPool()` raises `"Tried to
instantiate dummy base class MemPool"`, since no real XPU backend is
compiled in. Fixed to check `torch.xpu.is_available()` instead, the
actually-correct gate -- verified this now raises a clear error in
exactly the scenario the wrong check would have let through silently
until first real use.

**Verified on real XPU hardware by the user (Intel Arc B580) -- all
three pass/fail checks passed, and the two diagnostic ones are worth
reading precisely:**
- `[1]`/`[2]`/`[3]` (construction, bit-exact correctness vs.
  `use_mempool=False`, `free_all()` actually returning real device
  memory via `torch.xpu.memory_allocated()`): all PASS.
- `[4]` (fragmentation comparison, diagnostic): `reserved`/
  `peak_reserved` came back **identical** with and without MemPool for
  this workload (`allocated=172210176, reserved=314572800,
  peak_reserved=314572800` both ways). Doesn't confirm the
  fragmentation-reduction claim for this particular allocation pattern --
  stated precisely rather than claimed as a win, since the numbers
  simply don't show a difference here, not because MemPool failed at
  anything.
- `[5]` (`--stress`, OOM-retry closer look): both configurations
  succeeded allocating the same ~6.8GB target with no difference --
  inconclusive for the `#159674`-style tradeoff at this budget fraction
  on this hardware, not evidence either way.

A real, separate, more practically important issue the user reported
from actual ComfyUI use (not this file, not this session's work) while
testing this: intermittent "Device lost" errors and silent hangs after
VRAM-pressure events (specifically: after a VAE decode spike during
preview generation, or after model-merge-then-generate workflows) --
described as looking like something gets offloaded under memory pressure
but isn't correctly loaded back despite free VRAM being available
afterward. Not investigated or fixed here -- out of scope for this
session's `nodes/`-only work, and this is core/trainer.py territory, not
this package's. Worth a pointer for whoever picks this up: a
`kohya-ss/musubi-tuner` discussion about Wan2.2 training on the same B580
hardware describes a matching hang-after-offload symptom, traced there
to a `synchronize_device()` call missing its `device` argument on the
non-CUDA path -- a plausible root-cause *shape* (an async/non-blocking
transfer without a matching explicit synchronize on the XPU path) worth
checking `core/trainer.py`'s own offload code for, not a confirmed
diagnosis of this project's issue.

`nodes/smoke_tests/xpu_mempool_hardware_check.py` needed one fix after
being written blind: `torch.xpu.memory_allocated()`-style calls reject a
bare `"xpu"` device string (`ValueError: Expected a torch.device with a
specified index or an integer, but got: xpu`) -- the user fixed this by
setting `DEVICE = 0`, confirmed working. Not something CPU-only
development could have caught -- this sandbox's dummy XPU backend never
reaches the code path that validates this.

**Verified** (buffer bookkeeping itself, unrelated to MemPool):
`nodes/smoke_tests/smoke_test_memory_manager.py` (real
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
passes on CPU, and **user-confirmed passing on real XPU hardware too**
(same two checks, same result, actual device). Note this is real torch
throughout, not the numpy-backed fake tensor earlier pieces of this
package were verified with -- torch became installable in this session's
environment, which is itself a small, genuine upgrade in verification
strength over the mock-based approach used for the algorithm/strategy
work above.

### Third data point: AdafactorAlgorithm, and the boundary it found

`AdafactorAlgorithm` (`nodes/optimizer/algorithms/adafactor.py`) is the
second `Algorithm` built in this package -- the actual test of the "one
open architectural question" above. Reference is this codebase's own
`core.optimizers.ChunkedXPUAdafactor` directly (not an external repo --
see the module docstring for why that's the right reference here), a
fresh reimplementation verified step-by-step against it with real torch.

**What generalized cleanly, with zero interface changes:** the
factored/non-factored second-moment math, RMS clipping, and momentum all
fit `compute_update(grad, state, scratch)` exactly as already defined.

**What needed one small, honest addition:** Adafactor's `rho_t` schedule
is a single, monotonically increasing value shared across every
parameter in a step -- genuinely different from anything CAME needed
(fixed EMA betas, no schedule at all), and `compute_update()` (called
once *per parameter*) has no way to advance a step counter exactly once
*per real step* on its own. Added `Algorithm.begin_step(n_steps)` to the
ABC: a default no-op (so `CAMEAlgorithm` needed no change at all), called
once by each `ExecutionStrategy.step()` before its per-parameter loop.
Small, mechanical, and precedented -- the same "default no-op hook,
overridden only where needed" shape `ExecutionStrategy`'s own
`offload_extra`/`reload_extra`/`free_extra` already use.

**What did NOT generalize, found rather than assumed away:** Adafactor's
`scale_parameter=True` mode (the legacy default) computes its effective
step size from `clamp(param_rms**2, min) * lr` -- genuinely dependent on
both `lr` and the *live parameter's own current magnitude*, neither of
which the contract passes to `compute_update()`. Its weight decay
(`p *= 1 - wd*alpha_t`) is coupled to that same `alpha_t` and is a
multiplicative rescale of the live parameter, not an additive delta the
contract can express at all. `AdafactorAlgorithm`'s first slice
implements only the `scale_parameter=False, weight_decay=0` case, where
the effective step size reduces to exactly `lr` (for any realistic
`eps1 < 1`) -- fitting the unmodified contract precisely. This is a
real, user-facing limitation: `ComposedAdafactorOptimizerNode` is **not**
a drop-in replacement for `AdafactorOptimizerNode`'s default
configuration (`scale_parameter=True, weight_decay=1.0`), and it says so
explicitly rather than silently ignoring those inputs -- see
`composed_adafactor.py`'s module docstring. Two real paths forward,
neither taken yet, deliberately surfaced rather than picked by guessing:
extend `compute_update()` to receive `lr` and the live parameter (a real
contract change, second thing to need it after CAME might make a
pattern clearer), or handle weight decay via a separate, generic
strategy-level hook and treat `scale_parameter` as staying out of scope
longer-term. Worth a real decision with the user, not a default.

**A genuine, unrelated finding surfaced along the way:** an early
verification pass (small, "toy-sized" test parameters) showed huge
discrepancies that turned out to be comparing against the wrong code
path entirely -- `ChunkedXPUAdafactor` silently routes any parameter
under 10,000 elements through a completely different tiny-parameter
batching fast path, not the per-parameter path this class ports. Caught
by checking `legacy.vr[0] is None` after a step that should have
populated it, not assumed. Separately, testing the momentum (`beta1`)
case with float32 parameters surfaced a real, previously-undocumented
aliasing bug in the legacy reference itself: `p.data.sub_(g.to(dtype=
p.dtype).mul_(alpha_t))`, where `g` is `self.exp_avg[i]` -- when a
parameter's dtype equals the state's float32 dtype exactly, `.to(dtype=
p.dtype)` is a no-op returning the *same tensor* (confirmed directly:
`t.to(dtype=t.dtype) is t` -> `True`), so the following `.mul_()`
permanently corrupts the momentum buffer in place, every step. Doesn't
affect real training (bf16 parameters always get a fresh tensor from
`.to()`, confirmed by re-running the same comparison under bf16 and
seeing the divergence collapse to ordinary quantization noise) -- logged
as a new, informational entry in `docs/suspicious_findings.md` rather
than fixed (`nodes/` doesn't touch `core/`), since it's real but not
urgent and this isn't the place to fix it.

**Verified**: `nodes/smoke_tests/smoke_test_adafactor_equivalence.py`
(real torch, CPU, against `core.optimizers.ChunkedXPUAdafactor` directly
via a standard import -- this codebase's own `fastapi`/`pydantic`
dependencies make that importable without any workaround) -- both
strategies, with and without momentum, factored and non-factored
parameters, all within documented, honestly-derived tolerances (float32,
no momentum: ~2e-6 max abs diff, essentially exact; bf16, with and
without momentum: ~4-9e-3, consistent with ordinary bf16 rounding noise
compounding over 40 steps, checked to grow mildly and sub-linearly
rather than exploding). `nodes/smoke_tests/smoke_test_composed_adafactor.py`
mirrors `smoke_test_composed_came.py`'s toy-regression + full-lifecycle
check (both strategies) -- passes on CPU. Neither yet run on real XPU
hardware by the user -- see "Concrete next step" above.

**Update, next session: the boundary above was resolved, not left as a
permanent limitation** -- see "Universal Algorithm contract: lr, param,
and decoupled decay" below. Left as-is above rather than rewritten, since
it's an accurate record of the reasoning that led to the extension, not
a stale claim.

### Universal Algorithm contract: lr, param, and decoupled decay

Prompted by a real user report: `scale_parameter=True` appeared to make
almost no training progress. Investigated rather than assumed --
`scale_parameter`'s effective step size is `clamp(param_rms**2, min=
~1e-6) * lr`; for a parameter initialized at or near zero (LoRA's B
matrix, by convention, starts at exactly zero), `param_rms` starts at the
floor, so `alpha_t` collapses to roughly a millionth of plain `lr`
(confirmed directly: for the default `eps`, `alpha_t ≈ 1e-6 * lr`). Since
updates then stay tiny, the parameter stays near zero, so `alpha_t` stays
near the floor -- a self-reinforcing near-standstill. Not a bug: verified
both this port and the legacy reference reproduce the identical floor
value. Very likely the actual explanation for what was observed.

Given that, and an explicit ask to make the underlying design "more
versatile/universal" rather than special-cased, `Algorithm.compute_update()`
was extended for real (see `algorithms/base.py`'s module docstring for
the full contract): it now receives `param` (read-only) and `lr`, and
returns `(delta, decay)` instead of a bare "unit" update --
`delta` is the final, already-`lr`-scaled amount to subtract; `decay` is
`None` or a multiplicative factor an `ExecutionStrategy` applies to
`param.data` *before* subtracting `delta`, matching every legacy
optimizer's own decay-then-step order. `Algorithm` still never mutates
`param` directly -- `decay` is a description, not an action -- keeping
the "pure math, only `state` is mutated in place" property intact even
though `param` is now visible to it.

**What this unlocked, concretely:**
- `AdafactorAlgorithm` now implements `scale_parameter` (both settings)
  and `weight_decay` for real -- no longer a documented gap.
  `ComposedAdafactorOptimizerNode` exposes both as ports, but
  deliberately defaults them off (`False`/`0.0`) rather than matching
  `AdafactorOptimizerNode`'s unusual legacy defaults
  (`scale_parameter=True, weight_decay=1.0`, the latter of which shrinks
  any parameter ~5%/step at typical lr) -- changing this Node's defaults
  to match would have silently changed already-tested toy-regression
  behavior. Pass both explicitly to get legacy parity; verified to do so
  exactly (see below).
- `CAMEAlgorithm` gained `weight_decay` support too, essentially for
  free -- it never needed `lr`/`param` for its own math, but folding
  `lr` into its own return value and adding decay through the same
  generic mechanism cost almost nothing once the contract existed. Real,
  working feature parity CAME's port didn't have before this session.
  `ComposedCAMEOptimizerNode` now exposes `weight_decay` (default `0.0`,
  matching the legacy wrapper's own default exactly -- no behavior
  change for existing callers who don't pass it).
- Both `SimpleLoopStrategy` and `ChunkedScratchBufferStrategy` updated
  to the new call site (`delta, decay = algorithm.compute_update(grad, p,
  states[i], param_lr[i], scratch=...)`, then `if decay is not None:
  p.data.mul_(decay)` before `p.data.sub_(delta)`) -- mechanical, same
  shape for both, no strategy needs to know which algorithms use `decay`
  and which don't.

**Verified, stated precisely:** extended
`smoke_test_adafactor_equivalence.py` to cover the full configuration
surface against `core.optimizers.ChunkedXPUAdafactor` directly --
`scale_parameter` on and off, `weight_decay` on and off (including the
legacy's own default combination, `scale_parameter=True, weight_decay=
1.0`), momentum on and off, both strategies, float32 and bf16. All match
to float32 precision in float32 (~6e-8 to ~2e-6 depending on
configuration) and to ordinary bf16 rounding-noise levels in bf16 --
`scale_parameter=True`'s own formula checked in the well-behaved
(non-zero-init) regime specifically, so this confirms the `p_rms`-based
math itself is faithfully ported, not just the degenerate case. New file
`smoke_test_came_equivalence.py` does the same for `CAMEAlgorithm`
against `core.optimizers.ChunkedXPUCAME` directly -- the first time this
class was checked against the *in-repo* legacy reference with real
tensors, rather than the original external-reference numpy comparison.
float32 matches to ~4e-6 (with or without decay -- confirmed identical,
so decay itself adds no error); bf16 shows a real, bounded, mildly-
growing divergence (~4e-3 to ~5e-2 over 40 steps) present identically
with `weight_decay=0`, so unrelated to this session's change -- likely
from CAME's own longer chain of sqrt/divide operations compounding bf16
rounding more per step than Adafactor's simpler one. Noted as a new data
point, not chased further this session.

**A test-methodology bug was also found and fixed along the way,
unrelated to the contract extension itself:** running the new
`smoke_test_composed_adafactor.py` on real XPU hardware produced a false
failure -- `AdafactorAlgorithm`'s gradient-normalized update doesn't
shrink its own step size as loss shrinks (unlike raw-gradient SGD), so
once training converges very tightly (it reached ~1e-6 loss on this toy
problem on the user's XPU run), it keeps taking full-sized steps and
visibly oscillates around the optimum -- confirmed this happens with
*zero* offload/reload involved at all, just continued training, by
direct investigation. The test's old check compared resumed loss against
the loss immediately before offload, which happened to be sampled right
at a lucky near-zero point -- comparing noise to noise. Fixed in both
`smoke_test_composed_came.py` and `smoke_test_composed_adafactor.py`
two ways: (1) added a direct, much stronger check -- snapshot state
before offload, compare with `torch.equal()` after reload, independent
of any downstream training dynamics at all; (2) changed the training-
continues check to compare against the *original* starting loss rather
than the fragile near-offload value. Confirmed the fix still catches a
real corruption (simulated by substituting `reset_states()` for the
round trip) before committing it, and confirmed it doesn't change what
CAME's own already-passing real-XPU numbers report.

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

