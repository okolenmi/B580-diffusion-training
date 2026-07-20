# The `nodes/` package: design

Companion to `docs/node_architecture_refactor_plan.md` -- that document
covers the *why* and the multi-session migration strategy; this one covers
the concrete class design for the new `nodes/` package, written before
implementation.

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

